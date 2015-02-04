import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

default_cmap_table = plt.get_cmap('YlGn')(np.linspace(0.03, 0.6, 5))[:, :3]

maxscore = 10.


class HotSpotData(object):

    """
    A data object for holding site and resource data for performing
    site-scoring, ranking and other analysis operations.
    """

    def __init__(self, data, resdata):
        self.data = data
        self.resdata = resdata

    def __getitem__(self, val):
        return self.__class__(self.data.iloc[val].copy(), self.resdata)

    def to_excel(self, fname):
        """
        Write the data in this data object to a Microsoft Excel file (.xlsx).

        Parameters
        ----------

        fname : string
                The filename to write to.

        Notes
        -----
        If fname does not end in .xlsx or .xls, '.xlsx' will be appended
        to the file name.

        """
        if basestring in fname.__class__.__mro__:
            if not (fname.endswith('.xlsx') or fname.endswith('.xls')):
                fname += '.xlsx'
            fname = pd.io.excel.ExcelWriter(fname)
        self.data.to_excel(fname, sheet_name='SiteData')
        self.resdata.to_excel(fname, sheet_name='ResourceData', index=False)
        fname.close()

    def write_table_legend(self, fname='default_legend.html',
                           colors=default_cmap_table):
        colors = np.array(colors) * 255
        ncolor = colors.shape[0]
        if np.mod(maxscore / ncolor, 1) == 0:
            frmt = '%d'
        else:
            frmt = '%0.1f'
        vls = np.linspace(maxscore / ncolor, 10, ncolor)
        dat = np.array(['(' + frmt + ' - ' + frmt + ']'] * ncolor, dtype='S10')
        for idx, vl in enumerate(vls):
            dat[idx] = dat[idx] % (vl - maxscore / ncolor, vl)
        dat[0] = dat[0].replace('(', '[')
        dat = dat[::-1]
        tbl = pd.DataFrame({'Score': dat}).to_html(index=False)
        tbl = tbl.replace(r'border="1" ', 'style="border-collapse: collapse; text-align:center;"')
        tbl = tbl.replace('<td', '<td style="%s"')
        out = np.empty(ncolor, dtype='O')
        for idx in xrange(ncolor):
            out[idx] = 'background-color:rgb(%d, %d, %d)' % tuple(colors[idx])
        out = out[::-1]
        tbl = tbl % tuple(out)
        if not (fname.endswith('.htm') | fname.endswith('.html')):
            fname += '.html'
        with open(fname, 'w') as f:
            f.write(tbl)

    def to_html(self, fname,
                columns=['energy_cost', 'load', 'dist',
                         'resource', 'depth', 'shipping', 'score_total'],
                colors=default_cmap_table,
                hline_widths=[2, 1],
                ):
        colors = np.array(colors) * 255
        dat = self.data[columns]
        tbl = dat.to_html(float_format=lambda s: ('%0.1f' % s))
        tbl = tbl.replace(r'border="1" ', 'style="border-collapse: collapse;"')
        tbl = tbl.replace('<td', '<td style="%s"')
        out = np.empty_like(dat, dtype='O')
        out[:] = ''
        for idx in xrange(dat.shape[0]):
            for ic, col in enumerate(columns):
                if col.startswith('score_'):
                    col_score = col
                else:
                    col_score = 'score_' + col
                if col_score in self.data:
                        ind_clr = int((self.data[col_score][idx]) / maxscore * colors.shape[0])
                        ind_clr = min(ind_clr, colors.shape[0] - 1)
                        out[idx, ic] = 'background-color:rgb(%d, %d, %d)' % tuple(colors[ind_clr])
        tbl = tbl % tuple(out.flatten())
        if not (fname.endswith('.htm') | fname.endswith('.html')):
            fname += '.html'
        with open(fname, 'w') as f:
            f.write(tbl)

    def to_csv(self, fname, resource_fname=None):
        """
        Write the site-data in this object to a comma-separated-value (csv)
        file.

        Parameters
        ----------

        fname : string
                The filename to write to.

        resource_fname : string (optional)
                         If specified, the resource data will be
                         written to this file.

        Notes
        -----
        If the fname (or resource_fname) do not end in .csv, that file
        extension will be added to the file name.

        """
        if not fname.endswith('.csv'):
            fname += '.csv'
        self.data.to_csv(fname + '.csv',)
        if resource_fname is not None:
            if not resource_fname.endswith('.csv'):
                resource_fname += '.csv'
            self.resdata.to_csv(resource_fname, index=False)

    def __repr__(self,):
        return self.data.__repr__()

    def __copy__(self):
        return HotSpotData(self.data.copy(), self.resdata.copy())

    copy = __copy__

    def rank(self, clear_0_nan=True, ):
        """
        Sort the site-data by the 'score_total' column (secondary
        sorted by: load).

        The resource-data will be sorted by 'name', then 'score_total'
        then 'resource'.

        Parameters
        ----------

        clear_0_nan : {True*, False}
                      Specifies whether to clear site entries that
                      have a score of zero or NaN.

        """
        bdi = np.isnan(self.data['score_total']) | (self.data['score_total'] == 0)
        resbdi = np.isnan(self.resdata['score_total'])
        if clear_0_nan:
            self.data = self.data[~bdi]
            self.resdata = self.resdata[~resbdi]
        else:
            self.data['score_total'][bdi] = -1
            self.resdata['score_total'][resbdi] = -1
        # Sort the results:
        self.data = self.data.sort(['score_total', 'load'],
                                   ascending=[False, False])
        self.resdata = self.resdata.sort(['name', 'score_total', 'resource'],
                                         ascending=[True, False, False])
        if not clear_0_nan:
            self.data['score_total'][bdi] = np.NaN
            self.resdata['score_total'][resbdi] = np.NaN


def score_data(data, spec):
    """
    This function uses the score table (or array) `spec` to calculate
    the scores for the data points in `data`.
    """

    # This if statement 'block' makes sure `data` is in the right
    # 'format' (object type)
    if data.__class__ is pd.Series:
        data = data.values
    elif np.ndarray not in data.__class__.__mro__:
        if not np.iterable(data):
            data = [data]
        data = np.array(data)

    # First initialize the output array.
    # Values not specified in the table should be NaN.
    out = np.zeros(data.shape) * np.NaN

    # Now perform a 'for loop' over the min, max, score values:
    for mn, mx, score in spec:
        # Test that data is within the range, and assign scores:
        out[((mn < data) & (data <= mx))] = score
    return out  # Return the output data from the function.


class ScoreTable(object):

    def __init__(self, table, weight=1):
        if np.array not in table.__class__.__mro__:
            self.table = np.array(table)
        self.weight = weight

    def __copy__(self,):
        return self.__class__(self.table.copy(), weight=self.weight)

    copy = __copy__

    def __repr__(self,):
        outstr = 'WEIGHT: %4.2f\n------ Range --------- :  Score\n' % (self.weight, )
        for v in self.table:
            outstr += ' %9.1f - %9.1f : %5.1f\n' % tuple(v)
        return outstr

    def __call__(self, data):
        return score_data(data, self.table)

    def sumweight(self, data):
        return self(data) * self.weight

    def prodweight(self, data):
        return self(data) ** self.weight
