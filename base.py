import pandas as pd
import numpy as np


class HotSpotData(object):

    """
    A data object for holding site and resource data for performing
    site-scoring, ranking and other analysis operations.
    """

    def __init__(self, data, resdata):
        self.data = data
        self.resdata = resdata

    def __getitem__(self, val):
        out = self.data.loc[val]
        out.resdata = self.resdata[self.resdata.name == val]
        out.set_value('max_resource', max(out.resdata.resource))
        return out

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
