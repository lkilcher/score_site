import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mplc
import simplekml

default_cmap_table = plt.get_cmap('YlGn')(np.linspace(0.03, 0.6, 5))[:, :3] * 255
#default_cmap = plt.get_cmap('Reds')(np.linspace(0.00, 1, 5))[:, :3] * 255
default_cmap = plt.get_cmap('Reds')

maxscore = 10.


class HotSpot(object):

    @property
    def name(self,):
        return self.data.name

    @property
    def lon(self,):
        return self.data['lon']

    @property
    def lat(self,):
        return self.data['lat']

    @property
    def region(self,):
        return self.data['region']

    @property
    def shipping(self,):
        return self.data['shipping']

    @property
    def load(self,):
        return self.data['load']

    @property
    def energy_cost(self,):
        return self.data['energy_cost']

    coe = energy_cost

    def __init__(self, data, resdata, model=None):
        self.data = data
        self.resdata = resdata
        self.model = model

    def to_kml(self, buf,
               mapdata='resource',
               cmap=default_cmap,
               clims=[None, None]):
        if buf.__class__ in [simplekml.Kml, simplekml.Folder]:
            kml = buf
            fl = False
        else:
            kml = simplekml.Kml()
            fl = True
        pt = kml.newpoint(name=self.data.name, coords=[(self.lon, self.lat)])
        pt.lookat = simplekml.LookAt(
            gxaltitudemode=simplekml.GxAltitudeMode.relativetoseafloor,
            latitude=self.lat, longitude=self.lon, range=6.4E4, heading=0, tilt=0)
        pt.iconstyle.hotspot.x = 0.5
        pt.iconstyle.hotspot.xunits = 'fraction'
        pt.iconstyle.hotspot.y = 0.0
        pt.iconstyle.hotspot.yunits = 'fraction'
        pt.iconstyle.scale = 2
        pt.style.iconstyle.color = 'FF0000FF'  # red
        pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-blank.png'
        ed = simplekml.ExtendedData()
        ed.newdata('COE',
                   '{:0.3f}'.format(self.coe),
                   'Energy Cost ($/kWh)')
        ed.newdata('load',
                   self.load,
                   'Average Load (kW)')
        ed.newdata('shipping',
                   self.shipping,
                   'Shipping Cost ($/tonne)')
        pt.extendeddata = ed

        if clims is None:
            clims = [None, None]
        clims = list(clims)
        if clims[0] is None:
            clims[0] = self.resdata[mapdata].min()
        if clims[1] is None:
            clims[1] = self.resdata[mapdata].max()

        for idx, d in self.resdata.iterrows():
            pt = kml.newpoint(name='', coords=[(d.lon, d.lat)])
            pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-blank-lv.png'
            c = mplc.rgb2hex(cmap(float(
                (d[mapdata] - clims[0]) / (clims[1] - clims[0])
                )))
            c = 'FF' + c[5:] + c[3:5] + c[1:3]
            pt.style.iconstyle.color = c
            ed = simplekml.ExtendedData()
            for dval in ['lon', 'lat', 'depth', 'resource', 'dist', 'score_total']:
                if dval in d:
                    ed.newdata(dval + '_val',
                               '%0.2f' % d[dval],
                               dval)
            pt.extendeddata = ed
        if fl:
            kml.save(buf)


class HotSpotData(object):

    """
    A data object for holding site and resource data for performing
    site-scoring, ranking and other analysis operations.
    """

    def __init__(self, data, resdata, model=None):
        self.data = data
        self.resdata = resdata
        self.model = model

    def __getitem__(self, val):
        try:
            dtmp = self.data.iloc[val].copy()
        except KeyError:
            dtmp = self.data.loc[val].copy()
        except TypeError:
            dtmp = self.data.loc[val].copy()
        if dtmp.ndim == 1:
            return HotSpot(dtmp,
                           self.resdata.loc[self.resdata.name == dtmp.name],
                           model=self.model,)
        else:
            return self.__class__(dtmp, self.resdata, self.model)

    def to_kml(self, buf,
               mapdata='resource',
               cmap=default_cmap,
               clims=[None, None]):
        if buf.__class__ is simplekml.Kml:
            kml = buf
            fl = False
        else:
            kml = simplekml.Kml()
            fl = True
        # Handle the default clims:
        if clims is None:
            clims = [None, None]
        clims = list(clims)
        if clims[0] is None:
            clims[0] = self.resdata[mapdata].min()
        if clims[1] is None:
            clims[1] = self.resdata[mapdata].max()
        # Write the data for each location:
        for idx in xrange(self.data.shape[0]):
            dnow = self[idx]
            fol = kml.newfolder(name=dnow.name)
            dnow.to_kml(fol,
                        mapdata=mapdata,
                        cmap=cmap,
                        clims=clims)
            if idx == 0:
                vw = fol.allfeatures[0].lookat
        kml.document.lookat = vw
        if fl:
            kml.save(buf)

    def to_excel(self, buf):
        """
        Write the data in this data object to a Microsoft Excel file (.xlsx).

        Parameters
        ----------

        buf : string
              The filename or buffer to write to.

        Notes
        -----
        If buf does not end in .xlsx or .xls, '.xlsx' will be appended
        to the file name.

        """
        if basestring in buf.__class__.__mro__:
            if not (buf.endswith('.xlsx') or buf.endswith('.xls')):
                buf += '.xlsx'
            buf = pd.io.excel.ExcelWriter(buf)
            #buf = pd.io.excel.ExcelWriter(buf, 'openpyxl')
        self.data.to_excel(buf, sheet_name='SiteData')
        self.resdata.to_excel(buf, sheet_name='ResourceData', index=False)
        buf.close()

    def _write_table_legend(self, buf='default_legend.html',
                            colors=default_cmap_table,
                            ):
    ## def _write_table_legend(self, buf='default_legend.html',
    ##                         cmap=default_cmap_table,
    ##                         vals=[0, 2, 4, 6, 8, 10],
    ##                         ):
        """
        Write the table legend for the colormap specified by `colors`
        to `buf`. `colors` should be the same colormap used to write
        an html table using :meth:`to_html`.

        Parameters
        ----------

        buf : string

        """
        ncolor = colors.shape[0]
        if (maxscore / ncolor % 1) == 0:
            frmt = '%d'
        else:
            frmt = '%0.1f'
        vls = np.linspace(maxscore / ncolor, 10, ncolor)
        dat = np.array(['(' + frmt + ' - ' + frmt + ']'] * ncolor, dtype='S10')
        for idx, vl in enumerate(vls):
            dat[idx] = dat[idx] % (vl - maxscore / ncolor, vl)
        dat[0] = dat[0].replace('(', '[')
        dat = dat[::-1]
        tbl = pd.DataFrame({'Score': dat}).to_html(index=False) + '\n'
        tbl = tbl.replace(r'border="1" ',
                          'style="border-collapse: collapse; text-align:center;"')
        tbl = tbl.replace('<td', '<td style="%s"')
        out = np.empty(ncolor, dtype='O')
        for idx in xrange(ncolor):
            out[idx] = 'background-color:rgb(%d, %d, %d)' % tuple(colors[idx])
        out = out[::-1]
        tbl = tbl % tuple(out)
        if basestring in buf.__class__.__mro__:
            if not (buf.endswith('.htm') | buf.endswith('.html')):
                buf += '.html'
            with open(buf, 'w') as fl:
                fl.write(tbl)
        else:
            buf.write(tbl)

    def to_html(self, buf,
                columns=['energy_cost', 'load', 'dist',
                         'resource', 'depth', 'shipping', 'score_total'],
                colors=default_cmap_table,
                hline_widths=None,
                include_legend=True,
                weights_in_head=True,
                ):
        dat = self.data[columns]
        tbl = dat.to_html(float_format=lambda s: ('%0.1f' % s)) + '\n'
        tbl = tbl.replace(r'border="1" ', 'style="border-collapse: collapse;"')
        form = np.empty_like(dat, dtype='O')
        form[:] = ''
        tbl = tbl.replace('<th>',
                          '<th style="border-bottom:solid %0.1fpt;">' % hline_widths[0],
                          len(columns) + 1)  # +1 for index
        if self.model is not None and weights_in_head:
            for var, t in self.model.scorers.iteritems():
                tbl = tbl.replace(var, var + ' (%0.1g)' % (t.weight,), 1)
        for irow in xrange(dat.shape[0]):
            # This is for the first (index) column:
            if hline_widths is not None:
                lw_txt = ('border-bottom:solid %0.1fpt; ' %
                          hline_widths[(irow + 1) % len(hline_widths)])
            else:
                lw_txt = ''
            tbl = tbl.replace('<th>',
                              '<th style="text-align:left; %s">' % lw_txt,
                              1)
            form[irow] += lw_txt
            for icol, col in enumerate(columns):
                tbl = tbl.replace('<td>',
                                  '<td style="text-align:right; {form[%d][%d]}">' % (irow, icol),
                                  1)
                if col.startswith('score_'):
                    col_score = col
                else:
                    col_score = 'score_' + col
                if col_score in self.data:
                    ind_clr = int((self.data[col_score][irow]) / maxscore * colors.shape[0])
                    ind_clr = min(ind_clr, colors.shape[0] - 1)
                    form[irow, icol] += 'background-color:rgb(%d, %d, %d); ' % tuple(colors[ind_clr])
        tbl = tbl.format(form=form)
        if basestring in buf.__class__.__mro__:
            if not (buf.endswith('.htm') | buf.endswith('.html')):
                buf += '.html'
            with open(buf, 'w') as fl:
                fl.write(tbl)
                if include_legend:
                    self._write_table_legend(fl, colors=colors)
        else:
            fl.write(tbl)
            if include_legend:
                self.write_table_legend(fl)

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
