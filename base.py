"""
This module defines the base object types for score_site:

- HotSpotCollection: This is the primary data type of the score_site
  package. It contains the resource and site data for a collection of
  sites.

- HotSpot : This is a site and data class for a single 'hot spot'.

"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mplc
import simplekml
from os import remove, path

package_root = path.realpath(__file__).replace("\\", "/").rsplit('/', 1)[0] + '/'

sub_cmap = lambda cmap, low, high: lambda val: cmap((high - low) * val + low)

cmaps = {'green': sub_cmap(plt.get_cmap('YlGn', ), 0.1, 0.7),
         'red': plt.get_cmap('Reds'),
         }


maxscore = 10.0


def _write_kmz_legend(buf, title, values, cmap, clims):
    if values is False:
        return
    if values is None:
        values = np.arange(clims[1],
                           clims[0] - (clims[1] - clims[0]) * 0.1,
                           -(clims[1] - clims[0]) / 5.)
    if abs(values[-1] < 0.01):
        values[-1] = 0.0

    ol = buf.document.newscreenoverlay(
        name='Legend',
        overlayxy=simplekml.OverlayXY(x=0,
                                      y=1,
                                      xunits='fraction',
                                      yunits='fraction'),
        screenxy=simplekml.ScreenXY(x=0, y=1,
                                    xunits='fraction',
                                    yunits='fraction'),
        rotationxy=None,
        rotation=None,)
    ol.size.x = 0
    ol.size.y = 0
    ol.size.xunits = simplekml.Units.fraction
    ol.size.yunits = simplekml.Units.fraction

    inter = plt.isinteractive()
    plt.interactive(False)

    fignum = np.random.randint(100000, 200000)

    fig = plt.figure(fignum, figsize=(3, 4))
    fig.clf()
    ax = plt.axes([0, 0, 1, 1])
    #ax.set_visible(False)
    hndls = []
    lbls = []
    for val in values:
        if val % 1 == 0:
            lbls.append('%d' % val)
        else:
            lbls.append('%0.1f' % val)
        cval = (np.float(val) - clims[0]) / (clims[1] - clims[0])
        hndls.append(ax.plot(np.NaN, np.NaN, 'o',
                             mfc=cmap(cval), mec='none', ms=20,
                             label=lbls[-1])[0])
    lgnd = fig.legend(hndls, lbls, title=title,
                      loc='upper left', numpoints=1)
    lgnd.get_frame().set_facecolor((1., 1., 1., .8))
    lgnd.get_frame().set_edgecolor('none')
    ax.set_visible(False)
    fig.savefig('_legend.png', transparent=True, dpi=130)
    buf.addfile('_legend.png')
    ol.icon.href = '_legend.png'
    plt.close(fig)
    plt.interactive(inter)


class HotSpot(object):

    """
    A single site 'hot spot' class.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
           The site data for the site.
    resdata : :class:`pandas.DataFrame`
              The resource data for the site.
    model : A :mod:`scorers` model, or None (default).
            A scoring model used to score the data.
    units : :class:`dict`
            A dictionary of the units for the site and resource data.
    """

    # These 'properties' define shortcuts to the site data.
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

    def __init__(self, data, resdata, model=None, units={}):
        self.units = units
        self.data = data
        self.resdata = resdata
        self.model = model

    def to_kmz(self, buf,
               mapdata='resource',
               cmap=cmaps['red'],
               clims=[None, None],
               legend_vals=None,
               ):
        """
        Write a google earth '.kmz' file of the data. This method
        writes spatial data to a .kmz file as colored dots. It also
        writes meta-data to the dots, and to the sites.

        Parameters
        ----------
        buf  : string
               The filename, or simplekml.Kml object to write data to.
        mapdata : string
                  The column of `resdata` to map spatially.
        cmap : <matplotlib.colors.ColorMap>
               The colormap for the spatilly mapped data.
        clims : iterable(2)
                Two element object that indicates the [min, max]
                values for the colormap. By default these values
                are chosen to fill the data.
        legend_vals : iterable
                      The list of values to display in the legend. By
                      default (legend_vals=None) the legend will have
                      six values between clims[0] and
                      clims[1]. legend_vals=False will not write a
                      legend.

        """
        if buf.__class__ in [simplekml.Kml, simplekml.Folder]:
            kml = buf
            fl = False
        else:
            kml = simplekml.Kml()
            fl = True
        pt = kml.newpoint(name=self.data.name, coords=[(self.lon, self.lat)])
        pt.lookat = simplekml.LookAt(
            gxaltitudemode=simplekml.GxAltitudeMode.relativetoseafloor,
            latitude=self.lat, longitude=self.lon, range=6.4E4,
            heading=0, tilt=0)
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

        unt = ''
        if mapdata in self.units:
            unt = ' [' + self.units[mapdata] + ']'

        _write_kmz_legend(kml, values=legend_vals,
                          title=mapdata.rstrip('_total').title() + unt,
                          cmap=cmap, clims=clims)

        fol = kml.newfolder(name=mapdata)

        for idx, d in self.resdata.iterrows():
            pt = fol.newpoint(name='', coords=[(d.lon, d.lat)])
            pt.style.iconstyle.icon.href = 'files/icon_circle.png'
            c = mplc.rgb2hex(cmap(float(
                (d[mapdata] - clims[0]) / (clims[1] - clims[0])
            )))
            c = 'FF' + c[5:] + c[3:5] + c[1:3]
            pt.style.iconstyle.color = c
            ed = simplekml.ExtendedData()
            for dval in ['lon', 'lat', 'depth',
                         'resource', 'dist', 'score_total']:
                if dval in d:
                    ed.newdata(dval + '_val',
                               '%0.2f' % d[dval],
                               dval)
            pt.extendeddata = ed

        if fl:
            kml.addfile(package_root + '_img/icon_circle.png')
            kml.savekmz(buf)
            try:
                remove('_legend.png')
            except:
                pass


class HotSpotCollection(object):

    """
    A data object for holding site data (:attr:`data`) and resource
    data (:attr:`resdata`) for a collection of sites, for performing
    site-scoring, ranking and other analysis operations.

    1) Site data is a single point for each site with scorable
    'attributes' (e.g. Yakutat, Alaska has a 'load' and a 'cost of
    energy'.)

    2) Resource data is spatially distributed information (spatial
    attributes) associated with a site (e.g. wave energy density and
    water depth).


    Parameters
    ----------
    data : `pandas.DataFrame`
           DataFrame containing the 'site' (market,lat,lon) data.
    resdata : `pandas.DataFrame`
              DataFrame containing the resource (depth, resource, lat,
              lon) data.
    model : score_site.scorers.Model
            The scoring model that has been applied to the data.
    units : dict
            A dict of strings containing the units of the fields
            (columns) in `data` and `resdata`.


    """

    def __init__(self, data, resdata, model=None, units={}):
        self.data = data
        self.resdata = resdata
        self.model = model
        self.units = units

    def __getitem__(self, val):
        try:
            dtmp = self.data.iloc[val].copy()
        except KeyError:
            dtmp = self.data.loc[val].copy()
        except TypeError:
            dtmp = self.data.loc[val].copy()
        except ValueError:
            dtmp = self.data[val].copy()
        if dtmp.ndim == 1:
            return HotSpot(dtmp,
                           self.resdata.loc[self.resdata.name == dtmp.name],
                           model=self.model,
                           units=self.units.copy(),
                           )
        else:
            bidx = np.zeros(len(self.resdata['name']), dtype='bool')
            for nm in dtmp.index:
                # Loop over names and grab the right resource data.
                bidx |= self.resdata['name'] == nm
            return self.__class__(dtmp,
                                  self.resdata[bidx],
                                  self.model,
                                  units=self.units.copy())

    def to_kmz(self, buf,
               mapdata='resource',
               cmap=cmaps['red'],
               clims=[None, None],
               legend_vals=None,
               ):
        """
        Write a google earth '.kmz' file of the data. This method
        writes spatial data to a .kmz file as colored dots. It also
        writes meta-data to the dots, and to the sites.

        Parameters
        ----------
        buf  : string
               The filename, or simplekml.Kml object to write data to.
        mapdata : string
                  The column of `resdata` to map spatially.
        cmap : <matplotlib.colors.ColorMap>
               The colormap for the spatilly mapped data.
        clims : iterable(2)
                Two element object that indicates the [min, max]
                values for the colormap. By default these values
                are chosen to fill the data.
        legend_vals : iterable
                      The list of values to display in the legend. By
                      default (legend_vals=None) the legend will have
                      six values between clims[0] and
                      clims[1]. legend_vals=False will not write a
                      legend.

        """
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
        unt = ''
        if mapdata in self.units:
            unt = ' [' + self.units[mapdata] + ']'
        ## Write the data for each location:
        _write_kmz_legend(kml, cmap=cmap,
                          title=mapdata.rstrip('_total').title() + unt,
                          values=legend_vals, clims=clims)
        for idx in xrange(self.data.shape[0]):
            dnow = self[idx]
            fol = kml.newfolder(name=dnow.name)
            dnow.to_kmz(fol,
                        mapdata=mapdata,
                        cmap=cmap,
                        clims=clims,
                        legend_vals=False)
            if idx == 0:
                vw = fol.allfeatures[0].lookat
        kml.document.lookat = vw
        if fl:
            kml.addfile(package_root + '_img/icon_circle.png')
            kml.savekmz(buf)
            try:
                pass
                remove('_legend.png')
            except:
                pass

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
        units = pd.Series(self.units)
        units = pd.DataFrame(units, index=units.index, columns=['units'])
        units.to_excel(buf, sheet_name='Units',)
        buf.close()

    def _write_table_legend(self, values=range(10, -1, -2),
                            buf='default_legend.html',
                            cmap=cmaps['green'],
                            ):
        """
        Write the table legend for the colormap specified by `cmap`
        to `buf`. `cmap` should be the same colormap used to write
        an html table using :meth:`to_html`.

        Parameters
        ----------

        buf : string

        """
        ncolor = len(values)
        dat = np.array(values)
        tbl = pd.DataFrame({'Score': dat}).to_html(index=False) + '\n'
        tbl = tbl.replace(r'border="1" ',
                          'style="border-collapse: collapse; text-align:center;"')
        tbl = tbl.replace('<td', '<td style="%s"')
        out = np.empty(ncolor, dtype='O')
        for idx in xrange(ncolor):
            out[idx] = ('background-color:rgb(%d, %d, %d)' %
                        tuple(np.array(cmap(values[idx] / maxscore)[:3]) * 255))
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
                cmap=cmaps['green'],
                hline_widths=[1],
                include_legend=True,
                weights_in_head=True,
                ):
        """
        Write the data in this HotSpotto an html table.

        Parameters
        ----------
        buf : string or <file buffer>
              The file to write the table to. If it is a string, and
              it does not end in '.htm' or '.html', the latter will be
              appended to the filename.
        columns : list of strings
                  The columns to print to the table.
        cmap : <matplotlib colormap>
               The colormap with which to color the table.
        hline_widths : iterable
                       A list of the line widths to use.
        include_legend : bool (default: True)
                         Specify whether to include a legend in the
                         output file.
        weights_in_head : bool (default: True)
                          Specify whether to include the weights of
                          each column with the title in the output
                          file.
        """
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
                    form[irow, icol] += ('background-color:rgb(%d, %d, %d); ' %
                                         tuple(np.array(cmap(self.data[col_score][irow] / maxscore)[:3]) * 255))
        tbl = tbl.format(form=form)
        if basestring in buf.__class__.__mro__:
            if not (buf.endswith('.htm') | buf.endswith('.html')):
                buf += '.html'
            with open(buf, 'w') as fl:
                fl.write(tbl)
                if include_legend:
                    self._write_table_legend(buf=fl, cmap=cmap)
        else:
            buf.write(tbl)
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
        return HotSpotCollection(self.data.copy(),
                                 self.resdata.copy(),
                                 self.model,
                                 units=self.units.copy())

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
