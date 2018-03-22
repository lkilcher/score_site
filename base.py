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
from .io_write import write_excel
import xlsxwriter as xlw
from copy import deepcopy
import six


def round_sigfig(val, nsig=2):
    try:
        len(val)
    except:
        return np.float(('{:.%dg}' % (nsig)).format(val))
    out = np.empty_like(val)
    for idx, v in enumerate(val):
        out[idx] = round_sigfig(v, nsig=nsig)
    return out

package_root = path.realpath(__file__).replace("\\", "/").rsplit('/', 1)[0] + '/'

results_change_text = dict(Region={'vPR': 'Caribbean',
                                   'NE': 'East Coast',
                                   'AKSE': 'Alaska',
                                   'AKBS': 'Alaska',
                                   'AK': 'Alaska', },
                           name={'N. Cal.': 'N. California',
                                 'C. Cal.': 'C. California',
                                 'S. Cal.': 'S. California',
                           })


## sub_cmap = lambda cmap, low, high: lambda val: cmap((high - low) * val + low)
class MyCmap(object):

    def __copy__(self, **kwargs):
        return self.__class__(deepcopy(self.cmap), deepcopy(self.norm), deepcopy(self.subr))

    def __init__(self, cmap, clims=[0.0, 1.0], sub_cmap_range=[0.0, 1.0], ):
        self.cmap = cmap
        self.norm = mplc.Normalize(clims[0], clims[1], clip=True)
        self.subr = mplc.Normalize(sub_cmap_range[0], sub_cmap_range[1])

    def hex(self, vals):
        out = np.empty(len(vals), dtype='S7')
        for idx, val in enumerate(vals):
            out[idx] = mplc.rgb2hex(self(val))
        return out

    def __call__(self, vals):
        return self.cmap(self.subr.inverse(self.norm(vals)))


class dict_array(np.ndarray):

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'O'  # Force object array.
        o = np.ndarray.__new__(cls, *args, **kwargs)
        o[:] = {}  # __setitem__ will make copies
        return o

    def update(self, **kwargs):
        for item in self.flat:
            item.update(**kwargs)

    def __iadd__(self, val):
        for item in self.flat:
            item.update(**val)

    def __getitem__(self, ind):
        return np.ndarray.__getitem__(self, ind)

    def __setslice__(self, i, j, val):
        self.__setitem__(slice(i, j), val)

    def __setitem__(self, ind, val):
        if isinstance(ind, six.string_types):
            for idx in range(len(self.flat)):
                self[idx][ind] = val
            return
        subarr = self[ind]
        if isinstance(val, dict):
            if subarr.__class__ is dict:  # one value.
                np.ndarray.__setitem__(self, ind, val.copy())
                return
            for idx in range(len(subarr.flat)):
                subarr.flat[idx] = val.copy()
            return
        if isinstance(val, (list, tuple)) and len(val) == len(subarr.flat):
            for idx in range(len(subarr.flat)):
                subarr.flat[idx] = val[idx]
            return
        if isinstance(val, dict_array) and val.size == subarr.size:
            for idx in range(len(subarr.flat)):
                subarr.flat[idx] = deepcopy(val.flat[idx])
            return
        raise ValueError('dict_arrays can only contain dictionaries.')

    def unique(self,):
        return np.unique(self)

## def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
##     new_cmap = mplc.LinearSegmentedColormap.from_list(
##         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
##         cmap(np.linspace(minval, maxval, n)))
##     return new_cmap

cmaps = {'score': MyCmap(plt.get_cmap('YlGn', ), [0, 10], [0.1, 0.7], ),
         'red': plt.get_cmap('Reds'),
         'red2': MyCmap(plt.get_cmap('Reds'), [0, 10], [0.0, 0.7]),
         'rank_sens': MyCmap(plt.get_cmap('bwr_r'), [-3, 3], [.2, .8])
         }
cmaps['score'].legend_vals = np.arange(10, -1, -2)
cmaps['rank_sens'].legend_vals = np.arange(-3, 4, 1)

maxscore = 10.0


def _format_dframe(dframe, cmaps=None, **kwargs):
    if cmaps is None:
        return
    out = dict_array(dframe.shape)
    out[:] = kwargs  # This performs a copy
    for icol, nm in enumerate(dframe.columns):
        for ky, cm in cmaps.items():
            if nm.startswith(ky):
                scr = dframe.loc[:, nm]
                nm_dat = nm.split('_', 1)[1]
                if nm_dat in dframe.columns:
                    icol_dat = dframe.columns.tolist().index(nm_dat)
                else:
                    icol_dat = None
                for irow, val in enumerate(scr):
                    out[irow, icol]['bg_color'] = mplc.rgb2hex(cm(val))
                    if icol_dat is not None:
                        out[irow, icol_dat] = out[irow, icol]
    return out


def _write_kmz_legend(buf, title, cmap):
    if not hasattr(cmap, 'legend_vals'):
        return
    values = cmap.legend_vals
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
        hndls.append(ax.plot(np.NaN, np.NaN, 'o',
                             mfc=cmap(val), mec='none', ms=20,
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


class HotSpotBase(object):

    _excelsheet_write_order = ['Site', 'Resource', 'Units', ]

    def __init__(self, **kwargs):
        self.data = dict(model=None)
        self.data.update(**kwargs)

    def __getattr__(self, nm):
        try:
            return self.data[nm]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (str(self.__class__).split('.')[-1].rstrip("'>"), str(nm)))

    ## def __setattr__(self, nm, val):
    ##     if nm in self.data:
    ##         self.data[nm] = val
    ##     else:
    ##         object.__setattr__(self, nm, val)


class HotSpot(HotSpotBase):

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
        return self.data['Site'].name

    @property
    def lon(self,):
        return self.data['Site']['lon']

    @property
    def lat(self,):
        return self.data['Site']['lat']

    @property
    def region(self,):
        return self.data['Site']['region']

    @property
    def shipping(self,):
        return self.data['Site']['shipping']

    @property
    def load(self,):
        return self.data['Site']['load']

    @property
    def energy_cost(self,):
        return self.data['Site']['rate_avoid']

    coe = energy_cost

    def __init__(self, **kwargs):
        self.data = kwargs

    def to_kmz(self, buf,
               mapdata='resource',
               cmap=cmaps['red'],
               legend=True,
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
                  The data to color spatially.
        cmap : <matplotlib.colors.ColorMap>
               The colormap for the spatilly mapped data.
        legend : bool {True*, False}
           Print the legend?

        """
        if buf.__class__ in [simplekml.Kml, simplekml.Folder]:
            kml = buf
            fl = False
        else:
            kml = simplekml.Kml()
            fl = True
        pt = kml.newpoint(name=self.data['Site'].name, coords=[(self.lon, self.lat)])
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
        if self.data['Resource'] is not None:
            unt = ''
            if mapdata in self.data['Units']:
                unt = ' [' + self.data['Units'][mapdata] + ']'

            if legend:
                _write_kmz_legend(kml,
                                  title=mapdata.rstrip('_total').title() + unt,
                                  cmap=cmap)

            fol = kml.newfolder(name=mapdata)

            for idx, d in self.data['Resource'].iterrows():
                pt = fol.newpoint(name='', coords=[(d.lon, d.lat)])
                pt.style.iconstyle.icon.href = 'files/icon_circle.png'
                c = mplc.rgb2hex(cmap(d[mapdata]))
                c = 'FF' + c[5:] + c[3:5] + c[1:3]
                pt.style.iconstyle.color = c
                ed = simplekml.ExtendedData()
                for dval in ['lon', 'lat', 'depth',
                             'resource', 'range', 'score_total']:
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


class HotSpotCollection(HotSpotBase):

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

    def __getitem__(self, val):
        out = deepcopy(self.data)
        try:
            out['Site'] = out['Site'].iloc[val]
        except KeyError:
            out['Site'] = out['Site'].loc[val]
        except TypeError:
            out['Site'] = self.data['Site'].loc[val]
        except ValueError:
            out['Site'] = self.data['Site'][val]
        if out['Site'].ndim == 1:
            if 'Resource' in out:
                out['Resource'] = out['Resource'][out['Resource']['name'] == out['Site'].name]
            return HotSpot(**out)
        else:
            if 'Resource' in out:
                bidx = np.zeros(len(self.Resource['name']), dtype='bool')
                for nm in out['Site'].index:
                    # Loop over names and grab the right resource data.
                    bidx |= out['Resource'].name == nm
                out['Resource'] = self.Resource[bidx]
            return self.__class__(**out)

    @property
    def ind_max_resource(self,):
        idx = np.zeros(self.Resource.name.shape, dtype='bool')
        for nm in self.Site.index:
            itmp = self.Resource.name == nm
            if itmp.sum() == 0:
                continue
            idx[np.argmax(self.Resource.resource[itmp])] = True
        return idx

    def to_kmz(self, buf,
               mapdata='resource',
               cmap=cmaps['red'],
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

        """
        if buf.__class__ is simplekml.Kml:
            kml = buf
            fl = False
        else:
            kml = simplekml.Kml()
            fl = True
        if self.Resource is not None:
            unt = ''
            if mapdata in self.Units:
                unt = ' [' + self.Units[mapdata] + ']'
            ## Write the data for each location:
            _write_kmz_legend(kml, cmap=cmap,
                              title=mapdata.rstrip('_total').title() + unt,)
        for idx in range(self.Site.shape[0]):
            dnow = self[idx]
            fol = kml.newfolder(name=dnow.name)
            dnow.to_kmz(fol,
                        mapdata=mapdata,
                        cmap=cmap,
                        legend=False)
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

    def to_results_excel(self, buf, cols,
                         bg_cmaps=None, titles=None, form_dict={},
                         units={}, **kwargs):
        if isinstance(buf, six.string_types):
            if not (buf.endswith('.xlsx') or buf.endswith('.xls')):
                buf += '.xlsx'
            buf = xlw.Workbook(buf)
        icol = [np.nonzero(self.data['Site'].columns == res)[0][0] for res in cols]
        form_site = _format_dframe(self.data['Site'], bg_cmaps, **kwargs)
        form = dict_array((form_site.shape[0] + 1, len(icol) + 2))
        form[1:, 2:] = form_site[:, icol]
        form[0].update(bold=True, bottom=2, right=1, text_wrap=True, valign='top')
        form[0, 0].update(width=2.9)
        form[0, 1].update(width=20)
        form[1:].update(right=1, left=1, bottom=1)
        form[:, 1].update(bold=True, right=2)
        if 'market_constraint' in cols:
            idx = cols.index('market_constraint')
            if cols[idx - 1] == 'market':
                idx += 2  # This is for the offset of columns
                form[1:, idx] = form[1:, idx - 1]
                form[1:, idx].update(left=0)
                form[1:, idx - 1].update(right=0)
                form[1, idx].update(width=1.2)
                form[0, idx - 1:(idx + 1)].update(merge='market')
            if cols[idx + 1] == 'market':
                idx += 2  # This is for the offset of columns
                form[1:, idx] = form[1:, idx + 1]
                form[1:, idx].update(left=0)
                form[1:, idx + 1].update(right=0)
                form[0, idx:(idx + 2)].update(merge='market')
        if titles is None:
            titles = deepcopy(cols)
            for idx in range(len(titles)):
                if titles[idx] == 'score_total':
                    titles[idx] = 'Score'
                if titles[idx] == 'rate_avoid':
                    titles[idx] = 'Energy Cost'
                if titles[idx].startswith('comp_rank'):
                    titles[idx] = 'Rank\nChange'
                if titles[idx].startswith('rank_sens'):
                    titles[idx] = 'Rank\nSens.'
                titles[idx] = titles[idx].title()
                if cols[idx] in self.data['Units'].index:
                    unit = str(self.data['Units'].loc[cols[idx]].values[0])
                    if unit == '$/tonne':
                        unit = '$/ton'
                elif cols[idx] == 'market':
                    unit = 'MW'
                elif cols[idx] == 'range':
                    unit = 'km'
                elif cols[idx] == 'rate_avoid':
                    unit = '$/kWh'
                else:
                    unit = None
                if cols[idx] in units:
                    unit = units[cols[idx]]
                if unit is not None:
                    titles[idx] += u'\n[{}]'.format(unit)
        out = np.vstack((np.array(['', '', ] + titles)[None, :],
                         np.hstack((
                             np.arange(1, len(self.data['Site']) + 1)[:, None],
                             self.data['Site'].index[:, None],
                             np.array(self.data['Site'].iloc[:, icol], )))))
        for nm, f in form_dict.items():
            if nm in cols:
                ind = cols.index(nm) + 2
                form[:, ind].update(**f)
        if 'market' in cols:
            ind = cols.index('market') + 2
            out[1:, ind] = round_sigfig(out[1:, ind], 2)
        for idx, nm in enumerate(cols):
            if nm.startswith('rank_'):
                form[1, idx + 2].update(width=7)
                form[:, idx + 2].update(align='center', num_format='[<0]-0.0;[>=0]_-0.0')
            elif nm.startswith('comp_rank'):
                form[1, idx + 2].update(width=7)
                form[:, idx + 2].update(align='center', num_format='[<0]-0;[>=0]_-0')
        for ky, vals in results_change_text.items():
            if ky in cols:
                ind = cols.index(ky) + 2
                for old, new in vals.items():
                    out[out[:, ind] == old, ind] = new
            ind = 1
            for old, new in results_change_text['name'].items():
                out[out[:, ind] == old, ind] = new
        write_excel(buf, out, format=form, sheet_name='results')

    def to_excel(self, buf, bg_cmaps=None, results_args=None, **kwargs):
        """
        Write the data in this data object to a Microsoft Excel file (.xlsx).

        Parameters
        ----------

        buf : string
              The filename or buffer to write to.
        bg_cmaps : dict of cmaps
            The key should be the leading text of a column (or set of
            columns) that will have their background colored.

        Notes
        -----
        If buf does not end in .xlsx or .xls, '.xlsx' will be appended
        to the file name.

        """
        if isinstance(buf, six.string_types):
            if not (buf.endswith('.xlsx') or buf.endswith('.xls')):
                buf += '.xlsx'
            buf = xlw.Workbook(buf)
        for nm in (self._excelsheet_write_order +
                   list(set(self.data.keys()) - set(self._excelsheet_write_order))):
            dat = self.data[nm]
            if isinstance(dat, pd.DataFrame):
                form = _format_dframe(dat, bg_cmaps, **kwargs)
                write_excel(buf, dat,
                            format=form,
                            sheet_name=nm)
            elif isinstance(dat, dict) and nm == 'Units':
                outd = pd.DataFrame(data={'units': dat.values()}, index=dat.keys())
                write_excel(buf, outd, sheet_name=nm)
            elif hasattr(dat, 'to_excel'):
                dat.to_excel(buf, sheet_name=nm)
        if results_args is not None:
            self.to_results_excel(buf, bg_cmaps=bg_cmaps, **results_args)
        if bg_cmaps is not None:
            lgnds = []
            maxlen = 0
            for nm, cm in bg_cmaps.items():
                if cm is not None and hasattr(cm, 'legend_vals'):
                    lgnds.append(nm)
                    maxlen = max(maxlen, len(cm.legend_vals))
            if maxlen:
                out = np.empty((1 + maxlen, len(lgnds)), dtype='O')
                format = dict_array(out.shape)
                for idx, nm in enumerate(lgnds):
                    cm = bg_cmaps[nm]
                    out[0, idx] = nm
                    out[1:(len(cm.legend_vals) + 1), idx] = cm.legend_vals
                    format[1:(len(cm.legend_vals) + 1), idx] = [
                        {'bg_color': mplc.rgb2hex(val)}
                        for val in cm(cm.legend_vals)]
                write_excel(buf, out,
                            format=format,
                            sheet_name='legend')
        buf.close()

    def _write_table_legend(self, values=range(10, -1, -2),
                            buf='default_legend.html',
                            cmap=cmaps['score'],
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
        for idx in range(ncolor):
            out[idx] = ('background-color:rgb(%d, %d, %d)' %
                        tuple(np.array(cmap(values[idx] / maxscore)[:3]) * 255))
        tbl = tbl % tuple(out)
        if six.string_types in buf.__class__.__mro__:
            if not (buf.endswith('.htm') | buf.endswith('.html')):
                buf += '.html'
            with open(buf, 'w') as fl:
                fl.write(tbl)
        else:
            buf.write(tbl)

    def to_html(self, buf,
                columns=['energy_cost', 'load', 'dist',
                         'resource', 'depth', 'shipping', 'score_total'],
                cmap=cmaps['score'],
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
        dat = self.Site[columns]
        tbl = dat.to_html(float_format=lambda s: ('%0.1f' % s)) + '\n'
        tbl = tbl.replace(r'border="1" ', 'style="border-collapse: collapse;"')
        form = np.empty_like(dat, dtype='O')
        form[:] = ''
        tbl = tbl.replace('<th>',
                          '<th style="border-bottom:solid %0.1fpt;">' % hline_widths[0],
                          len(columns) + 1)  # +1 for index
        if self.model is not None and weights_in_head:
            for var, t in self.model.scorers.items():
                tbl = tbl.replace(var, var + ' (%0.1g)' % (t.weight,), 1)
        for irow in range(dat.shape[0]):
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
                if col_score in self.Site:
                    form[irow, icol] += ('background-color:rgb(%d, %d, %d); ' %
                                         tuple(np.array(cmap(self.Site[col_score][irow] / maxscore)[:3]) * 255))
        tbl = tbl.format(form=form)
        if six.string_types in buf.__class__.__mro__:
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
        self.Site.to_csv(fname + '.csv',)
        if resource_fname is not None:
            if not resource_fname.endswith('.csv'):
                resource_fname += '.csv'
            if self.Resource is not None:
                self.Resource.to_csv(resource_fname, index=False)

    def __repr__(self,):
        return self.Site.__repr__()

    def __copy__(self):
        return HotSpotCollection(**deepcopy(self.data))

    copy = __copy__

    def rank(self,
             sort_by=['score_total', 'load'],
             ascending=False,
             sort_res_by=['name', 'score_total', 'resource'],
             ascending_res=[True, False, False],
             clear_0_nan=True,
             scores_adjacent=True,
             ):
        """
        Sort the site-data by the 'score_total' column (secondary
        sorted by: load).

        The resource-data will be sorted by 'name', then 'score_total'
        then 'resource'.

        Parameters
        ----------

        sort_by : list
                  A list of column names to use as the sort criteria.
        ascending : list
                  Boolean for each sort_by value, whether it should be
                  ascending.

        sort_res_by : list
                   A list of column names to sort the resdata.
        ascending_res : list
                  Boolean for each sort_res_by value, whether it
                  should be ascending.

        clear_0_nan : {True*, False}
                      Specifies whether to clear site entries that
                      have a score of zero or NaN.

        """
        if isinstance(ascending, bool):
            ascending = [ascending]
        if len(ascending) == 1:
            ascending *= len(sort_by)
        bdi = np.isnan(self.Site['score_total']) | (self.Site['score_total'] == 0)
        if 'Resource' in self.data:
            resbdi = np.isnan(self.data['Resource']['score_total'])
        if clear_0_nan:
            self.data['Site_all'] = self.data['Site'].copy()
            self.data['Site_all'].ix[bdi, 'score_total'] = -1
            self.data['Site'] = self.data['Site'][~bdi]
            if 'Resource' in self.data:
                self.data['Resource'] = self.data['Resource'][~resbdi]
        else:
            self.Site.ix[bdi, 'score_total'] = -1
            if 'Resource' in self.data:
                self.data['Resource'].ix[resbdi, 'score_total'] = -1
        # Sort the results:
        self.data['Site'] = self.data['Site'].sort_values(sort_by,
                                                   ascending=ascending)
        if 'Site_all' in self.data:
            self.data['Site_all'] = self.data['Site_all'].sort_values(sort_by,
                                                               ascending=ascending)
            self.data['Site_all'].ix[bdi, 'score_total'] = np.NaN
        if 'Resource' in self.data:
            self.data['Resource'] = self.data['Resource'].sort_values(sort_res_by,
                                                               ascending=ascending_res)
        if not clear_0_nan:
            self.data['Site'].ix[bdi, 'score_total'] = np.NaN
            if 'Resource' in self.data:
                self.data['Resource'].ix[resbdi, 'score_total'] = np.NaN

    def organize_cols(self, col_ord=None, scores_adjacent=True, unlisted_first=True):
        """
        Organize the columns of the data.

        Parameters
        ----------
        col_ord : list
            The order of the columns that you would like.
        scores_adjacent : bool
            Whether the columns that are scores should be moved to be
            adjacent to the column it corresponds to.
        unlisted_first : bool or None
            Whether the unlisted items should be first (True), last
            (False), or not included (None).
        """
        def get_order(cols):
            unlisted = []
            ordered = []
            for col in cols:
                if col.startswith('score') and scores_adjacent and col not in col_ord:
                    continue
                if col not in col_ord:
                    unlisted.append(col)
                    if 'score_' + col in cols:
                        unlisted.append('score_' + col)
            for col in col_ord:
                if col in cols:
                    ordered.append(col)
                    if 'score_' + col in cols:
                        ordered.append('score_' + col)
            if unlisted_first is None:
                order = ordered
            elif unlisted_first:
                order = unlisted + ordered
            else:
                order = ordered + unlisted
            return order
        self.data['Site'] = self.data['Site'].loc[:, get_order(self.Site.columns)]
        self.data['Resource'] = self.data['Resource'].loc[:, get_order(self.Resource.columns)]

    def _zeros_series(self,):
        return pd.Series(np.zeros(len(self.Site.index)), index=self.Site.index)

    def compare_scores(self, other, name='score_total',
                       compare_units='percent', append_to_other=False):
        """
        Compare the scores of these results to those of another model
        output.
        """
        if compare_units not in ['percent', 'diff']:
            raise Exception("The compare_units input option most be either 'percent' or 'difff'.")
        comp = self._zeros_series()
        for loc in self.data['Site'].index:
            if loc not in other.data['Site'].index:
                comp[loc] = np.NaN
            else:
                o = other.data['Site'].loc[loc][name]
                comp[loc] = (self.data['Site'].loc[loc][name] - o)
                if compare_units == 'percent':
                    comp[loc] /= o
        self.data['Site']['comp_' + name + '-' + compare_units[0] + '_' + other.model.tag] = comp
        if append_to_other:
            other.data['Site']['comp_' + name + '-' + compare_units[0] + '_' + self.model.tag] = comp

    def compare_rank(self, other, name='score_total', append_to_other=False):
        comp = self._zeros_series()
        r = self.data['Site'][name].copy()
        r.sort_values(ascending=False)
        r = pd.Series(np.arange(len(r)), index=r.index)
        ro = other.data['Site'][name].copy()
        ro.sort_values(ascending=False)
        ro = pd.Series(np.arange(len(ro)), index=ro.index)
        for loc in r.index:
            if loc not in ro.index:
                comp[loc] = np.NaN
            else:
                comp[loc] = ro[loc] - r[loc]
        self.data['Site']['comp_rank_' + other.model.tag] = comp
        if append_to_other:
            other.data['Site']['comp_rank_' + self.model.tag] = comp

    def mean_rank_sensitivity(self, ):
        lst = []
        for nm in self.data['Site']:
            if nm.startswith('comp_rank_'):
                lst += [nm]
        return np.mean(self.data['Site'].ix[:, lst], 1)
