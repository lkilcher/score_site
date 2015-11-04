"""
This module defines the function for reading data into
:class:`base.HotSpotCollections`.
"""
import pandas as pd
import xlsxwriter as xlw
import xlsxwriter.utility as xlu
import numpy as np


def write_excel(buf, array, format=None, sheet_name='Sheet1', index=True, header=True, na_rep=''):
    if isinstance(buf, basestring):
        wb = xlw.Workbook(buf)
        closefile = True
    else:
        wb = buf
        closefile = False
    ws = wb.add_worksheet(sheet_name)
    merge = {}
    forms = {}
    if format is not None:
        for irow in range(format.shape[0]):
            for icol, f in enumerate(format[irow]):
                if f is None:
                    continue
                if 'height' in f:
                    ws.set_row(irow, irow, f.pop('height'))
                if 'width' in f:
                    ws.set_column(icol, icol, f.pop('width'))
                if 'merge' in f:
                    mval = f.pop('merge')
                    if mval not in merge:
                        merge[mval] = [(irow, icol)]
                    else:
                        merge[mval].append((irow, icol))
        formind = np.empty(format.shape, dtype=int)
        tmp = np.unique(format)
        for idx, f in enumerate(tmp):
            if f is not None:
                forms[idx] = wb.add_format(f)
                formind[format == f] = idx

    if array.__class__ is pd.DataFrame:
        forms['header'] = wb.add_format({'border': 1})
        forms['header'].set_bold()
        col_labels = array.columns.tolist()
        ind_labels = array.index.tolist()
        array = array.values
    else:
        header = False
        index = False

    for irow in range(array.shape[0]):
        for icol in range(array.shape[1]):
            if format is not None and format[irow, icol] is not None:
                f = forms[formind[irow, icol]]
            else:
                f = None
            val = array[irow, icol]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = na_rep
            ws.write(irow + header, icol + index, val, f)
    for val in merge.itervalues():
        val = np.array(val)
        rowcol = [val[:, 0].min(),
                  val[:, 1].min(),
                  val[:, 0].max(),
                  val[:, 1].max()]
        ws.merge_range(xlu.xl_range(*rowcol),
                       array[rowcol[0], rowcol[1]],
                       forms[formind[rowcol[0], rowcol[1]]])
    if header:
        for idx, lbl in enumerate(col_labels):
            if lbl in [None, np.NaN]:
                lbl = na_rep
            ws.write(0, idx + index, lbl, forms['header'])
    if index:
        for idx, lbl in enumerate(ind_labels):
            if lbl in [None, np.NaN]:
                lbl = na_rep
            ws.write(idx + int(header), 0, lbl, forms['header'])
    if closefile:
        wb.close()
