"""
This module defines the function for reading data into
:class:`base.HotSpotCollections`.
"""
from base import HotSpotCollection
import pandas as pd


def load_excel(fname):
    """
    Load data from the Excel file 'fname'. This file must contain
    'SiteData' and 'ResourceData' sheets. It can also contain a
    'Units' sheet.

    The 'SiteData' sheet should have sites as rows (site name must be
    the first column and these values must be unique). The first row
    should have headers that define the column names (e.g. 'lat',
    'lon', etc.).

    The 'ResourceData' sheet should have a 'name' column; the values
    in this column should match one of the 'name' values in the
    'SiteData sheet. This sheet can have multiple rows with the same
    value in the 'name' column.  This allows a single entry in the
    'SiteData' sheet to have multiple 'ResourceData' points.  The
    first row must have the headers that define the column names.

    The optional 'Units' sheet is used to specify the units of the
    columns in the previous two sheets. It should have only two
    columns: A) the first indicates the variable name, which must
    match the column header in the other sheet(s) (e.g. 'load' or
    'dist'), and B) the 'unit' string (e.g. 'kWh' or 'km').

    """

    data = pd.read_excel(fname, 'SiteData', index_col=0)
    resdata = pd.read_excel(fname, 'ResourceData')
    try:
        units = pd.read_excel(fname, 'Units',).iloc[:, 0].to_dict()
    except:
        units = {}
    return HotSpotCollection(data, resdata, units=units)
