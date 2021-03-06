"""
A module for computing shipping costs based on regional base- and
rate-costs.

The primary function for use outside this modules is
:func:`calc_shipping`.

"""

import numpy as np
import pandas as pd

deg2rad = np.pi / 180.


def calc_dist(lon0, lat0, lons, lats):
    """
    Calculate the great-circle distance between (lon0, lat0) and an
    array of (lons, lats), in US miles.
    """
    lon0, lat0, lons, lats = map(lambda x: x * deg2rad,
                                 (lon0, lat0, lons, lats))
    phi0 = np.pi / 2 - lat0
    phi = np.pi / 2 - lats

    arg = (np.sin(phi0) * np.sin(phi) * np.cos(lon0 - lons) +
           np.cos(phi0) * np.cos(phi))
    return np.arccos(arg) * 3959


class _ShipCost(object):
    """
    A shipping cost calculator class.

    Parameters
    ----------
    lon0 : float
           The base point longitude.
    lat0 : float
           The base point latitude.
    rate : float
           The cost ($ per tonne per mile) of shipping from that point
           to a point in its region.
    start_cost : float
                 The cost to get to the base point from a mainland
                 U.S. port ($ per tonne).
    """
    # Now we are getting into advanced 'object-oriented' programming.
    # The 'class' declaration defines a new 'object type'.

    # Within the class we can define 'methods'.
    # The __init__ method is a special one that gets called when we
    # create a new 'instance' of the 'shipCost' class.
    def __init__(self, base_point, rate, start_cost=0,
                 region_name=None, port_name=''):
        # All methods must have 'self' as the first input.

        # Here we just store the inputs as 'attributes' of this
        # instance.
        self.base_lon = base_point[0]
        self.base_lat = base_point[1]
        self.rate = rate
        self.start_cost = start_cost

    # We can define as many methods as we like:
    def _calc_dist(self, lon, lat):
        """
        Calculate the distance between the points lon, lat and the
        'base_point' of this instance.
        """
        return calc_dist(self.base_lon, self.base_lat, lon, lat)

    # __call__ is another special method. See below for details...
    def __call__(self, lon, lat):
        """
        Calculate the shipping cost for points 'lon', 'lat'.
        """
        return self._calc_dist(lon, lat) * self.rate + self.start_cost

# Now define the shipping costs for each region:

loc = {'Seattle': (-122.354225, 47.585205),
       'Anchorage': (-149.931388, 61.195310),
       'Unalaska': (-166.542329, 53.895248),
       'Honolulu': (-157.884029, 21.313276),
       'Puerto Rico': (-66.141875, 18.435342),
       'Portland ME': (-70.255, 43.661),
       }

# Now we define the shipping cost calculators for each region:
ship_cost_funcs = {'AKSE': _ShipCost(loc['Seattle'],
                                     0.10, 0,
                                     'AKSE', 'Seattle'),
                   'AK': _ShipCost(loc['Anchorage'],
                                   0.12, 84,
                                   'AK', 'Anchorage'),
                   'Pacific': _ShipCost(loc['Honolulu'],
                                        0.08, 88,
                                        'Pacific', 'Honolulu'),
                   'vPR': _ShipCost(loc['Puerto Rico'],
                                    0.08, 38,
                                    'vPR', 'Puerto Rico'),
                   'NE': _ShipCost(loc['Portland ME'],
                                   0.08, 0,
                                   'NE', 'Portland ME'),
                   }

ship_cost_funcs['AKBS'] = _ShipCost(loc['Unalaska'], 0.15,
                                   ship_cost_funcs['AK'](*loc['Unalaska']),  # See NOTE1
                                   'AKBS', 'Unalaska')
# NOTE1: The base shipping cost from Unalaska is based on the shipping
# cost to Anchorage, plus the shipping cost to Unalaska.


def calc_shipping(region, lons, lats):

    """
    Calculate the shipping costs for points in `region` at `lons`,
    `lats`.

    Parameters
    ----------
    region : array_like(N) of strings
             each string indicates the region that each lat/lon
             pair is in.
    lons : array_like(N) of floats
           longitudes of the points.
    lons : array_like(N) of floats
           latitudes of the points.

    Notes
    -----

    The 'region' must match one of the keys in the ship_cost_funcs
    dictionary in this module.

    """
    if region.__class__ is pd.Series:
        idx = region.index
        region = region.values
    else:
        idx = None
    out = np.zeros_like(region)
    if lons.__class__ is pd.Series:
        lons = lons.values
    if lats.__class__ is pd.Series:
        lats = lats.values
    for ky, func in ship_cost_funcs.iteritems():
        inds = region == ky
        out[inds] = func(lons[inds], lats[inds]).astype(np.float32)
    return out
