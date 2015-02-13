from base import HotSpotData
import pandas as pd
from shipping import calc_shipping


def load_excel(fname):
    data = pd.read_excel(fname, 'SiteData')
    resdata = pd.read_excel(fname, 'ResourceData')
    try:
        units = pd.read_excel(fname, 'Units',).iloc[:, 0].to_dict()
    except:
        units = {}
    return HotSpotData(data, resdata, units=units)


if __name__ == '__main__':

    dat0 = pd.read_excel(
        'HotspotSites_Resources_edited.xlsx', 'HotspotSites', index_col='name')
    dat = dat0.iloc[:, [0, 1, 2, 3, 5]]
    dat.columns = ['lat', 'lon', 'load', 'energy_cost', 'region']
    dat.index.name = None

    dat['shipping'] = calc_shipping(dat['region'], dat['lon'], dat['lat'])

    #dat_res = pd.read_excel('Wave_power_density_hotspots.xlsx',
    #                        'wpd_hotspot_spatialjoin', ).iloc[:, [11, 3, 5, 6, 7, 8]]
    #dat_res.columns = ['name', 'dist', 'lon', 'lat', 'depth', 'resource']
    dat_res = pd.read_excel('Wave_power_density_hotspots02.xlsx',
                            'wpd_hotspot_spatialjoin', ).iloc[:, [10, 3, 4, 5, 6, 7]]
    dat_res.columns = ['name', 'dist', 'lon', 'lat', 'depth', 'resource']
    dat_res.dist *= 0.001

    out = HotSpotData(dat, dat_res)

    out.to_excel('WaveData_All.xlsx')
