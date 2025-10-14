import numpy as np
import pandas as pd

hdb_detail = pd.read_csv("../../data/auxiliary-data/sg-hdb-block-details.csv")
mrt_stations = pd.read_csv("../../data/auxiliary-data/sg-mrt-stations.csv")
hawkers = pd.read_csv("../../data/auxiliary-data/sg-gov-hawkers.csv")
shopping_malls = pd.read_csv("../../data/auxiliary-data/sg-shopping-malls.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))  # meters

def nearest_dist(lat, lon, facility_xy):
    facility_xy = facility_xy.to_numpy()
    dists = haversine(lat, lon, facility_xy[:,0], facility_xy[:,1])
    j = np.argmin(dists)
    return dists[j] # mrt.iloc[j]['NAME']


def add_geo_data(df, hdb_detail=hdb_detail, mrt_stations=mrt_stations, hawkers=hawkers, shopping_malls=shopping_malls):
    hdb_renamed = hdb_detail.rename(columns={'ADDRESS':'STREET'})

    # get latitude and longtitude
    df = df.merge(
        hdb_renamed[['TOWN','BLOCK','STREET','LATITUDE','LONGITUDE']],
        on=['TOWN','BLOCK','STREET'],
        how='left'
    )

    # get nearest mrt dist (mrt station)
    mrt_xy = mrt_stations[['LATITUDE','LONGITUDE']]
    out = df.apply(lambda r: pd.Series(nearest_dist(r['LATITUDE'], r['LONGITUDE'], mrt_xy),
                    index=['DIST_TO_NEAREST_MRT_M']), axis=1) # in meters
    df = pd.concat([df, out], axis=1)

    # get nearest hawker dist
    hawker_xy = hawkers[['LATITUDE','LONGITUDE']]
    out = df.apply(lambda r: pd.Series(nearest_dist(r['LATITUDE'], r['LONGITUDE'], hawker_xy),
                    index=['DIST_TO_NEAREST_HAWKER_M']), axis=1) # in meters
    df = pd.concat([df, out], axis=1)

    # get nearest shopping mall dist
    shopping_malls_xy = shopping_malls[['LATITUDE','LONGITUDE']]
    out = df.apply(lambda r: pd.Series(nearest_dist(r['LATITUDE'], r['LONGITUDE'], shopping_malls_xy),
                    index=['DIST_TO_NEAREST_SHOP_M']), axis=1) # in meters
    df = pd.concat([df, out], axis=1)

    return df
