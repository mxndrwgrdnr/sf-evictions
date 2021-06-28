import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os

geog_file = 'Assessor_Historical_Secured_Property_Tax_Rolls.csv'
asr_file = 'assessor_2007-2016.csv'
out_file = 'assessor_2007-2016_fips_jun_2021.csv'
data_dir = '../data/'

def get_county_block_geoms(state_fips, county_fips):

    base_url = (
        'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/'
        'Tracts_Blocks/MapServer/12/query?where=STATE%3D{0}+and+COUNTY%3D{1}'
        '&outFields=GEOID%2CSTATE%2CCOUNTY%2CTRACT%2CBLKGRP%2CBLOCK%2CCENTLAT'
        '%2CCENTLON&outSR=%7B"wkid"+%3A+4326%7D&f=pjson')
    url = base_url.format(state_fips, county_fips)
    result = requests.get(url)
    features = result.json()['features']
    if len(features) >= 100000:
        raise RuntimeError("too many blocks in county to query at once!")
    else:
        df = pd.DataFrame()
        for feature in tqdm(features, total=len(features)):
            tmp = pd.DataFrame([feature['attributes']])
            tmp['geometry'] = Polygon(
                feature['geometry']['rings'][0],
                feature['geometry']['rings'][1:])
            df = pd.concat((df, tmp), ignore_index=True)
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
        return gdf


def fix_esri_geoms(geom_col):
    if geom_col.interiors:
        interiors = [Polygon(x) for x in geom_col.interiors]
        new_geom = MultiPolygon([Polygon(geom_col.exterior)] + interiors)
    else:
        new_geom = geom_col.buffer(0)
    return new_geom




if __name__ == '__main__':

    geog = pd.read_csv(os.path.join(data_dir, geog_file))
    geog = geog[~pd.isnull(geog['Closed Roll Year'])]
    geog['asr_yr'] = geog['Closed Roll Year'].astype(int)

    asr = pd.read_csv(os.path.join(data_dir, asr_file))
    cur_rows = len(asr)
    asr['Parcel Number'] = asr['RP1PRCLID'].str.replace(' ', '')
    asr = asr[~pd.isnull(asr['street_name'])]

    # dropped 844 rows!
    if len(asr) < cur_rows:
        print('Dropped {0} rows with null street names.'.format(
            cur_rows - len(asr)))
        cur_rows = len(asr)

    # merge on year and parcel ID
    asr_w_geog = pd.merge(
        asr, geog[['the_geom', 'asr_yr', 'Parcel Number']],
        on=['asr_yr', 'Parcel Number'], how='left')

    # merge nulls on just parcel ID
    null_mask = pd.isnull(asr_w_geog['the_geom'])
    null_matches = asr_w_geog[null_mask]
    null_merges = pd.merge(
        null_matches[[
            col for col in null_matches.columns if col != 'the_geom']],
        geog[['the_geom', 'Parcel Number']], on='Parcel Number')
    null_merges_grouped = null_merges[['Parcel Number', 'the_geom']].groupby(
        ['Parcel Number']).agg({'the_geom': ['nunique', 'first']})
    assert null_merges_grouped['the_geom']['nunique'].max() == 1

    null_idx = asr_w_geog.loc[null_mask, 'Parcel Number'].values
    null_merges_grouped = null_merges_grouped.reindex(null_idx)
    asr_w_geog.loc[null_mask, 'the_geom'] = null_merges_grouped[
        'the_geom']['first'].values
    if len(asr_w_geog) < cur_rows:
        print('Dropped {0} rows updating null geoms on Parcel ID.'.format(
            cur_rows - len(asr_w_geog)))
        cur_rows = len(asr_w_geog)

    # geocode remaining nulls
    null_mask = pd.isnull(asr_w_geog['the_geom'])
    geolocator = Nominatim(user_agent="foo")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    asr_w_geog['loc_name'] = asr_w_geog['house_2'].astype(str) + ' ' + \
        asr_w_geog['street_name'] + ' ' + asr_w_geog['street_type'] + \
        ', San Francisco, CA'
    asr_w_geog.loc[
        (null_mask) & (asr_w_geog['street_name'] == 'BUENA VISTA WEST'),
        'loc_name'] = asr_w_geog['house_2'].astype(str) + ' BUENA VISTA ' + \
        asr_w_geog['street_type'] + ' West, San Francisco, CA'
    asr_w_geog.loc[
        (null_mask) & (asr_w_geog['street_name'] == 'TELEGRAPH HILL'),
        'loc_name'] = asr_w_geog['house_2'].astype(str) + \
        ' TELEGRAPH HILL, San Francisco, CA'

    tqdm.pandas()
    asr_w_geog.loc[
        pd.isnull(asr_w_geog['the_geom']), 'geocode'] = asr_w_geog.loc[
        pd.isnull(asr_w_geog['the_geom']), 'loc_name'].progress_apply(geocode)

    asr_w_geog.loc[
        pd.isnull(asr_w_geog['the_geom']), 'the_geom'] = asr_w_geog.loc[
        pd.isnull(asr_w_geog['the_geom']), 'geocode'].apply(
        lambda x: str(tuple(x.point[:2])) if x is not None else None)

    if len(asr_w_geog) < cur_rows:
        print(
            'Dropped {0} rows updating null geoms with geocoded'
            ' geoms.'.format(cur_rows - len(asr_w_geog)))
        cur_rows = len(asr_w_geog)

    # get block geoms
    all_blocks = []
    for county in ['075', '081']:
        county_blocks_gdf = get_county_block_geoms('06', county)
        all_blocks.append(county_blocks_gdf)

    blocks_gdf = gpd.GeoDataFrame(
        pd.concat(all_blocks, ignore_index=True), crs="EPSG:4326")

    invalid_mask = ~blocks_gdf.geometry.is_valid
    blocks_gdf.loc[invalid_mask, 'geometry'] = blocks_gdf.loc[
        invalid_mask, 'geometry'].apply(fix_esri_geoms)

    # merge to get fips codes
    asr_w_geog['latlon'] = asr_w_geog['the_geom'].apply(
        lambda x: x.replace('(', '').replace(')', '').split(',')[:2])
    asr_w_geog['latitude'] = asr_w_geog['latlon'].apply(lambda x: float(x[0]))
    asr_w_geog['longitude'] = asr_w_geog['latlon'].apply(lambda x: float(x[1]))
    asr_w_geog['geometry'] = gpd.points_from_xy(
        asr_w_geog['longitude'], asr_w_geog['latitude'])
    asr_w_geog = gpd.GeoDataFrame(asr_w_geog, geometry='geometry')
    asr_w_geog.crs = blocks_gdf.crs

    asr_w_fips = gpd.sjoin(
        asr_w_geog, blocks_gdf[['GEOID', 'geometry']], op='within', how='left')

    if len(asr_w_geog) < cur_rows:
        print('Dropped {0} rows merging block geoms'.format(
            cur_rows - len(asr_w_geog)))
        cur_rows = len(asr_w_geog)
    asr_w_fips[[col for col in asr.columns] + ['GEOID', 'latitude', 'longitude']].to_csv(
        os.path.join(data_dir, out_file))
