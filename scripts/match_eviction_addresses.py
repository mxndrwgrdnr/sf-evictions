import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from shapely import wkt


def clean_eviction_data(ev):

    if 'index' not in ev.columns:
        ev.reset_index(inplace=True)

    ev = ev[(~pd.isnull(ev['address'])) & (ev['address'] != 'UNKNOWN')
            & (ev['address'].str.contains('^[0-9]'))]

    ev.address = ev.address.str.upper()

    ev.loc[ev['address'].str.contains('APT\.'), 'apt'] = ev.loc[
        ev['address'].str.contains('APT\.'), 'address'].str.split('\s#?APT\.?').str[1]
    ev.loc[ev['address'].str.contains('APT\.'), 'address'] = ev.loc[
        ev['address'].str.contains('APT\.'), 'address'].str.split('\s#?APT\.?').str[0]

    ev.loc[ev['address'].str.contains('#'), 'apt'] = ev.loc[
        ev['address'].str.contains('#'), 'address'].str.split('\s#').str[1]
    ev.loc[ev['address'].str.contains('#'), 'address'] = ev.loc[
        ev['address'].str.contains('#'), 'address'].str.split('\s#').str[0]

    ev['address'] = ev['address'].str.replace(
        '|'.join(['STRETT', 'STRRET', 'STREET31', 'SREET', 'DTREET', 'STREEET']), 'ST')
    ev['address'] = ev['address'].str.replace(
        '|'.join(['STRET', 'STRRE', 'STREE$']), 'ST')
    ev['address'] = ev['address'].str.replace(
        '|'.join(['AVENEU', 'AVENE', 'AVENE', 'AVNEUE', 'AAVE']), 'AVE')
    ev['address'] = ev['address'].str.replace('BOUELVARD', 'BLVD')
    ev['address'] = ev['address'].str.replace('VAN VAN', 'VAN')
    ev['address'] = ev['address'].str.replace('ST ST', 'ST')
    ev['address'] = ev['address'].str.replace('AVE AVE', 'AVE')
    ev['address'] = ev['address'].str.replace('MERCED MERCED', 'MERCED')
    ev['address'] = ev['address'].str.replace('POSTREET', 'POST ST')
    ev['address'] = ev['address'].str.replace('21STREET', '21ST ST')
    ev['address'] = ev['address'].str.replace('JOOSTREET', 'JOOST AVE')
    ev['address'] = ev['address'].str.replace('LOCUSTREET', 'LOCUS ST')
    ev['address'] = ev['address'].str.replace('BUSTREET', 'BUSH ST')
    ev['address'] = ev['address'].str.replace('1STREET', '1ST ST')
    ev['address'] = ev['address'].str.replace('AMHERSTREET', 'AMHERST ST')
    ev['address'] = ev['address'].str.replace('TURKSTREET', 'TURK ST')
    ev['address'] = ev['address'].str.replace('HARRISOIN', 'HARRISON')
    ev['address'] = ev['address'].str.replace('BOARDWAY', 'BROADWAY')
    ev['address'] = ev['address'].str.replace("檀蘝涛虧檀迦", "")
    ev['address'] = ev['address'].str.replace("涛宕", "")
    ev['address'] = ev['address'].str.replace("'", "")

    ev.loc[ev['address'] == "20 FRANKLIN STREET", 'address'] = "1580-1598 MARKET ST"
    ev.loc[ev['address'] == "57 TAYLOR STREET", 'address'] = "101-105 TURK ST"
    ev.loc[ev['address'] == "455 EDDY STREET", 'address'] = "350 TURK ST"
    ev.loc[ev['address'] == "790 VALLEJO STREET", 'address'] = "1500-1506 POWELL ST"
    ev.loc[ev['address'] == "2 EMERY LANE", 'address'] = "734-752 VALLEJO ST"
    ev.loc[ev['address'] == "1091 BUSH STREET", 'address'] = "850 LEAVENWORTH ST"
    ev.loc[ev['address'] == "795 20TH AVENUE", 'address'] = "4400 FULTON ST"
    ev.loc[ev['address'] == "440 DAVIS COURT", 'address'] = "100 WASHINGTON ST"
    ev.loc[ev['address'] == "405 12TH AVENUE", 'address'] = "4801 GEARY BLVD"
    ev.loc[ev['address'] == "4 BECKETT STREET", 'address'] = "670 JACKSON ST"
    ev.loc[ev['address'] == "874 SACRAMENTO STREET", 'address'] = "800 STOCKTON ST"
    ev.loc[ev['address'] == "265 NORTH POINT STREET", 'address'] = "2310-2390 POWELL ST"
    ev.loc[ev['address'] == "20 12TH STREET", 'address'] = "1613 MARKET ST"
    ev.loc[ev['address'] == "609 ASHBURY STREET", 'address'] = "1501-1509 HAIGHT ST"
    ev.loc[ev['address'] == "22 VANDEWATER STREET", 'address'] = "333 BAY ST"
    ev.loc[ev['address'] == "160 BAY STREET", 'address'] = "2210-2290 STOCKTON ST"
    ev.loc[ev['address'] == "505 26TH AVENUE", 'address'] = "6201-6209 GEARY BLVD"
    ev.loc[ev['address'] == "3410 22ND STREET", 'address'] = "994-998 GUERRERO ST"
    ev.loc[ev['address'] == "1312 UTAH STREET", 'address'] = "2601-2611 24TH ST"
    ev.loc[ev['address'] == "1290 HAYES STREET", 'address'] = "600-604 DIVISADERO ST"
    ev.loc[ev['address'] == "130 COSO AVE", 'address'] = "1 LUNDY'S LN"
    ev.loc[ev['address'] == '3444 16TH STREET', 'address'] = "3440 16TH ST"
    ev.loc[ev['address'] == '603 NATOMA STREET', 'address'] = "170 7TH ST"


    ev.loc[ev['address'].str.contains('[0-9]02ND'), 'address'] = ev.loc[
        ev['address'].str.contains('[0-9]02ND'), 'address'].str.replace('02ND', ' 2ND')
    ev.loc[ev['address'].str.contains('\s[0-9A-Z]$'), 'address'] = ev.loc[
        ev['address'].str.contains('\s[0-9A-Z]$'), 'address'].str.split(' ').str[:-1].str.join(' ')

    ev.loc[ev['address'].str.contains('BROADWAY'), 'street_type'] = 'ST'
    # ev.loc[ev['address'].str.contains('CESAR CHAVEZ'), 'street_type'] = 'BLVD'
    ev.loc[ev['address'].str.contains('RUSSIA'), 'street_type'] = 'AVE'

    ev.loc[ev['petition'] == 'M101171', 'address'] = '531 GONZALEZ DRIVE'
    ev.loc[ev['petition'] == 'M111009', 'address'] = '55 CHUMASERO DRIVE'
    ev.loc[ev['petition'] == 'M112072', 'address'] = '125 CAMBON DRIVE'
    ev.loc[ev['petition'] == 'M131872', 'address'] = '1921 ELLIS STREET'
    ev.loc[ev['petition'] == 'M140347', 'address'] = '326 LONDON STREET'
    ev.loc[ev['petition'] == 'E980001', 'address'] = '1551A 20TH AVE'
    ev.loc[ev['petition'] == 'E991754', 'address'] = '1271 FILBERT ST'
    ev.loc[ev['petition'] == 'M2K0279', 'address'] = '2364 FULTON ST'
    ev.loc[ev['petition'] == 'S000521', 'address'] = '431 SOMERSET ST'
    ev.loc[ev['petition'] == 'S000417', 'address'] = '1201 GUERRERO ST'

    # parkmerced
    ev.loc[ev['address'].str.contains(
        'GONZALEZ|FONT|SERRANO|CHUMASERO|ARBALLO|GARCES|CAMBON|VIDAL|GRIJALVA|TAPIA|BUCARELI|RIVAS|CRESPI|CARDENAS|HIGUERA'),
        'address'] = '3711 19TH AVE'

    ev = ev[ev['address'] != 'NO ADDRESS PROVIDED']

    # clean street types
    ev['street_type'] = ev['address'].str.split(' ').str[-1]

    st_typ_dict = {'STREET': 'ST', 'AVENUE': 'AVE', 'DRIVE': 'DR', 'BOULEVARD': 'BLVD', 'COURT': 'CT',
                   'TERRACE': 'TER', 'PLACE': 'PL', 'HIGHWAY': 'HWY', 'LANE': 'LN', 'ROAD': 'RD', 'ALLEY': 'ALY',
                   'CIRCLE': 'CIR', 'SQUARE': 'SQ', 'PLAZA': 'PLZ', 'HILLS': 'HL', 'HILL': 'HL'
                   }
    ev = ev.replace({'street_type': st_typ_dict})

    ev.loc[~ev['street_type'].isin(valid_street_types), 'street_type'] = None

    # clean street numbers
    ev['street_num'] = ev['address'].str.split(' ').str[0]
    ev['house_1'] = ''
    ev['house_2'] = ev['street_num']
    ev.loc[ev['street_num'].str.contains('-'), 'house_1'] = ev.loc[
        ev['street_num'].str.contains('-'), 'street_num'].str.split('-').str[0]
    ev.loc[ev['street_num'].str.contains('-'), 'house_2'] = ev.loc[
        ev['street_num'].str.contains('-'), 'street_num'].str.split('-').str[1]

    ev['house_1'] = ev['house_1'].str.replace('\D', '')
    ev['house_2'] = ev['house_2'].str.replace('\D', '')

    # clean street names
    ev['street_name'] = None

    ev.loc[~pd.isnull(ev['street_type']), 'street_name'] = ev.loc[
        ~pd.isnull(ev['street_type']), 'address'].str.split(' ').str[1:-1].str.join(' ')

    ev.loc[pd.isnull(ev['street_name']), 'street_name'] = ev.loc[
        pd.isnull(ev['street_name']), 'address'].str.split(' ').str[1:].str.join(' ')

    ev.loc[ev['street_name'].str.contains(
        '^0'), 'street_name'] = ev.loc[ev['street_name'].str.contains('^0'), 'street_name'].str[1:]

    ev.loc[ev['street_name'].str.contains('\s[0-9]+$'), 'street_name'] = ev.loc[
        ev['street_name'].str.contains('\s[0-9]+$'), 'street_name'].str.split('\s[0-9]+$').str[0]

    ev.loc[ev['street_name'].str.contains('\s[NEWS]\.?$|\sEAST$|\sWEST$'), 'street_name'] = ev.loc[
        ev['street_name'].str.contains('\s[NEWS]\.?$|\sEAST$|\sWEST$'), 'street_name'].str.split('\s[NEWS]\.?$|\sEAST$|\sWEST$').str[0]

    ev.loc[ev['street_name'].str.contains('\sSTREET$|\sAVENUE$|\sST$|\sAVE$'), 'street_type'] = ev.loc[
        ev['street_name'].str.contains('\sSTREET$|\sAVENUE$|\sST$|\sAVE$'), 'street_name'].str.split(' ').str[-1]
    ev = ev.replace({'street_type': st_typ_dict})
    ev.loc[ev['street_name'].str.contains('\sSTREET$|\sAVENUE$|\sST$|\sAVE$'), 'street_name'] = ev.loc[
        ev['street_name'].str.contains('\sSTREET$|\sAVENUE$|\sST$|\sAVE$'), 'street_name'].str.split(' ').str[0]

    ev.loc[ev['house_2'] == '', 'house_2'] = ev['house_1']
    ev.loc[(ev['street_name'] == 'MASON') & (ev['house_2'] == '55'), 'house_2'] = '45'
    ev.loc[(ev['street_name'] == 'GEARY') & (ev['house_2'] == '909'), 'house_2'] = '905'
    ev.loc[(ev['street_name'] == 'POST') & (ev['house_2'] == '1086'), 'house_2'] = '1092'
    ev.loc[(ev['street_name'] == '6TH') & (ev['house_2'] == '125'), 'house_2'] = '117'
    ev.loc[(ev['street_name'] == 'WALLER') & (ev['house_2'] == '840'), 'house_2'] = '8400'
    ev.loc[(ev['street_name'] == 'BATTERY') & (ev['house_2'] == '550'), 'house_2'] = '556'
    ev.loc[(ev['street_name'] == '23RD') & (ev['house_2'] == '2609'), 'house_2'] = '2607'
    ev.loc[(ev['street_name'] == 'HAYES') & (ev['house_2'] == '1665'), 'house_2'] = '1663'
    ev.loc[(ev['street_name'] == 'HAYES') & (ev['house_2'].isin(['1599', '1555', '1509'])), 'house_2'] = '1500'
    ev.loc[(ev['street_name'] == 'HAYES') & (ev['house_2'] == '1360'), 'house_2'] = '1364'
    ev.loc[(ev['street_name'] == 'GEARY') & (ev['house_2'] == '990'), 'house_2'] = '9900'
    ev.loc[ev['street_name'] == 'EDINBURG', 'street_name'] = 'EDINBURGH'
    ev.loc[ev['street_name'] == 'EDINBURG', 'street_type'] = 'ST'
    ev.loc[ev['address'].str.contains('VULCAN'), 'street_type'] = 'SW'
    ev.loc[ev['address'].str.contains('VULCAN'), 'street_name'] = 'VULCAN'
    ev.loc[ev['address'].str.contains('CLINTON PARK'), 'street_name'] = 'CLINTON PARK'
    ev.loc[ev['address'].str.contains('CLINTON PARK'), 'street_type'] = None
    ev.loc[ev['address'].str.contains('.*SO.*VAN NESS.*'), 'street_name'] = 'SOUTH VAN NESS'

    ev.loc[pd.isnull(ev['year']), 'year'] = ev.loc[pd.isnull(ev['year']), 'date'].str[0:4].astype(int)

    ev.loc[ev['house_1'] == '', 'house_1'] = -999
    ev.loc[pd.isnull(ev['house_1']), 'house_1'] = -999
    ev['house_1'] = ev['house_1'].astype(int)
    ev['house_2'] = ev['house_2'].astype(int)

    return ev


def add_asr_attrs(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['bldg_type']
    df.loc[df['index'] == i, 'num_units'] = match['UNITS']
    df.loc[df['index'] == i, 'year_built'] = match['YRBLT']
    df.loc[df['index'] == i, 'matched_house'] = match['house_2']
    df.loc[df['index'] == i, 'matched_street'] = match['street_name']
    df.loc[df['index'] == i, 'matched_street_type'] = match['street_type']
    df.loc[df['index'] == i, 'matched'] = True
    df.loc[df['index'] == i, 'match_year'] = match['asr_yr']


def exact_match(row, df, row_col, df_col):

    if row['street_type'] is None:
        match = df[
            (df['street_name'] == row['street_name']) &
            (df[df_col] == row[row_col]) &
            (df[df_col] > 0)]  # can't match on -999 house nums
    else:
        match = df[
            (df['street_name'] == row['street_name']) &
            (df[df_col] == row[row_col]) &
            (df[df_col] > 0) &  # can't match on -999 house nums
            ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['asr_yr'] - row['year'])
            match = match.sort_values(['year_diff', 'asr_yr'])
        match = match.iloc[0]

    return match


def fuzzy_match_row_btwn_df(row, df, row_col, ascending=True):

    if ascending:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] > row[row_col]) & (df['house_1'] < row[row_col])) &
                (df['house_1'] > 0)]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] > row[row_col]) & (df['house_1'] < row[row_col])) &
                (df['house_1'] > 0) &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    else:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] < row[row_col]) & (df['house_1'] > row[row_col])) &
                (df['house_1'] > 0)]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] < row[row_col]) & (df['house_1'] > row[row_col])) &
                (df['house_1'] > 0) &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    # make sure match is on same side of the street
    match = match[(int(row[row_col]) % 2 == match['house_2'].astype(int) % 2)]

    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['asr_yr'] - row['year'])
            match = match.sort_values(['year_diff', 'asr_yr'])
        match = match.iloc[0]

    return match


def fuzzy_match_df_btwn_row(df, row, df_col, ascending=True):

    if ascending:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] > row['house_1']) & (df[df_col] < row['house_2'])) &
                (row['house_1'] > 0)]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] > row['house_1']) & (df[df_col] < row['house_2'])) &
                (row['house_1'] > 0) &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]
    else:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] < row['house_1']) & (df[df_col] > row['house_2'])) &
                (row['house_1'] > 0)]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] < row['house_1']) & (df[df_col] > row['house_2'])) &
                (row['house_1'] > 0) &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    # make sure match is on same side of the street
    match = match[(int(row['house_2']) % 2 == match[df_col].astype(int) % 2)]

    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['asr_yr'] - row['year'])
            match = match.sort_values(['year_diff', 'asr_yr'])
        match = match.iloc[0]

    return match


def match_sequence(ev, asr_grouped_by_yr):
    """
    Order of operations:
    1. Exact matching on main eviction address ("house_2") w/ Assessor
       - assessor "house_2" = eviction "house_2"
       - infutor "house_2" = eviction "house_2"
       - assesor "house_1" = eviction "house_2"
    2. Fuzzy matching on main eviction address ("house_2") w/ Assessor
       - eviction "house_2" between assessor "house_1" and "house_2"
    3. Exact matching on secondary eviction address ("house_1") w/ Assessor
       - assessor "house_2" = eviction "house_1"
       - assessor "house_1" = eviction "house_1"
       - infutor "house_2" = eviction "house_1
    4. Fuzzy matching on secondary eviction address ("house_1") w/ Assessor
       - eviction "house_1" between assessor "house_1" and "house_2"
    5. Fuzzy matching of Assessor addresses between eviction addresses
       - assessor "house_2" between eviction "house_1" and "house_2"
       - assessor "house_1" between eviction "house_1" and "house_2"
    """

    ### 1. Exact matching on main eviction address ("house_2")
    for i, row in tqdm(ev[pd.isnull(ev['asr_index'])].iterrows(), total=len(ev[pd.isnull(ev['asr_index'])])):

        # w/ main assessor address
        match_asr = exact_match(row, asr_grouped_by_yr, 'house_2', 'house_2')
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

        # w/ secondary assessor address
        match_asr_2 = exact_match(row, asr_grouped_by_yr, 'house_2', 'house_1')
        if len(match_asr_2) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr_2['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr_2['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr_2['index'], 'ev_count_post_07'] += 1
            continue

    print('After running step 1, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[pd.isnull(ev['asr_index'])]), np.round((len(ev[pd.isnull(ev['asr_index'])]) / total_ev) * 100, 3)))

    #### 2. Fuzzy matching on main eviction address ("house_2") w/ Assesor
    for i, row in tqdm(ev[pd.isnull(ev['asr_index'])].iterrows(), total=len(ev[pd.isnull(ev['asr_index'])])):

        # assessor addresses descending
        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_2', ascending=False)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1

            continue

        # assessor addresses ascending
        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_2', ascending=True)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1

            continue

    print('After running step 2, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[pd.isnull(ev['asr_index'])]), np.round((len(ev[pd.isnull(ev['asr_index'])]) / total_ev) * 100, 3)))

    #### 3. Exact matching on secondary eviction address ("house_1")
    for i, row in tqdm(ev[pd.isnull(ev['asr_index'])].iterrows(), total=len(ev[pd.isnull(ev['asr_index'])])):

        if row['house_1'] < 1:
            continue

        # w/ main assessor address
        match_asr = exact_match(row, asr_grouped_by_yr, 'house_1', 'house_2')
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

        # w/ secondary assessor address
        match_asr_2 = exact_match(row, asr_grouped_by_yr, 'house_1', 'house_1')
        if len(match_asr_2) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr_2['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr_2['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

    print('After running step 3, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[pd.isnull(ev['asr_index'])]), np.round((len(ev[pd.isnull(ev['asr_index'])]) / total_ev) * 100, 3)))

    #### 4. Fuzzy matching on secondary eviction address ("house_1") w/ Assessor 
    for i, row in tqdm(ev[pd.isnull(ev['asr_index'])].iterrows(), total=len(ev[pd.isnull(ev['asr_index'])])):
        
        if row['house_1'] < 1:
            continue

        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_1', ascending=False)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_1', ascending=True)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

    print('After running step 4, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[pd.isnull(ev['asr_index'])]), np.round((len(ev[pd.isnull(ev['asr_index'])]) / total_ev) * 100, 3)))

    #### 5. Fuzzy matching of Assessor addresses between eviction addresses
    for i, row in tqdm(ev[pd.isnull(ev['asr_index'])].iterrows(), total=len(ev[pd.isnull(ev['asr_index'])])):

        if row['house_1'] < 1:
            continue

        # using main assesor address ("house_2")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_2', ascending=True)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

        # using main assesor address ("house_2")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_2', ascending=False)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue
        
        # using secondary assesor address ("house_1")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_1', ascending=True)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

        # using secondary assesor address ("house_1")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_1', ascending=False)
        if len(match_asr) > 0:
            ev.loc[ev['index'] == row['index'], 'asr_index'] = match_asr['index']
            asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count'] += 1
            if row['year'] > 2006:
                asr_grouped_by_yr.loc[asr_grouped_by_yr['index'] == match_asr['index'], 'ev_count_post_07'] += 1
            continue

    print('After running step 5, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[pd.isnull(ev['asr_index'])]), np.round((len(ev[pd.isnull(ev['asr_index'])]) / total_ev) * 100, 3)))

    return ev


def get_asr_addrs(ev, asr, addr_parcels):

    for i, row in tqdm(addr_parcels.iterrows(), total=len(addr_parcels)):

        ev_row = ev[ev['index'] == row['index']]
        match = asr[asr['RP1PRCLID'].str.replace(" ", "").isin(row['Parcel Number'])]
        match = match.drop_duplicates(['house_1', 'house_2', 'street_name', 'street_type'])

        if len(match) > 1:

            # if all matches have the same parcel id, then the duplicates are simply
            # the same parcel from different assessor roll years and we retain the 
            # address associated with the roll year closest to the eviction year
            if len(match['RP1PRCLID'].unique()) == 1:
                match = asr[asr['RP1PRCLID'] == match['RP1PRCLID'].values[0]]
                match['year_diff'] = np.abs(match['asr_yr'] - ev_row['year'])
                match = match.sort_values(['year_diff', 'asr_yr'])
                match = match.head(1)

            # exact match on street name
            elif len(match[match['street_name'] == ev_row['street_name'].values[0]]) > 0:
                match = match[match['street_name'] == ev_row['street_name'].values[0]]

                if len(match) > 1:

                    # exact match on house_2
                    if len(match[match['house_2'] == ev_row['house_2'].values[0]]) > 0:
                        match = match[match['house_2'] == ev_row['house_2'].values[0]]

                        if len(match) > 1:

                            # if all rows have the same year built and rent control
                            # eligibility, then it doesn't matter which address we
                            # assign it to.
                            if (len((match['YRBLT'] < 1980).unique()) == 1) & (len(match['rc_eligible'].unique()) == 1):
                                match = match.sample(1)

                    elif len(match[match['house_2'].astype(str) == str(ev_row['house_2'].values[0]) + '0']) > 0:
                        match = match[match['house_2'].astype(str) == str(ev_row['house_2'].values[0]) + '0']

                    elif len(match) == 2:
                        match = match.sort_values('house_2')
                        if str(match.iloc[0]['house_2']) + '0' == str(match.iloc[1]['house_2']):
                            match = match.head(1)

            elif (len((match['YRBLT'] < 1980).unique()) == 1) & (len(match['rc_eligible'].unique()) == 1):
                if (len(match['street_name'].unique()) == 1) & (len(match['house_2'].unique()) == 1):
                    match = match.sample(1)
                elif len(match) == 2:
                    match = match.sort_values('house_2')
                    if str(match.iloc[0]['house_2']) + '0' == str(match.iloc[1]['house_2']):
                        match = match.head(1)

        if len(match) < 1:
            continue
        elif len(match) > 1:
            print('There must be an exception you havent considered yet in matching assessor addresses')
            break
        else:
            # update the addresses so that they'll match on the next pass through
            ev.loc[ev['index'] == row['index'], 'house_2'] = match['house_2'].astype(int).values[0]
            ev.loc[ev['index'] == row['index'], 'house_1'] = match['house_1'].astype(int).values[0]
            ev.loc[ev['index'] == row['index'], 'street_name'] = match['street_name'].values[0]
            ev.loc[ev['index'] == row['index'], 'street_type'] = match['street_type'].values[0]

    return ev

valid_street_types = [
    'ST', 'AVE', 'CT', 'CIR', 'BLVD', 'WAY', 'DR', 'TER', 'HWY', 'HL',
    'PL', 'LN', 'RD', 'PARK', 'ALY', 'PLZ', 'ROW', 'WALK', 'SQ', 'SW']

if __name__ == '__main__':


    ##################################
    # LOAD AND PROCESS ASSESSOR DATA #
    ##################################
    print('Loading SF assessor data.')
    asr = pd.read_csv('../data/assessor_2007-2018_clean_w_none_sttyps.csv', low_memory=False)

    print('Creating SF assessor data universe.')
    asr_grouped_by_yr = asr.groupby(['asr_yr', 'house_1', 'house_2', 'street_name', 'street_type']).agg(
        total_units=('UNITS', 'sum'), diff_unit_counts=('UNITS', 'nunique'), min_units=('UNITS', 'min'),
        diff_bldg_types=('bldg_type', 'nunique'), bldg_type_min=('bldg_type', 'min'), bldg_type_max=('bldg_type', 'max'),
        diff_rc_eligibility=('rc_eligible', 'nunique'), any_rc_eligibility=('rc_eligible', 'max'),
        diff_years_built=('YRBLT', 'nunique'), year_built_min=('YRBLT', 'min'), year_built_max=('YRBLT', 'max')
    ).reset_index()

    if 'index' not in asr_grouped_by_yr.columns:
        asr_grouped_by_yr.reset_index(inplace=True)

    ##################################
    # LOAD AND PROCESS EVICTION DATA #
    ##################################
    print('Loading eviction data.')
    ev = pd.read_csv('../data/all_sf_evictions_2017.csv', low_memory=False)
    # ev = pd.read_csv('../data/ev_matched.csv', low_memory=False)

    print('Cleaning eviction data.')
    ev = clean_eviction_data(ev)

    if 'asr_index' not in ev.columns:
        ev['asr_index'] = None

    if 'ev_count' not in asr_grouped_by_yr.columns:
        asr_grouped_by_yr['ev_count_post_07'] = 0
        asr_grouped_by_yr['ev_count'] = 0
    total_ev = len(ev)

    ev = match_sequence(ev, asr_grouped_by_yr)

    addrs = pd.read_csv('../data/Addresses_with_Units_-_Enterprise_Addressing_System.csv')
    merged = pd.merge(
        ev[pd.isnull(ev['asr_index'])],
        addrs[['Address Number', 'Street Name', 'Street Type', 'point', 'Parcel Number']],
        left_on=['house_2', 'street_name', 'street_type'],
        right_on=['Address Number', 'Street Name', 'Street Type'])
    addr_parcels = merged.groupby('index')['Parcel Number'].apply(
        lambda x: list([z for z in x if type(z) == str])).reset_index()

    ev = get_asr_addrs(ev, asr, addr_parcels)
    ev = match_sequence(ev, asr_grouped_by_yr)

    merged = pd.merge(
        ev[pd.isnull(ev['asr_index'])],
        addrs[['Address Number', 'Street Name', 'Street Type', 'point', 'Parcel Number']],
        left_on=['house_2', 'street_name', 'street_type'],
        right_on=['Address Number', 'Street Name', 'Street Type'])
    merged = merged.drop_duplicates(['index', 'point'])

    merged['point'] = merged['point'].apply(wkt.loads)
    merged = gpd.GeoDataFrame(merged, geometry='point')
    parcels = gpd.read_file(
        '/Users/max/Documents/cal/2020_01_spring/evictions/data/Parcels   Active and Retired/'
        'geo_export_0ee67a28-1bbf-45e3-bcfe-b7d8f6c8355e.shp')
    merged.crs = parcels.crs
    merged = gpd.sjoin(merged, parcels[[
        'blklot', 'block_num', 'from_addre', 'to_address', 'street_nam', 'street_typ', 'geometry']], op='intersects')

    for i in tqdm(merged['index'].unique(), total=len(merged['index'].unique())):

        matches = merged[(~pd.isnull(merged['to_address'])) & (merged['index'] == i)]

        if len(matches) == 0:
            continue

        if len(matches) == 1:
            match = matches
        else:
            matches = asr[
                (asr['street_name'].isin(matches['street_nam'])) & 
                ((asr['street_type'].isin(matches['street_typ'])) | (pd.isnull(asr['street_type']))) &
                ((asr['house_1'].astype(str).isin(
                                    list(np.unique(np.concatenate((matches['from_addre'], matches['to_address'])))))) | 
                                (asr['house_2'].astype(str).isin(
                                    list(np.unique(np.concatenate((matches['from_addre'], matches['to_address'])))))))]
            matches = matches.drop_duplicates(['house_1', 'house_2', 'street_name', 'street_type'])
            if len(matches == 1):
                match = matches
            elif len(matches) == 0:
                print("No matches...that's weird")
                break
            else:
                print('Too many matches')
                break

        ev.loc[ev['index'] == i, 'house_1'] = match['house_1'].values[0]
        ev.loc[ev['index'] == i, 'house_2'] = match['house_2'].values[0]
        ev.loc[ev['index'] == i, 'street_name'] = match['street_name'].values[0]
        ev.loc[ev['index'] == i, 'street_type'] = match['street_type'].values[0]

    ev = match_sequence(ev, asr_grouped_by_yr)



    # ###########
    # # SAVE IT #
    # ###########
    ev.to_csv('../data/ev_matched_w_none_sttyps.csv', index=False)
    asr_grouped_by_yr.to_csv('../data/asr_grouped_by_yr_w_none_sttyps.csv', index=False)
