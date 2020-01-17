import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import numpy as np


def clean_assessor_data(asr):
    asr = asr[asr['Property Location'] != '0000 0000                       0000']

    asr['house_1'] = asr['Property Location'].str[0:4].str.lstrip('0')
    asr['house_2'] = asr['Property Location'].str[5:9].str.lstrip('0')

    asr['house_1'] = asr['house_1'].str.replace('\D', '')
    asr['house_2'] = asr['house_2'].str.replace('\D', '')

    asr = asr[asr['house_2'] != '']
    asr = asr[~asr['Property Location'].str.contains('SITUS TO BE ASSIGNED')]

    asr['street_name'] = asr['Property Location'].str[9:].str.strip().str.split(' ').str[:-1].str.join(' ').str.strip().str.lstrip('0')
    asr['street_rest'] = asr['Property Location'].str[9:].str.strip().str.split(' ').str[-1].str.strip()

    asr['street_type'] = None
    asr['unit_num'] = None

    asr.loc[asr['street_rest'].str.len().isin([6, 7]), 'street_type'] = asr.loc[
        asr['street_rest'].str.len().isin([6, 7]), 'street_rest'].str[0:2]
    asr.loc[asr['street_rest'].str.len().isin([6, 7]), 'unit_num'] = asr.loc[
        asr['street_rest'].str.len().isin([6, 7]), 'street_rest'].str[2:]
    asr.loc[asr['street_rest'].str.len().isin([4, 5]), 'unit_num'] = asr.loc[
        asr['street_rest'].str.len().isin([4, 5]), 'street_rest']

    asr.loc[asr['Property Location'].str.contains(
        'NORTH POINT'), 'street_name'] = 'NORTH POINT'
    asr.loc[asr['Property Location'].str.contains(
        'NORTH POINT'), 'street_type'] = 'ST'

    asr.loc[asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_type'] = asr.loc[
        asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'),
        'street_name'].str.extract('(\sAVE$|\sAVENUE$|\sSTREET$|\sST$)', expand=False).str.strip().str[0:2]
    asr.loc[asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_name'] = asr.loc[
        asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_name'].str.split(
            '\sAVE$|\sAVENUE$|\sSTREET$|\sST$').str[0]

    asr.loc[asr['street_name'].str.contains(
        '\sSTT$|\sSTIT$|\sSTITE$'), 'street_type'] = 'street'
    asr.loc[asr['street_name'].str.contains('\sSTT$|\sSTIT$|\sSTITE$'), 'street_name'] = asr.loc[
        asr['street_name'].str.contains('\sSTT$|\sSTIT$|\sSTITE$'), 'street_name'].str.split(
            '\sSTT$|\sSTIT$|\sSTITE$').str[0].str.strip()

    asr.loc[asr['street_name'].str.contains('\sNOR$'), 'street_type'] = 'BLVD'
    asr.loc[asr['street_name'].str.contains('FARRELL'), 'street_name'] = 'OFARRELL'
    asr.loc[asr['street_name'] == 'EDINBURG', 'street_name'] = 'EDINBURGH'
    asr.loc[asr['street_name'] == 'EDINBURG', 'street_type'] = 'ST'
    asr.loc[asr['Property Location'].str.contains('BROADWAY'), 'street_type'] = 'ST'
    # asr.loc[asr['Property Location'].str.contains('CESAR CHAVEZ'), 'street_type'] = 'BLVD'
    asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'] = \
        asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'].str[0:3]
    asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_1'].astype(str).str.len() == 4), 'house_1'] = \
        asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_1'].astype(str).str.len() == 4), 'house_1'].str[0:3]

    st_typ_dict = {'street': 'ST', 'AV': 'AVE', 'BL': 'BLVD', 'WY': 'WAY',
                   'TE': 'TER', 'PK': 'PARK', 'HW': 'HWY', 'LANE': 'LN', 'AL': 'ALY',
                   'CR': 'CIR', 'LA': 'LN', 'PZ': 'PLZ', 'TR': 'TER', 'RW': 'ROW', 'BV': 'BLVD',
                   'WK': 'WALK'}
    asr = asr.replace({'street_type': st_typ_dict})

    bldg_typ_dict = {'SRES': 1, 'GOVT': 2, 'IND': 3, 'COMM': 4,
                     'COMR': 5, 'COMO': 6, 'COMH': 7, 'MISC': 8, 'MRES': 9}

    asr['bldg_type'] = asr.replace({'Use Code': bldg_typ_dict})['Use Code']
    return asr


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

    ev.loc[ev['address'].str.contains('VULCAN'), 'street_type'] = 'SW'
    ev.loc[ev['address'].str.contains('VULCAN'), 'street_name'] = 'VULCAN'

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
    ev.loc[pd.isnull(ev['year']), 'year'] = ev.loc[pd.isnull(ev['year']), 'date'].str[0:4].astype(int)

    return ev


def add_inf_attrs(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['PROP_IND'].values[0]
    df.loc[df['index'] == i, 'num_units'] = match['PROP_APTNBR'].values[0]
    df.loc[df['index'] == i, 'year_built'] = match['PROP_YRBLD'].values[0]
    df.loc[df['index'] == i, 'matched_house'] = match['house_2'].values[0]
    df.loc[df['index'] == i, 'matched_street'] = match['street_name'].values[0]
    df.loc[df['index'] == i, 'matched_street_type'] = \
        match['street_type'].values[0]
    df.loc[df['index'] == i, 'matched'] = True


def add_inf_2_attrs(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['PROP_IND'].values[0]
    df.loc[df['index'] == i, 'num_units'] = match['PROP_APTNBR'].values[0]
    df.loc[df['index'] == i, 'year_built'] = match['PROP_YRBLD'].values[0]
    df.loc[df['index'] == i, 'matched_house'] = match['house_1'].values[0]
    df.loc[df['index'] == i, 'matched_street'] = match['STREET2'].values[0]
    df.loc[df['index'] == i, 'matched_street_type'] = \
        match['STRTYPE2'].values[0]
    df.loc[df['index'] == i, 'matched'] = True


def add_asr_attrs(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['bldg_type']
    df.loc[df['index'] == i, 'num_units'] = match['Number of Units']
    df.loc[df['index'] == i, 'year_built'] = match['YRBLT']
    df.loc[df['index'] == i, 'matched_house'] = match['house_2']
    df.loc[df['index'] == i, 'matched_street'] = match['street_name']
    df.loc[df['index'] == i, 'matched_street_type'] = match['street_type']
    df.loc[df['index'] == i, 'matched'] = True
    df.loc[df['index'] == i, 'match_year'] = match['Closed Roll Year']


def add_asr_attrs_og(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['RP1CLACDE'].values[0]
    df.loc[df['index'] == i, 'num_units'] = match['UNITS'].values[0]
    df.loc[df['index'] == i, 'year_built'] = match['YRBLT'].values[0]
    df.loc[df['index'] == i, 'matched_house'] = match['house_2'].values[0]
    df.loc[df['index'] == i, 'matched_street'] = \
        match['street_name'].values[0]
    df.loc[df['index'] == i, 'matched_street_type'] = \
        match['street_type'].values[0]
    df.loc[df['index'] == i, 'matched'] = True


def add_asr_2_attrs(df, i, match):
    df.loc[df['index'] == i, 'bldg_type'] = match['bldg_type']
    df.loc[df['index'] == i, 'num_units'] = match['Number of Units']
    df.loc[df['index'] == i, 'year_built'] = match['YRBLT']
    df.loc[df['index'] == i, 'matched_house'] = match['house_1']
    df.loc[df['index'] == i, 'matched_street'] = match['street_name']
    df.loc[df['index'] == i, 'matched_street_type'] = match['street_type']
    df.loc[df['index'] == i, 'matched'] = True
    df.loc[df['index'] == i, 'match_year'] = match['Closed Roll Year']


def exact_match(row, df, row_col, df_col):

    if row['street_type'] is None:
        match = df[
            (df['street_name'] == row['street_name']) &
            (df[df_col] == row[row_col])]
    else:
        match = df[
            (df['street_name'] == row['street_name']) &
            (df[df_col] == row[row_col]) &
            ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]
    
    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['Closed Roll Year'] - row['year'])
            match = match.sort_values(['year_diff', 'Closed Roll Year'])
        match = match.iloc[0]

    return match


def fuzzy_match_row_btwn_df(row, df, row_col, ascending=True):

    if ascending:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] > row[row_col]) & (df['house_1'] < row[row_col])) &
                (df['house_1'] != '')]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] > row[row_col]) & (df['house_1'] < row[row_col])) &
                (df['house_1'] != '') &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    else:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] < row[row_col]) & (df['house_1'] > row[row_col])) &
                (df['house_1'] != '')]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df['house_2'] < row[row_col]) & (df['house_1'] > row[row_col])) &
                (df['house_1'] != '') &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    # make sure match is on same side of the street
    match = match[(int(row[row_col]) % 2 == match['house_2'].astype(int) % 2)]

    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['Closed Roll Year'] - row['year'])
            match = match.sort_values(['year_diff', 'Closed Roll Year'])
        match = match.iloc[0]

    return match


def fuzzy_match_df_btwn_row(df, row, df_col, ascending=True):

    if ascending:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] > row['house_1']) & (df[df_col] < row['house_2'])) &
                (row['house_1'] != '')]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] > row['house_1']) & (df[df_col] < row['house_2'])) &
                (row['house_1'] != '') &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]
    else:
        if row['street_type'] is None:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] < row['house_1']) & (df[df_col] > row['house_2'])) &
                (row['house_1'] != '')]
        else:
            match = df[
                (df['street_name'] == row['street_name']) &
                ((df[df_col] < row['house_1']) & (df[df_col] > row['house_2'])) &
                (row['house_1'] != '') &
                ((df['street_type'] == row['street_type']) | (pd.isnull(df['street_type'])))]

    # make sure match is on same side of the street
    match = match[(int(row['house_2']) % 2 == match[df_col].astype(int) % 2)]

    if len(match) > 0:
        if len(match) > 1:
            match['year_diff'] = np.abs(match['Closed Roll Year'] - row['year'])
            match = match.sort_values(['year_diff', 'Closed Roll Year'])
        match = match.iloc[0]

    return match

valid_street_types = [
    'ST', 'AVE', 'CT', 'CIR', 'BLVD', 'WAY', 'DR', 'TER', 'HWY', 'HL',
    'PL', 'LN', 'RD', 'PARK','ALY', 'PLZ', 'ROW', 'WALK', 'SQ']

if __name__ == '__main__':

    #################################
    # LOAD AND PROCESS INFUTOR DATA #
    #################################

    # print('Loading Infutor data.')
    # inf = pd.read_csv('bq-results-20200103-133422-6ftj59lyt5kp.csv',
    #                   usecols=['RECTYPE', 'HOUSE', 'PREDIR', 'STREET', 'STRTYPE', 'POSTDIR', 'APTTYPE', 'APTNBR',
    #                            'HOUSE2', 'PREDIR2', 'STREET2', 'STRTYPE2', 'POSTDIR2',
    #                            'ZIP', 'PROP_IND', 'PROP_APTNBR', 'PROP_YRBLD'],
    #                   low_memory=False)

    # print('Cleaning Infutor data.')
    # inf['HOUSE'] = inf['HOUSE'].str.replace('\D', '')
    # inf['HOUSE2'] = inf['HOUSE'].str.replace('\D', '')

    # inf.loc[inf.index.values == 187863, 'HOUSE'] = '1'
    # inf.loc[inf.index.values == 101898, 'HOUSE'] = '2104'
    # inf.loc[inf.index.values == 101898, 'STREET'] = 'BAYSHORE'
    # inf.loc[inf.index.values == 158905, 'HOUSE'] = '163'
    # inf.loc[inf.index.values == 158905, 'STREET'] = 'FAIR OAKS'

    # inf.loc[(~pd.isnull(inf['STREET'])) & (inf['HOUSE'].str.contains('[A-Za-z][A-Za-z]')), 'STRTYPE'] = \
    #     inf.loc[(~pd.isnull(inf['STREET'])) & (
    #         inf['HOUSE'].str.contains('[A-Za-z][A-Za-z]')), 'STREET']

    # inf.loc[(~pd.isnull(inf['STREET'])) & (inf['HOUSE'].str.contains('[A-Za-z][A-Za-z]')), 'STREET'] = \
    #     inf.loc[(~pd.isnull(inf['STREET'])) & (inf['HOUSE'].str.contains(
    #         '[A-Za-z][A-Za-z]')), 'HOUSE'].str.split('-').str[1]

    # inf.loc[(~pd.isnull(inf['STREET'])) & (inf['HOUSE'].str.contains('[A-Za-z][A-Za-z]')), 'HOUSE'] = \
    #     inf.loc[(~pd.isnull(inf['STREET'])) & (inf['HOUSE'].str.contains(
    #         '[A-Za-z][A-Za-z]')), 'HOUSE'].str.split('-').str[0]

    # inf = inf[(inf['PROP_IND'].isin(['10', '11', '21', '22']))]

    # st_typ_dict = {'STREET': 'ST', 'AVENUE': 'AVE', 'HLS': 'HL'}
    # inf = inf.replace({'STRTYPE': st_typ_dict})
    # inf = inf.replace({'STRTYPE2': st_typ_dict})

    # inf = inf[inf['HOUSE'] != '']

    # inf['house_2'] = inf['HOUSE']
    # inf['house_1'] = inf['HOUSE2']
    # inf['street_name'] = inf['STREET']
    # inf['street_type'] = inf['STRTYPE']

    # inf.loc[(~pd.isnull(inf['street_name'])) & (
    #     inf['street_name'].str.contains('BROADWAY')), 'street_type'] = 'ST'
    # inf.loc[(~pd.isnull(inf['STREET2'])) & (
    #     inf['STREET2'].str.contains('BROADWAY')), 'STRTYPE2'] = 'ST'
    # inf.loc[(~pd.isnull(inf['street_name'])) & (
    #     inf['street_name'].str.contains('FARRELL')), 'street_name'] = 'OFARRELL'

    # inf_grouped = inf.groupby(['house_2', 'street_name', 'street_type']).aggregate(
    #     {'PROP_IND': 'max', 'PROP_YRBLD': 'min', 'PROP_APTNBR': 'sum'}).reset_index()

    # inf_grouped_12 = inf.groupby(['house_2', 'house_1', 'street_name', 'STREET2', 'street_type', 'STRTYPE2']).aggregate(
    #     {'PROP_IND': 'max', 'PROP_YRBLD': 'min', 'PROP_APTNBR': 'sum'}).reset_index()

    ##################################
    # LOAD AND PROCESS ASSESSOR DATA #
    ##################################
    print('Loading SF assessor data.')
    asr = pd.read_csv('Assessor_Historical_Secured_Property_Tax_Rolls.csv', low_memory=False)

    print('Cleaning SF assessor data.')
    asr = clean_assessor_data(asr)

    asr_latest = pd.read_excel(
        '2019.8.12__SF_ASR_Secured_Roll_Data_2017-2018_0.xlsx')

    # Using year 2017-2018 year-built dates, THIS IS WRONG
    asr = pd.merge(asr, asr_latest[['PROPLOC', 'YRBLT']],
                   left_on='Property Location', right_on='PROPLOC', how='left')

    asr_grouped_by_yr = asr.groupby(['Closed Roll Year', 'house_1', 'house_2', 'street_name', 'street_type']).aggregate(
        {'YRBLT': 'min', 'bldg_type': 'max', 'Number of Units': 'sum'}).reset_index()
    asr_grouped = asr[asr['Closed Roll Year'] == 2017].groupby(['house_1', 'house_2', 'street_name', 'street_type']).aggregate(
        {'YRBLT': 'min', 'bldg_type': 'max', 'Number of Units': 'sum'}).reset_index()

    ##################################
    # LOAD AND PROCESS EVICTION DATA #
    ##################################
    print('Loading eviction data.')
    # ev = pd.read_csv('all_sf_evictions_2017.csv', low_memory=False)
    ev = pd.read_csv('ev_matched.csv', low_memory=False)

    print('Cleaning eviction data.')
    ev = clean_eviction_data(ev)

    ####################################################################################
    #                                   MATCHING                                       #
    ####################################################################################
    # Order of operations:
    # 1. Exact matching on main eviction address ("house_2") w/ Infutor and Assessor
    #    - assessor "house_2" = eviction "house_2"
    #    - infutor "house_2" = eviction "house_2"
    #    - assesor "house_1" = eviction "house_2"
    # 2. Fuzzy matching on main eviction address ("house_2") w/ Assessor
    #    - eviction "house_2" between assessor "house_1" and "house_2"
    # 3. Exact matching on secondary eviction address ("house_1") w/ Infutor and Assessor
    #    - assessor "house_2" = eviction "house_1"
    #    - assessor "house_1" = eviction "house_1"
    #    - infutor "house_2" = eviction "house_1
    # 4. Fuzzy matching on secondary eviction address ("house_1") w/ Assessor
    #    - eviction "house_1" between assessor "house_1" and "house_2"
    # 5. Fuzzy matching of Assessor addresses between eviction addresses
    #    - assessor "house_2" between eviction "house_1" and "house_2"
    #    - assessor "house_1" between eviction "house_1" and "house_2"
    # 6. Fuzzy matching of main Infutor address between eviction addresses
    #    - infutor "house_2" between eviction "house_1" and "house_2"
    #    - infutor "house_1" between eviction "house_1" and "house_2"
    # 7. Fuzzy matching on main eviction address ("house_2") w/ Infutor
    #    - eviction "house_2" between infutor "house_1" and "house_2"
    # 8. Fuzzy matching on secondary eviction address ("house_1") w/ Infutor
    #    - eviction "house_1" between infutor "house_1" and "house_2"
    ####################################################################################

    if 'matched' not in ev.columns:
        ev['matched'] = False
        ev['bldg_type'] = None
        ev['num_units'] = None
        ev['year_built'] = None
        ev['matched_house'] = None
        ev['matched_street'] = None
        ev['matched_street_type'] = None
        ev['match_year'] = None
        too_many_matches = OrderedDict()
    total_ev = len(ev)

    #### 1. Exact matching on main eviction address ("house_2")
    for i, row in tqdm(ev[~ev['matched']].iterrows(), total=len(ev[~ev['matched']])):

        # w/ main assessor address
        match_asr = exact_match(row, asr_grouped_by_yr, 'house_2', 'house_2')
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

        # w/ secondary assessor address
        match_asr_2 = exact_match(row, asr_grouped_by_yr, 'house_2', 'house_1')
        if len(match_asr_2) > 0:
            add_asr_attrs(ev, row['index'], match_asr_2)
            continue

    print('After running step 1, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[~ev['matched']]), np.round((len(ev[~ev['matched']]) / total_ev) * 100, 3)))

    #### 2. Fuzzy matching on main eviction address ("house_2") w/ Assesor
    for i, row in tqdm(ev[(~ev['matched'])].iterrows(), total=len(ev[~ev['matched']])):

        # assessor addresses descending
        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_2', ascending=False)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

        # assessor addresses ascending
        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_2', ascending=True)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

    print('After running step 2, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[~ev['matched']]), np.round((len(ev[~ev['matched']]) / total_ev) * 100, 3)))

    #### 3. Exact matching on secondary eviction address ("house_1")
    for i, row in tqdm(ev[(~ev['matched'])].iterrows(), total=len(ev[~ev['matched']])):

        if row['house_1'] == '':
            continue

        # w/ main assessor address
        match_asr = exact_match(row, asr_grouped_by_yr, 'house_1', 'house_2')
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

        # w/ secondary assessor address
        match_asr_2 = exact_match(row, asr_grouped_by_yr, 'house_1', 'house_1')
        if len(match_asr_2) > 0:
            add_asr_attrs(ev, row['index'], match_asr_2)
            continue
        
        # # w/ main Infutor address        
        # match_inf = exact_match(row, inf_grouped, 'house_1', 'house_2')
        # if len(match_inf) > 0:
        #     if len(match_inf) == 1:
        #         add_inf_attrs(ev, row['index'], match_inf)
        #         continue
        #     else:
        #         too_many_matches[row['index']] = 'inf'

    print('After running step 3, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[~ev['matched']]), np.round((len(ev[~ev['matched']]) / total_ev) * 100, 3)))

    #### 4. Fuzzy matching on secondary eviction address ("house_1") w/ Assessor 
    for i, row in tqdm(ev[(~ev['matched'])].iterrows(), total=len(ev[~ev['matched']])):
        if len(row['house_1']) == 0:
            continue

        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_1', ascending=False)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue
        
        match_asr = fuzzy_match_row_btwn_df(row, asr_grouped_by_yr, 'house_1', ascending=True)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

    print('After running step 4, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[~ev['matched']]), np.round((len(ev[~ev['matched']]) / total_ev) * 100, 3)))

    #### 5. Fuzzy matching of Assessor addresses between eviction addresses
    for i, row in tqdm(ev[(~ev['matched'])].iterrows(), total=len(ev[~ev['matched']])):

        if len(row['house_1']) == 0:
            continue

        # using main assesor address ("house_2")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_2', ascending=True)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

        # using main assesor address ("house_2")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_2', ascending=False)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue
        
        # using secondary assesor address ("house_1")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_1', ascending=True)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

        # using secondary assesor address ("house_1")
        match_asr = fuzzy_match_df_btwn_row(asr_grouped_by_yr, row, 'house_1', ascending=False)
        if len(match_asr) > 0:
            add_asr_attrs(ev, row['index'], match_asr)
            continue

    print('After running step 5, {0} unmatched addresses remain ({1}%).'.format(
        len(ev[~ev['matched']]), np.round((len(ev[~ev['matched']]) / total_ev) * 100, 3)))

    ###########
    # SAVE IT #
    ###########
    # ev.to_csv('ev_matched_2.csv', index=False)
    # asr_grouped_by_yr.to_csv('asr_grouped_by_yr.csv', index=False)
