import pandas as pd
import numpy as np
import re


valid_street_types = [
    'ST', 'AVE', 'CT', 'CIR', 'BLVD', 'WAY', 'DR', 'TER', 'HWY', 'HL',
    'PL', 'LN', 'RD', 'PARK', 'ALY', 'PLZ', 'ROW', 'WALK', 'SQ', 'SW']


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


ev = pd.read_csv('../data/all_sf_evictions_2017.csv', low_memory=False)
pim = pd.read_csv('../data/sf_pim_addrs.csv')

ev = clean_eviction_data(ev)
ev['addr_clean'] = ev['house_2'].astype(str) + " " + ev['street_name'] + " " +  ev['street_type']

ev['addr_clean'] = ev['address'].str.replace('^\s*', '').str.replace(
    '/\s*$/', '')
ev['addr_clean'] = ev['addr_clean'].str.replace('/', '')
ev['addr_clean'] = ev['addr_clean'].str.replace('\\', '')
ev['addr_clean'] = ev['addr_clean'].str.upper()
ev['addr_clean'] = ev['addr_clean'].str.replace('      ', ' ')
ev['addr_clean'] = ev['addr_clean'].str.replace('     ', ' ')
ev['addr_clean'] = ev['addr_clean'].str.replace('    ', ' ')
ev['addr_clean'] = ev['addr_clean'].str.replace('   ', ' ')
ev['addr_clean'] = ev['addr_clean'].str.replace('  ', ' ')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 1ST', ' 01ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FIRST STREET', ' 01ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FIRST ST', ' 01ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 2ND', ' 02ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' SECOND', ' 02ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 3RD', ' 03RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRD', ' 03RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 4TH', ' 04TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTH', ' 04TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 5TH', ' 05TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FIFTH', ' 05TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 6TH', ' 06TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' SIXTH', ' 06TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 7TH', ' 07TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' SEVENTH', ' 07TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 8TH', ' 08TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' EIGHTH', ' 08TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' 9TH', ' 09TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' NINETH', ' 09TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TENTH', ' 10TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' ELEVENTH', ' 11TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWELTH', ' 12TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTEENTH', ' 13TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTEENTH', ' 14TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FIFTHTEENTH', ' 15TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' SIXTEENTH', ' 16TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' SEVENTEENTH', ' 17TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' EIGHTEENTH', ' 18TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' NINETEENTH', ' 19TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTIETH', ' 20TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-FIRST', ' 21ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYFIRST', ' 21ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-SECOND', ' 22ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYSECOND', ' 22ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-THIRD', ' 23RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYTHIRD', ' 23RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-FOURTH', ' 24TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYFOURTH', ' 24TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-FIFTH', ' 25TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYFIFTH', ' 25TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-SIXTH', ' 26TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYSIXTH', ' 26TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-SEVENTH', ' 27TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYSEVENTH', ' 27TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-EIGHTH', ' 28TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYEIGHTH', ' 28TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTY-NINETH', ' 29TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' TWENTYNINETH', ' 29TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTIETH', ' 30TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-FIRST', ' 31ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYFIRST', ' 31ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-SECOND', ' 32ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYSECOND', ' 32ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-THIRD', ' 33RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYTHIRD', ' 33RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-FOURTH', ' 34TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYFOURTH', ' 34TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-FIFTH', ' 35TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYFIFTH', ' 35TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-SIXTH', ' 36TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYSIXTH', ' 36TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-SEVENTH', ' 37TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYSEVENTH', ' 37TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-EIGHTTH', ' 38TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYEITGHTH', ' 38TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTY-NINETH', ' 39TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' THIRTYNINETH', ' 39TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTIETH', ' 40TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-FIRST', ' 41ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYFIRST', ' 41ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-SECOND', ' 42ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYSECOND', ' 42ND')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-THIRD', ' 43RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYTHIRD', ' 43RD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-FOURTH', ' 44TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYFOURTH', ' 44TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-FIFTH', ' 45TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYFIFTH', ' 45TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-SIXTH', ' 46TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYSIXTH', ' 46TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-SEVENTH', ' 47TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYSEVENTH', ' 47TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTY-EIGHTH', ' 48TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' FOURTYEIGHTH', ' 48TH')
ev['addr_clean'] = ev['addr_clean'].str.replace(' BAYSHORE', ' BAY SHORE')
ev['addr_clean'] = ev['addr_clean'].str.replace(
    ' S VAN NESS', ' SOUTH VAN NESS')

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: re.search('(\d+)[^-]*$', x).group(0)
    if (isinstance(x, str)) and (re.search('(\d+)[^-]*$', x)) else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:x.find('#') - 1]
    if (isinstance(x, str)) and ('#' in x) else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:-7] + ' AVE'
    if (isinstance(x, str)) and (x[-7::] == ' AVENUE') else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:-3] + ' AVE'
    if (isinstance(x, str)) and (x[-3::] == ' AV') else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:-5] + ' LN'
    if (isinstance(x, str)) and (x[-5::] == ' LANE') else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:-6] + ' PLZ'
    if (isinstance(x, str)) and (x[-6::] == ' PLAZA') else x)

ev['addr_clean'] = ev['addr_clean'].apply(
    lambda x: x[0:-8] + ' TER'
    if (isinstance(x, str)) and (x[-8::] == ' TERRACE') else x)

ev['addr_clean'] = ev['addr_clean'].str.replace(' STREET', ' ST')
ev['addr_clean'] = ev['addr_clean'].str.replace(' PLACE', ' PL')
# //address = address.replace(' AVENUE',' AVE')
ev['addr_clean'] = ev['addr_clean'].str.replace(' ALLEY', ' ALY')
ev['addr_clean'] = ev['addr_clean'].str.replace(' BOULEVARD', ' BLVD')
ev['addr_clean'] = ev['addr_clean'].str.replace(' CIRCLE', ' CIR')
ev['addr_clean'] = ev['addr_clean'].str.replace(' COURT', ' CT')
ev['addr_clean'] = ev['addr_clean'].str.replace(' DRIVE', ' DR')
# //address = address.replace(' HILL',' HL')
ev['addr_clean'] = ev['addr_clean'].str.replace(' HIGHWAY', ' HWY')
# //address = address.replace(' LANE',' LN')
# //address = address.replace(' PLAZA',' PLZ')
ev['addr_clean'] = ev['addr_clean'].str.replace(' ROAD', ' RD')

ev['addr_clean'] = ev['addr_clean'].str.replace("'", "")

ev["street_type_exists"] = False
ev.loc[ev['addr_clean'].str[-3::].isin(
    [" ST", " PL", " CT", " DR", " HL", " LN", " RD"]),
    "street_type_exists"] = True
ev.loc[ev['addr_clean'].str[-4::].isin(
    [" AVE", " ALY", " HWY", " CIR", " PLZ", " TER", " WAY", " ROW"]),
    "street_type_exists"] = True
ev.loc[ev['addr_clean'].str[-5::].isin(
    [" BLVD", " WEST", " EAST", " PARK", " LOOP", " WALK", " STWY"]),
    "street_type_exists"] = True
ev.loc[ev['addr_clean'].str[-6::].isin(
    [" SOUTH", " NORTH"]), "street_type_exists"] = True

ev = ev.drop(columns='blk_lot')

foo = pd.merge(
    ev[~pd.isnull(ev['addr_clean'])], pim[['addr_simple', 'blk_lot']],
    how='left',
    left_on='addr_clean', right_on='addr_simple')

ev['addr_array'] = ev['addr_clean'].str.split(" ")
ev['addr_num'] = ev['addr_array'].apply(
    lambda x: x[0] if (isinstance(x, list)) and (x[0].isnumeric()) else False)
ev['addr_st'] = ev['addr_array'].apply(
    lambda x: x[1] if (isinstance(x, list)) and (x[0].isnumeric()) else False)
