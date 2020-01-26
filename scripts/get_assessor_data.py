import pandas as pd
from tqdm import tqdm

files = [
    '../data/2019.8.12__SF_ASR_Secured_Roll_Data_2017-2018_0.xlsx',
    '../data/2019.8.12__SF_ASR_Secured_Roll_Data_2016-2017_0.xlsx',
    '../data/2019.8.12__SF_ASR_Secured_Roll_Data_2015-2016_0.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2014-2015.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2013-2014.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2012-2013.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2011-2012.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2010-2011.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2009-2010.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2008-2009.xlsx',
    '../data/2019.8.20__SF_ASR_Secured_Roll_Data_2007-2008.xlsx',
]

years = [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]

asr = pd.DataFrame()

for i, f in tqdm(enumerate(files), total=len(files)):

    tmp = pd.read_excel(f)
    tmp['asr_yr'] = years[i]
    asr = pd.concat((asr, tmp), sort=True)

codes = pd.read_csv('../data/Reference__Assessor-Recorder_Property_Class_Codes.csv')
code_dict = dict(zip(codes['Class Code'], codes['Use Code']))
rc_dict = dict(zip(codes['Class Code'], codes['rc_eligible']))

asr['use_code'] = asr['RP1CLACDE'].map(code_dict)
asr['rc_eligible'] = asr['RP1CLACDE'].map(rc_dict)
asr.loc[asr['RP1PRCLID'].isin(['1530 035', '1530 036']), 'PROPLOC'] = '0411 0409 14TH                AV0000'
asr.loc[asr['RP1PRCLID'].isin(['1432 061', '1432 060']), 'PROPLOC'] = '0355 0353 ARGUELLO            BL0000'
asr.loc[asr['RP1PRCLID'].isin(['1744 031', '1744 032']), 'PROPLOC'] = '0506 0504 HUGO                ST0000'
asr.loc[asr['RP1PRCLID'].isin(['1254 098', '1254 097']), 'PROPLOC'] = '0083 0081 DOWNEY              ST0000'
asr.loc[asr['RP1PRCLID'].isin(['0942 051', '0942 052']), 'PROPLOC'] = '2794 2792 FILBERT              ST0000'
asr.loc[asr['RP1PRCLID'] == '1187 016', 'PROPLOC'] = '0000 0048 ASHBURY             ST0000'
asr.loc[asr['RP1PRCLID'] == '1187 017', 'PROPLOC'] = '0000 0050 ASHBURY             ST0000'
asr.loc[asr['RP1PRCLID'].isin(
    ['3730 188', '3730 189', '3730 190', '3730 191', '3730 192']), 'PROPLOC'] = '0000 0019 RAUSCH             ST0000'
asr.loc[asr['RP1PRCLID'] == '3630 031', 'PROPLOC'] = '0175 0175 CHATTANOOGA         ST0000'
asr.loc[asr['RP1PRCLID'] == '3630 032', 'PROPLOC'] = '0177 0177 CHATTANOOGA         ST0000'
asr.loc[asr['RP1PRCLID'] == '3630 033', 'PROPLOC'] = '0179 0179 CHATTANOOGA         ST0000'
asr.loc[asr['RP1PRCLID'] == '3630 030', 'PROPLOC'] = '0173 0173 CHATTANOOGA         ST0000'
asr.loc[asr['RP1PRCLID'] == '3731 242', 'PROPLOC'] = '0000 0038 MOSS                ST0000'


asr = asr[asr['PROPLOC'] != '0000 0000                       0000']

asr['house_1'] = asr['PROPLOC'].str[0:4].str.lstrip('0')
asr['house_2'] = asr['PROPLOC'].str[5:9].str.lstrip('0')

asr['house_1'] = asr['house_1'].str.replace('\D', '')
asr.loc[asr['house_1'] == '', 'house_1'] = -999
asr['house_2'] = asr['house_2'].str.replace('\D', '')

asr = asr[asr['house_2'] != '']
asr = asr[~asr['PROPLOC'].str.contains('SITUS TO BE ASSIGNED')]

asr['street_name'] = asr['PROPLOC'].str[10:].str.strip().str.split(' ').str[:-1].str.join(' ').str.strip().str.lstrip('0')
asr['street_rest'] = asr['PROPLOC'].str[10:].str.strip().str.split(' ').str[-1].str.strip()

asr['street_type'] = None
asr['unit_num'] = None

asr.loc[asr['street_rest'].str.len().isin([6, 7]), 'street_type'] = asr.loc[
    asr['street_rest'].str.len().isin([6, 7]), 'street_rest'].str[0:2]
asr.loc[asr['street_rest'].str.len().isin([6, 7]), 'unit_num'] = asr.loc[
    asr['street_rest'].str.len().isin([6, 7]), 'street_rest'].str[2:]
asr.loc[asr['street_rest'].str.len().isin([4, 5]), 'unit_num'] = asr.loc[
    asr['street_rest'].str.len().isin([4, 5]), 'street_rest']

asr.loc[asr['PROPLOC'].str.contains(
    'NORTH POINT'), 'street_name'] = 'NORTH POINT'
asr.loc[asr['PROPLOC'].str.contains(
    'NORTH POINT'), 'street_type'] = 'ST'

asr.loc[asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_type'] = asr.loc[
    asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'),
    'street_name'].str.extract('(\sAVE$|\sAVENUE$|\sSTREET$|\sST$)', expand=False).str.strip().str[0:2]
asr.loc[asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_name'] = asr.loc[
    asr['street_name'].str.contains('\sAVE$|\sAVENUE$|\sSTREET$|\sST$'), 'street_name'].str.split(
        '\sAVE$|\sAVENUE$|\sSTREET$|\sST$').str[0]

asr.loc[(~pd.isnull(asr['street_name'])) & (asr['street_name'].str.contains(
    '\sSTT$|\sSTIT$|\sSTITE$|\sSTNIT$')), 'street_type'] = 'street'
asr.loc[(~pd.isnull(asr['street_name'])) & (asr['street_name'].str.contains('\sSTT$|\sSTIT$|\sSTITE$|\sSTNIT$')), 'street_name'] = asr.loc[
    (~pd.isnull(asr['street_name'])) & 
    (asr['street_name'].str.contains('\sSTT$|\sSTIT$|\sSTITE$|\sSTNIT$')), 'street_name'].str.split(
        '\sSTT$|\sSTIT$|\sSTITE$|\sSTNIT$').str[0].str.strip()

asr.loc[asr['street_name'].str.contains('\sNOR$'), 'street_type'] = 'BLVD'
asr.loc[asr['street_name'].str.contains('FARRELL'), 'street_name'] = 'OFARRELL'
asr.loc[asr['street_name'] == 'EDINBURG', 'street_name'] = 'EDINBURGH'
asr.loc[asr['street_name'] == 'EDINBURG', 'street_type'] = 'ST'
asr.loc[asr['PROPLOC'].str.contains('.*SO.*VAN NESS.*'), 'street_name'] = 'SOUTH VAN NESS'
asr.loc[asr['PROPLOC'].str.contains('.*SO.*VAN NESS.*'), 'street_type'] = 'AVE'
asr.loc[asr['PROPLOC'].str.contains('BROADWAY'), 'street_type'] = 'ST'

# for pre in ['A', 'B', 'C']:
#     for street in ['COLLINGWOOD', 'HAYES', 'MASONIC', 'RODGERS']:
#         asr.loc[asr['street_name'] == pre + street, 'street_name'] = street
# for pre in ['A', 'B']:
#     for street in [
#             # 'CHURCH', 'UPPER', '14TH', 'FOLSOM', 'PINE', 'FREDERICK', 'PROSPECT', 'HARPER', 'PARNASSUS',
#             # 'MACONDRAY',
#             'STANYAN']:
#         asr.loc[asr['street_name'] == pre + street, 'street_name'] = street
# for pre in ['A']:
#     for street in ['DOWNEY', 'CLINTON PARK']:
#         asr.loc[asr['street_name'] == pre + street, 'street_name'] = street

# many streets have 4 digit street numbers with trailing zeros not present in the eviction records

# asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'] = \
#     asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'].str[0:3]
# asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_1'].astype(str).str.len() == 4), 'house_1'] = \
#     asr.loc[(asr['street_name'] == 'FREDERICK') & (asr['house_1'].astype(str).str.len() == 4), 'house_1'].str[0:3]
# asr.loc[(asr['street_name'] == 'DOWNEY') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'] = \
#     asr.loc[(asr['street_name'] == 'DOWNEY') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'].str[0:3]
# asr.loc[(asr['street_name'] == 'BELVEDERE') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'] = \
#     asr.loc[(asr['street_name'] == 'BELVEDERE') & (asr['house_2'].astype(str).str.len() == 4), 'house_2'].str[0:3]
# asr.loc[(asr['street_name']=='SHRADER') & (asr['house_2'] > 2000) & (asr['house_2'].astype(str).str[-1] == '0'), 'house_2'] = \
#     asr.loc[(asr['street_name']=='SHRADER') & (asr['house_2'] > 2000) & (asr['house_2'].astype(str).str[-1] == '0'), 'house_2'].str[0:3]
# asr.loc[(asr['street_name']=='SHRADER') & (asr['house_1'] > 2000) & (asr['house_1'].astype(str).str[-1] == '0'), 'house_1'] = \
#     asr.loc[(asr['street_name']=='SHRADER') & (asr['house_1'] > 2000) & (asr['house_1'].astype(str).str[-1] == '0'), 'house_1'].str[0:3]
# asr.loc[(asr['street_name']=='WALLER') & (asr['house_2'] > 1000) & (asr['house_2'].astype(str).str[-1] == '0'), 'house_2'] = \
#     asr.loc[(asr['street_name']=='WALLER') & (asr['house_2'] > 1000) & (asr['house_2'].astype(str).str[-1] == '0'), 'house_2'].str[0:3]
# asr.loc[(asr['street_name']=='WALLER') & (asr['house_1'] > 1000) & (asr['house_1'].astype(str).str[-1] == '0'), 'house_1'] = \
#     asr.loc[(asr['street_name']=='WALLER') & (asr['house_1'] > 1000) & (asr['house_1'].astype(str).str[-1] == '0'), 'house_1'].str[0:3]

asr.loc[asr['street_name'].str.contains('^VALLEY.*F$'), 'street_name'] = 'VALLEY'

# # a bunch of street names have an erroneous letter "V" appended to the beginning
# asr.loc[(~pd.isnull(asr['street_name'])) & (asr['street_name'].str.contains('^V[^AEIOU]')), 'street_name'] = \
#     asr.loc[(~pd.isnull(asr['street_name'])) & (asr['street_name'].str.contains('^V[^AEIOU]')), 'street_name'].str[1:]
# other_weird_vnames = [
#     'VELSIE', 'VUNDERWOOD', 'VEDGEHILL', 'VEGBERT', 'VOAKDALE', 'VANDOVER',
#     'VINGERSON', 'VERVINE', 'VEDDY', 'VEVANS', 'VUNION', 'VALEMANY',
#     'VARMSTRONG', 'VELMIRA', 'VIRVING', 'VOCEAN', 'VESMERALDA', 'VELLSWORTH',
#     'VORIZABA', 'VALABAMA', 'VARGUELLO', 'VATHENS', 'VOAK', 'VELLIS',
#     'VORTEGA', 'VALBERTA', 'VUPPER', 'VINGALLS', 'VELIZABETH', 'VARBOR',
#     'VINDIANA', 'VUNIVERSITY', 'VEUCALYPTUS', 'VAPOLLO', 'VULLOA', 'VALADDIN',
#     'VEATON', 'VEDGEWOOD', 'VERIE', 'VAQUAVISTA', 'VALTA', 'VALTON', 'VOTSEGO',
#     'VORD', 'VAPTOS', 'VEXETER', 'VOCTAVIA', 'VURBANO', 'VAGNON', 'VOGDEN',
#     'VASHTON', 'VAUSTIN', 'VASHBURY', 'VABBEY', 'VALDER', 'VARKANSAS',
#     'VOAK GROVE', 'VARCH', 'VEDGAR', 'VILLINOIS', 'VARLETA']
# asr.loc[asr['street_name'].isin(other_weird_vnames), 'street_name'] = \
#     asr.loc[asr['street_name'].isin(other_weird_vnames), 'street_name'].str[1:]

st_typ_dict = {'street': 'ST', 'AV': 'AVE', 'BL': 'BLVD', 'WY': 'WAY',
               'TE': 'TER', 'PK': 'PARK', 'HW': 'HWY', 'LANE': 'LN', 'AL': 'ALY',
               'CR': 'CIR', 'LA': 'LN', 'PZ': 'PLZ', 'TR': 'TER', 'RW': 'ROW', 'BV': 'BLVD',
               'WK': 'WALK'}
asr = asr.replace({'street_type': st_typ_dict})


bldg_typ_dict = {'SRES': 1, 'GOVT': 2, 'IND': 3, 'COMM': 4,
                 'COMR': 5, 'COMO': 6, 'COMH': 7, 'MISC': 8, 'MRES': 9}

asr['bldg_type'] = asr.replace({'use_code': bldg_typ_dict})['use_code']

asr.to_csv('../data/assessor_2007-2018_clean_w_none_sttyps.csv', index=False)