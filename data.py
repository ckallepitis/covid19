# Imports:
import pandas as pd
pd.options.mode.chained_assignment = None  # default  = 'warn'
import numpy as np
import io
import requests
import plotly.graph_objs as go
from elements import SPAIN_GEOLOCATIONS
from elements import AFRICA, AMERICAS, ASIA, EUROPE, OCEANIA

#=============================================================================#
#                                                                             #
#                         Data preparation - Worldwide                        #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Load Covid19 data ==========

url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
df.rename(columns = {'countriesAndTerritories': 'country'}, inplace = True)
df.country = df.country.replace('United_States_of_America','USA')
df.country = df.country.replace('United_Kingdom','UK')
df.sort_values(['countryterritoryCode',
                'year','month','day']).reset_index(drop = True)
df['dateRep'] = pd.to_datetime(df['dateRep'], format = '%d/%m/%Y').dt.date
df_bar_data = df
df = df[['dateRep','day','month','year','cases','deaths',
         'country', 'geoId','countryterritoryCode']]

#=============================================================================#
# ========== Line Graph ==========

#=============================================================================#
# ========== Cases ==========

df_pivot_c = df.pivot(index = 'dateRep', columns = 'country', values = 'cases')
df_pivot_c_cumsum = df_pivot_c.cumsum()

df_norm_100case = []
for column in df_pivot_c_cumsum.columns:
    df_norm_100case.append(df_pivot_c_cumsum[column]\
                           [df_pivot_c_cumsum[column] >= 100].values)

df_norm_100case = pd.DataFrame(df_norm_100case).T

df_norm_100case.columns = df_pivot_c_cumsum.columns

df_norm_100case = df_norm_100case.dropna(how = 'all',axis = 1)

y_cutoff = 1500000
x_start = 100
every_1day = [x_start]
every_2days = [x_start]
every_3days = [x_start]
every_4days = [x_start]
every_week = [x_start]
for i in range(1,77):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        every_1day.append(x)
    else: every_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        every_2days.append(y)
    else: every_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        every_3days.append(z)
    else: every_3days.append(z)
    v = x_start * ((2**(1/4)) ** i)
    if v > y_cutoff:
        v = None
        every_4days.append(v)
    else: every_4days.append(v)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        every_week.append(w)
    else: every_week.append(w)

Lines_c = pd.DataFrame(every_1day,columns = ['every 1 day'],dtype = np.float64)
Lines_c['every 2 days'] = pd.DataFrame(every_2days)
Lines_c['every 3 days'] = pd.DataFrame(every_3days)
Lines_c['every 4 days'] = pd.DataFrame(every_4days)
Lines_c['every week'] = pd.DataFrame(every_week)

#=============================================================================#
# ========== Deaths ==========

df_pivot_d = df.pivot(index = 'dateRep', columns = 'country', values = 'deaths')
df_pivot_d_cumsum = df_pivot_d.cumsum()

df_norm_10death = []
for column in df_pivot_d_cumsum.columns:
    df_norm_10death.append(df_pivot_d_cumsum[column]\
                           [df_pivot_d_cumsum[column] >= 10].values)

df_norm_10death = pd.DataFrame(df_norm_10death).T

df_norm_10death.columns = df_pivot_d_cumsum.columns

df_norm_10death = df_norm_10death.dropna(how = 'all', axis = 1)

y_cutoff = 70000
x_start = 10
every_1day = [x_start]
every_2days = [x_start]
every_3days = [x_start]
every_week = [x_start]
for i in range(1,77):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        every_1day.append(x)
    else: every_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        every_2days.append(y)
    else: every_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        every_3days.append(z)
    else: every_3days.append(z)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        every_week.append(w)
    else: every_week.append(w)

Lines_d = pd.DataFrame(every_1day,columns = ['every 1 day'],dtype = np.float64)
Lines_d['every 2 days'] = pd.DataFrame(every_2days)
Lines_d['every 3 days'] = pd.DataFrame(every_3days)
Lines_d['every week'] = pd.DataFrame(every_week)

#=============================================================================#
# ========== Deaths - Daily ==========

countries = df_pivot_d.agg(max) > 3
countries = (countries[countries == True].index.values)
df_pivot_3death = df_pivot_d[countries]

df_norm_3death = []
for column in df_pivot_3death.columns:
    temp = df_pivot_3death[column].reset_index(drop = True)
    df_norm_3death.append(df_pivot_3death[column].iloc[temp[temp > 3]\
                                                       .index[0]:].values)

df_norm_3death = pd.DataFrame(df_norm_3death).T
df_norm_3death.columns = df_pivot_3death.columns
df_norm_3death = df_norm_3death.rolling(7).mean().iloc[6:]\
.reset_index(drop = True)
df_norm_3death = df_norm_3death.dropna(how = 'all', axis = 1)
df_norm_3death = df_norm_3death.drop(['Algeria','Iraq','Philippines',
                                      'San_Marino','Slovenia'], axis = 1)

#=============================================================================#

df_traj = {
    'cases' : df_norm_100case,
    'deaths' : df_norm_10death,
    'daily_deaths' : df_norm_3death,
    'lines_cases' : Lines_c,
    'lines_deaths' : Lines_d,
}

#=============================================================================#
# ========== Bar-plot ==========

df_bar_cases = df_bar_data.drop(['day','month','year','geoId',
                            'countryterritoryCode','popData2018'], axis = 1)
df_bar_cases = df_bar_cases.groupby('country').sum().reset_index()
df_bar_cases['popData2018'] = df_bar_data['popData2018']
df_bar_cases['cases_per_mill'] = ((df_bar_cases['cases'] * 1e6) /\
                                  df_bar_cases['popData2018']).astype(int)
df_bar_cases_per_mill = df_bar_cases\
.sort_values('cases_per_mill', ascending = False).iloc[:11,:]
df_bar_cases = df_bar_cases.sort_values('cases', ascending = False).iloc[:11,:]

df_bar_deaths = df_bar_data.drop(['day','month','year','geoId',
                             'countryterritoryCode','popData2018'], axis = 1)
df_bar_deaths = df_bar_cases.groupby('country').sum().reset_index()
df_bar_deaths['popData2018'] = df_bar_data['popData2018']
df_bar_deaths['deaths_per_mill'] = ((df_bar_deaths['deaths'] * 1e6) /\
                                    df_bar_deaths['popData2018']).astype(int)
df_bar_deaths_per_mill = df_bar_deaths\
.sort_values('deaths_per_mill', ascending = False).iloc[:11,:]
df_bar_deaths = df_bar_deaths.sort_values('deaths',ascending= False).iloc[:11,:]

#=============================================================================#

df_bar = {
    'cases' : df_bar_cases,
    'deaths' : df_bar_deaths,
    'cases_per_mill' : df_bar_cases_per_mill,
    'deaths_per_mill' : df_bar_deaths_per_mill,
}

#=============================================================================#
# ========== Geo-plot ==========

df_geo = df[df.year != 2019].drop(['day','year','month','geoId'], axis = 1)
df_geo = df_geo[df_geo.country != 'Cases_on_an_international_conveyance_Japan']\
.reset_index(drop = True)
df_geo.columns = ['day','cases','deaths','country','iso_alpha']
df_geo['continent'] = 0
df_geo['day'] = pd.to_datetime(df_geo['day'], format = '%Y/%m/%d').dt.dayofyear
df_geo = df_geo.sort_values(['day','country']).reset_index(drop = True)
df_geo['cases'] = df_geo[['country', 'day','cases']]\
.groupby(['country', 'day']).sum()\
.groupby(level = 0).cumsum().reset_index()\
.sort_values(['day','country'])\
.reset_index(drop = True).cases
df_geo['deaths'] = df_geo[['country', 'day','deaths']]\
.groupby(['country', 'day']).sum()\
.groupby(level = 0).cumsum().reset_index()\
.sort_values(['day','country'])\
.reset_index(drop = True).deaths

for i, country in enumerate(df_geo.country):
    if country in AFRICA:
        df_geo['continent'].iloc[i] = 'Africa'
    elif country in AMERICAS:
        df_geo['continent'].iloc[i] = 'Americas'
    elif country in ASIA:
        df_geo['continent'].iloc[i] = 'Asia'
    elif country in EUROPE:
        df_geo['continent'].iloc[i] = 'Europe'
    elif country in OCEANIA:
        df_geo['continent'].iloc[i] = 'Oceania'

#=============================================================================#
#                                                                             #
#                         Data preparation - Spain                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Load Covid19 data ==========
urls = 'https://covid19.isciii.es/resources/serie_historica_acumulados.csv'
sp = requests.get(urls).content
dfs = pd.read_csv(io.StringIO(sp.decode('latin1')))
df_loc = pd.DataFrame(SPAIN_GEOLOCATIONS)

dfs = dfs.iloc[:-6,:]
dfs.columns = ['region','date','cases','PCR+','TestAc+',\
               'hospitalised','ICU','deaths','recovered']
dfs.date = pd.to_datetime(dfs.date, format = '%d/%m/%Y')
dfs['cases'] = dfs['cases'].combine_first(dfs['PCR+'])
dfs = dfs.drop(['PCR+','TestAc+'],axis = 1)
dfs = dfs.sort_values(['region','date'])
dfs = pd.merge(dfs, df_loc[['region','region_name']], how='inner', on='region')

#=============================================================================#
# ========== Line Graph ==========

#=============================================================================#
# ========== Cases ==========

dfs_pivot_c = dfs.pivot(index= 'date', columns= 'region_name', values= 'cases')

dfs_norm_10case = []
for column in dfs_pivot_c.columns:
    dfs_norm_10case.append(dfs_pivot_c[column][dfs_pivot_c[column]>=10].values)

dfs_norm_10case = pd.DataFrame(dfs_norm_10case).T

dfs_norm_10case.columns = dfs_pivot_c.columns

dfs_norm_10case = dfs_norm_10case.dropna(how = 'all',axis = 1)

y_cutoff = dfs_norm_10case.max().max() + dfs_norm_10case.max().max() * 0.2
x_start = 10
every_1day = [x_start]
every_2days = [x_start]
every_3days = [x_start]
every_4days = [x_start]
every_week = [x_start]
for i in range(1,dfs_norm_10case.shape[0] + 10):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        every_1day.append(x)
    else: every_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        every_2days.append(y)
    else: every_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        every_3days.append(z)
    else: every_3days.append(z)
    v = x_start * ((2**(1/4)) ** i)
    if v > y_cutoff:
        v = None
        every_4days.append(v)
    else: every_4days.append(v)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        every_week.append(w)
    else: every_week.append(w)

Lines_cs = pd.DataFrame(every_1day, columns= ['every 1 day'], dtype= np.float64)
Lines_cs['every 2 days'] = pd.DataFrame(every_2days)
Lines_cs['every 3 days'] = pd.DataFrame(every_3days)
Lines_cs['every 4 days'] = pd.DataFrame(every_4days)
Lines_cs['every week'] = pd.DataFrame(every_week)

#=============================================================================#
# ========== Deaths ==========

dfs_pivot_d = dfs.pivot(index= 'date', columns= 'region_name', values= 'deaths')

dfs_norm_3death = []
for column in dfs_pivot_d.columns:
    dfs_norm_3death.append(dfs_pivot_d[column][dfs_pivot_d[column] >= 3].values)

dfs_norm_3death = pd.DataFrame(dfs_norm_3death).T

dfs_norm_3death.columns = dfs_pivot_d.columns

dfs_norm_3death = dfs_norm_3death.dropna(how = 'all',axis = 1)

y_cutoff = 6000
x_start = 3
every_1day = [x_start]
every_2days = [x_start]
every_3days = [x_start]
every_week = [x_start]
for i in range(1,dfs_norm_3death.shape[0]+10):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        every_1day.append(x)
    else: every_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        every_2days.append(y)
    else: every_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        every_3days.append(z)
    else: every_3days.append(z)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        every_week.append(w)
    else: every_week.append(w)

Lines_ds = pd.DataFrame(every_1day, columns= ['every 1 day'], dtype= np.float64)
Lines_ds['every 2 days'] = pd.DataFrame(every_2days)
Lines_ds['every 3 days'] = pd.DataFrame(every_3days)
Lines_ds['every week'] = pd.DataFrame(every_week)

#=============================================================================#
# ========== Deaths - Daily ==========

dfs_pivot_d_daily = dfs_pivot_d.diff()
dfs_pivot_d_daily.loc['2020-03-08'] = dfs_pivot_d.loc['2020-03-08']

regions = dfs_pivot_d_daily.agg(max) > 3
regions = (regions[regions == True].index.values)
dfs_pivot_3death = dfs_pivot_d_daily[regions]

dfs_norm_3death_roll = []
for column in dfs_pivot_3death.columns:
    temp = dfs_pivot_3death[column].reset_index(drop = True)
    dfs_norm_3death_roll.append(dfs_pivot_3death[column].iloc[temp[temp > 3]\
                                                        .index[0]:].values)

dfs_norm_3death_roll = pd.DataFrame(dfs_norm_3death_roll).T
dfs_norm_3death_roll.columns = dfs_pivot_3death.columns
dfs_norm_3death_roll = dfs_norm_3death_roll.rolling(7).mean().iloc[6:]\
                                                      .reset_index(drop = True)
dfs_norm_3death_roll = dfs_norm_3death_roll.dropna(how = 'all',axis = 1)

#=============================================================================#

df_traj_spain = {
    'cases' : dfs_norm_10case,
    'deaths' : dfs_norm_3death,
    'daily_deaths' : dfs_norm_3death_roll,
    'lines_cases' : Lines_cs,
    'lines_deaths' : Lines_ds,
}

#=============================================================================#
# ========== Spain Map ==========

df_geo_spain = pd.merge(dfs,df_loc.drop('region_name',axis = 1),
                        how = 'inner', on = 'region').fillna(0)

#=============================================================================#
# ========== Sunburst chart ==========

datas = df_geo_spain[df_geo_spain.date == df_geo_spain.date.unique()\
                     [df_geo_spain.date.unique().__len__()-1]]

data_sun = datas[['region','cases','hospitalised',
                  'ICU','deaths','recovered']].copy()
data_sun['active'] = pd.Series(data_sun.loc[:,'cases'] -\
                               ( data_sun.loc[:,'deaths']\
                                + data_sun.loc[:,'recovered']))
data_sun['home_isolation'] = pd.Series(data_sun.loc[:,'active']\
                                       - ( data_sun.loc[:,'hospitalised']))
data_sun = data_sun.reset_index(drop = True)
data_sun['home_isolation'][7] = data_sun.loc[:,'home_isolation'][7] * (-1)
data_sun['home_isolation'][14] = data_sun.loc[:,'home_isolation'][14] * (-1)

data_sun_total = data_sun[['region','cases']]
data_sun_total = pd.melt(data_sun_total, id_vars = ['region'],
                         value_vars = ['cases'])

data_sun_cases = pd.melt(data_sun, id_vars = ['region'],
                         value_vars = ['active','deaths','recovered'])
data_sun_cases = data_sun_cases.sort_values('region')

data_sun_active = pd.melt(data_sun, id_vars = ['active'],
                          value_vars = ['hospitalised','ICU', 'home_isolation'])
data_sun_active['active'] = data_sun_active['active'].apply(str)
data_sun_active = data_sun_active.sort_values('active')

for r in data_sun_total.region:
    data_sun_cases[data_sun_cases.region == r] = \
    data_sun_cases[data_sun_cases.region == r]\
    .replace('active','active ('+ r +')')

a = [ 'active ('+ i +')' for i in list(data_sun_total.region)] * 3
a.sort()

labels = list(data_sun_total.region)   + list(data_sun_cases.variable) \
                                       + list(data_sun_active.variable)
parents = ['Spain'] * len(data_sun_total.region) \
                                       + list(data_sun_cases['region']) + a
values = list(data_sun_total['value'].abs()) \
         + list(data_sun_cases['value'].abs()) \
         + list(data_sun_active['value'].abs())

#=============================================================================#

df_sunburst = {
    'labels' : labels,
    'parents' : parents,
    'values' : values,
}

#=============================================================================#
#=============================================================================#
