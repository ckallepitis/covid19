# Imports:
import pandas as pd
import numpy as np
import io
import requests

#=============================================================================#
#                                                                             #
#                         Data preparation - Worldwide                        #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Load Covid19 data ==========

def get_covid_data():
    url = 'https://github.com/microsoft/Bing-COVID-19-Data/raw/master/data/Bing-COVID19-Data.csv'
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    df = df.drop(['ID', 'ISO2'], axis=1)
    df.columns = df.columns.str.lower()
    df.rename(columns={
        'updated': 'date',
        'confirmed': 'cases',
        'confirmedchange': 'daily_cases',
        'deathschange': 'daily_deaths',
        'recoveredchange': 'daily_recovered',
        'country_region': 'country'
    },
              inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.date
    from elements import ADD_DATA
    df = pd.merge(df, pd.DataFrame(ADD_DATA), on='iso3', how='left')
    df = df[df.adminregion1.isna()]
    c_list = df[(df.continent.isna()) | (df.popData2019.isna()) |
                (df.iso3.isna())].country.unique()
    df = df[~df.country.isin(c_list)].reset_index(drop=True)

    return df

#=============================================================================#
# ========== Line Graph ==========

def get_line_graph_data(df):

    df_line_data = df[[
        'date', 'cases', 'daily_cases', 'deaths', 'daily_deaths', 'country',
        'continent', 'popData2019', 'iso3'
    ]]
    df_line_data.daily_cases[df_line_data.daily_cases < 0] = 0
    df_line_data.daily_deaths[df_line_data.daily_deaths < 0] = 0
    df_line_data['daily deaths; 7-day rolling average'] = df_line_data.groupby(['country']).rolling(window=7)\
                                                                      .daily_deaths.mean().reset_index()\
                                                                      .set_index('level_1').daily_deaths
    df_line_data['daily cases; 7-day rolling average'] = df_line_data.groupby(['country']).rolling(window=7)\
                                                                     .daily_cases.mean().reset_index()\
                                                                     .set_index('level_1').daily_cases
    df_line_data['daily cases; 7-day rolling average'] = df_line_data['daily cases; 7-day rolling average']\
                                                       .fillna(0).astype(np.int64, errors='ignore')
    df_line_data['daily deaths; 7-day rolling average'] = df_line_data['daily deaths; 7-day rolling average']\
                                                         .fillna(0).astype(np.int64, errors='ignore')

    return df_line_data

#=============================================================================#
# ========== Bar-plot ==========

def get_bar_plot_data(df):

    df_bar_data = df[['cases','deaths','country','popData2019']]\
    .groupby('country').agg({'cases':'sum','deaths':'sum','popData2019':'first'}).reset_index()

    top = 15

    # Cases
    df_bar_cases = df_bar_data[['cases','country']]
    df_bar_cases = df_bar_cases.sort_values('cases', ascending = False).iloc[:top,:].reset_index(drop=True)
    df_bar_cases['cat'] = 'cases'

    # Cases per million pop
    df_bar_cases_per_mill = pd.DataFrame(df_bar_data['country'])
    df_bar_cases_per_mill['cases per million people'] = ((df_bar_data['cases'] * 1e6) /\
                                      df_bar_data['popData2019']).astype(int)
    df_bar_cases_per_mill['cat'] = 'cases per million people'

    df_bar_cases_per_mill = df_bar_cases_per_mill\
    .sort_values('cases per million people', ascending = False).iloc[:top,:].reset_index(drop=True)

    # Deaths
    df_bar_deaths = df_bar_data[['deaths','country']]
    df_bar_deaths = df_bar_deaths.sort_values('deaths', ascending = False).iloc[:top,:].reset_index(drop=True)
    df_bar_deaths['cat'] = 'deaths'

    # Cases per million pop
    df_bar_deaths_per_mill = pd.DataFrame(df_bar_data['country'])
    df_bar_deaths_per_mill['deaths per million people'] = ((df_bar_data['deaths'] * 1e6) /\
                                      df_bar_data['popData2019']).astype(int)
    df_bar_deaths_per_mill['cat'] = 'deaths per million people'

    df_bar_deaths_per_mill = df_bar_deaths_per_mill\
    .sort_values('deaths per million people', ascending = False).iloc[:top,:].reset_index(drop=True)

    # Create subplots data
    cols = ['cases','deaths','cases per million people','deaths per million people']
    df_bar = pd.concat([df_bar_cases,df_bar_deaths,df_bar_cases_per_mill,df_bar_deaths_per_mill],axis=0)
    df_bar = df_bar.assign(values=df_bar[cols].sum(1)).drop(cols, 1).reset_index(drop=True)

    return df_bar

#=============================================================================#
# ========== Geo-plot ==========

def get_geo_data(df):
    
    df_geo = df[['date','continent','country','iso3','cases','daily_cases','deaths','daily_deaths']]
    df_geo = df_geo[df_geo.country != 'South Sudan']
    df_geo['date'] = pd.to_datetime(df_geo['date'], format = '%Y-%m-%d')
    df_geo = df_geo.sort_values(['date','country']).reset_index(drop = True)
    
    df_geo = df_geo.set_index(['date', 'country'])\
    .unstack().transform(lambda v: v.ffill()).transform(lambda v: v.ffill())\
    .transform(lambda v: v.bfill()).asfreq('D')\
    .stack().sort_index(level=1).reset_index()

    df_geo.daily_cases[df_geo.daily_cases < 0] = 0
    df_geo.daily_deaths[df_geo.daily_deaths < 0] = 0
    df_geo.date = df_geo.date.dt.strftime("%d %b %y").astype(str)

    return df_geo

#=============================================================================#
#                                                                             #
#                         Data preparation - Spain                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Load Covid19 data ==========

def get_covid_data_Spain():

    url = 'https://github.com/datadista/datasets/raw/master/COVID%2019/'+\
          'ccaa_covid19_datos_sanidad_nueva_serie.csv'
    url_hospitals = 'https://github.com/datadista/datasets/raw/master/COVID%2019/'+\
                    'ccaa_ingresos_camas_convencionales_uci.csv'

    s = requests.get(url).content
    sh = requests.get(url_hospitals).content

    df_spain = pd.read_csv(io.StringIO(s.decode('utf-8')),error_bad_lines=False)
    df_spain = df_spain.drop(['cod_ine'],axis=1)
    df_hospitals = pd.read_csv(io.StringIO(sh.decode('utf-8')),error_bad_lines=False)
    df_hospitals = df_hospitals[['Fecha','CCAA','Altas COVID últimas 24 h']]
    df_spain = pd.merge(df_spain, df_hospitals, how='left', on=['Fecha', 'CCAA'])

    df_spain.columns = ['Date', 'Region', 'Cases', 'Hospitalised', 'ICU',
                        'Deaths', 'Recovered_24h']

    df_spain = df_spain.replace({'Region':
                                 {'Andalucía':'Andalusia',
                                  'Aragón':'Aragon',
                                  'Baleares':'Balearic Islands',
                                  'C. Valenciana':'Valencia',
                                  'Canarias':'Canary Islands',
                                  'Castilla La Mancha':'Castille-La Mancha',
                                  'Castilla y León':'Castille and Leon',
                                  'Cataluña':'Catalonia',
                                  'Navarra':'Navarre',
                                  'País Vasco':'Basque Country'}})

    df_spain = df_spain.query('Region != ["Ceuta","Melilla"]')
    df_spain.Date = pd.to_datetime(df_spain.Date, format = '%Y/%m/%d')

    df_spain['Daily_Cases'] = df_spain['Cases']
    df_spain['Daily_Deaths'] = df_spain['Deaths']
    df_spain.Daily_Cases[df_spain.Daily_Cases < 0] = 0
    df_spain.Daily_Deaths[df_spain.Daily_Deaths < 0] = 0
    df_spain['Cases'] = df_spain.groupby(['Region']).cumsum().Cases
    df_spain['Deaths'] = df_spain.groupby(['Region']).cumsum().Deaths

    df_spain['Daily Cases; 7-day rolling average'] = df_spain.groupby(['Region']).rolling(window=7)\
                                                                      .Daily_Cases.mean().reset_index()\
                                                                      .set_index('level_1').Daily_Cases
    df_spain['Daily Deaths; 7-day rolling average'] = df_spain.groupby(['Region']).rolling(window=7)\
                                                                     .Daily_Deaths.mean().reset_index()\
                                                                     .set_index('level_1').Daily_Deaths
    df_spain['Daily Cases; 7-day rolling average'] = df_spain['Daily Cases; 7-day rolling average']\
                                                       .fillna(0).astype(np.int64, errors='ignore')
    df_spain['Daily Deaths; 7-day rolling average'] = df_spain['Daily Deaths; 7-day rolling average']\
                                                         .fillna(0).astype(np.int64, errors='ignore')

    from elements import SPAIN_GEOLOCATIONS
    df_loc = pd.DataFrame(SPAIN_GEOLOCATIONS)

    df_spain = pd.merge(df_spain, df_loc, how='inner', on='Region')

    return df_spain

#=============================================================================#
# ========== Line Graph ==========


#=============================================================================#
# ========== Spain Map ==========

def get_geo_Spain_data(df_spain):

    df_geo_spain = df_spain[['Date','Region','Lat','Long','Cases','Deaths']]
    df_geo_spain = df_geo_spain.set_index(['Date', 'Region'])\
    .unstack().transform(lambda v: v.ffill()).transform(lambda v: v.ffill())\
    .transform(lambda v: v.bfill()).asfreq('D')\
    .stack().sort_index(level=1).reset_index()
    df_geo_spain = df_geo_spain.merge(df_spain[['Date','Region',
                             'Daily Cases; 7-day rolling average','Daily Deaths; 7-day rolling average']],
                   how='left', on=['Date','Region']).fillna(0)
    df_geo_spain.Date = df_geo_spain.Date.dt.strftime("%d %b %y").astype(str)
    df_geo_spain = df_geo_spain.rename(columns={'Daily Cases; 7-day rolling average':'Daily Cases',
                                                'Daily Deaths; 7-day rolling average':'Daily Deaths'})

    return df_geo_spain
#=============================================================================#
# ========== Sunburst chart ==========

def get_sunburst_data(df_spain):
    df_sunburst = df_spain[['Date', 'Region', 'Cases', 'Deaths','Hospitalised','ICU','Recovered_24h']]

    dd = df_spain[['Date','Recovered_24h']].dropna().loc[df_spain[['Date','Recovered_24h']].dropna().Date.idxmin()].Date
    d = df_spain[df_spain.Date == dd]
    d['Recovered'] = d.Cases - d.Deaths - d.Hospitalised - d.ICU
    df_sunburst.loc[(df_sunburst.Date == dd),'Recovered_24h'] = d.Recovered * 2
    df_sunburst['Recovered'] = df_sunburst.groupby(['Region']).cumsum().Recovered_24h

    df_sunburst = df_sunburst.drop('Recovered_24h',axis=1)
    df_sunburst['Active'] = df_sunburst['Cases'] - df_sunburst['Deaths'] - df_sunburst['Recovered']
    df_sunburst['Home_Isolation'] = df_sunburst['Active'] - df_sunburst['Hospitalised'] - df_sunburst['ICU']
    df_sunburst = df_sunburst.fillna(method='ffill')
    df_sunburst = df_sunburst.loc[df_sunburst.groupby('Region').Date.idxmax()]
    df_sunburst['Country'] = 'Spain'

    df_sunburst = df_sunburst.drop('Date',axis=1).melt(id_vars=['Country','Region'],
                                                       value_vars=['Deaths','Hospitalised','ICU','Home_Isolation',
                                                                   'Recovered'])
    df_sunburst['Cases'] = 'Active Cases'
    df_sunburst[(df_sunburst.variable == 'Deaths') | (df_sunburst.variable == 'Recovered')] = \
    df_sunburst.query("variable == ['Deaths','Recovered']").replace('Active Cases','Closed Cases')

    return df_sunburst

#=============================================================================#
#=============================================================================#
