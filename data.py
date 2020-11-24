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

def get_covid_data(url):
    #url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    df.rename(columns = {'countriesAndTerritories': 'country'}, inplace = True)
    df.country = df.country.replace('United_States_of_America','USA')
    df.country = df.country.replace('United_Kingdom','UK')
    df.sort_values(['countryterritoryCode',
                    'year','month','day']).reset_index(drop = True)
    df['dateRep'] = pd.to_datetime(df['dateRep'], format = '%d/%m/%Y').dt.date
    df = df[~df.country.isin(['Cases_on_an_international_conveyance_Japan',
                              'Wallis_and_Futuna'])]
    df = df[['dateRep','day','month','year','cases','deaths',
         'country', 'geoId','countryterritoryCode','continentExp','popData2019']]

    return df

#=============================================================================#
# ========== Line Graph ==========

def get_line_graph_data(df):

    df_line_data = df[['dateRep', 'cases', 'deaths', 'country','continentExp', 'popData2019']]
    df_line_data = df_line_data.rename(columns = {'dateRep': 'day', 'continentExp': 'continent'}, inplace = False)
    df_line_data = df_line_data.groupby(['continent','country','day']).sum()

    df_line_data['daily_cases'] = df_line_data['cases']
    df_line_data['daily_deaths'] = df_line_data['deaths']
    df_line_data['cases'] = df_line_data.groupby(['country']).cumsum().cases
    df_line_data['deaths'] = df_line_data.groupby(['country']).cumsum().deaths

    df_line_data = df_line_data.reset_index()
    df_line_data[(df_line_data['daily_cases'] < 0)|(df_line_data['daily_deaths'] < 0)] = 0

    return df_line_data

#=============================================================================#
# ========== Bar-plot ==========

def get_bar_plot_data(df):

    df_bar_data = df[['cases','deaths','country','popData2019']]\
    .groupby('country').agg({'cases':'sum','deaths':'sum','popData2019':'first'}).reset_index()

    # Cases
    df_bar_cases = df_bar_data[['cases','country']]
    df_bar_cases = df_bar_cases.sort_values('cases', ascending = False).iloc[:11,:].reset_index(drop=True)
    df_bar_cases['cat'] = 'cases'

    # Cases per million pop
    df_bar_cases_per_mill = pd.DataFrame(df_bar_data['country'])
    df_bar_cases_per_mill['cases_per_mill'] = ((df_bar_data['cases'] * 1e6) /\
                                      df_bar_data['popData2019']).astype(int)
    df_bar_cases_per_mill['cat'] = 'cases_per_mill'

    df_bar_cases_per_mill = df_bar_cases_per_mill\
    .sort_values('cases_per_mill', ascending = False).iloc[:11,:].reset_index(drop=True)

    # Deaths
    df_bar_deaths = df_bar_data[['deaths','country']]
    df_bar_deaths = df_bar_deaths.sort_values('deaths', ascending = False).iloc[:11,:].reset_index(drop=True)
    df_bar_deaths['cat'] = 'deaths'

    # Cases per million pop
    df_bar_deaths_per_mill = pd.DataFrame(df_bar_data['country'])
    df_bar_deaths_per_mill['deaths_per_mill'] = ((df_bar_data['deaths'] * 1e6) /\
                                      df_bar_data['popData2019']).astype(int)
    df_bar_deaths_per_mill['cat'] = 'deaths_per_mill'

    df_bar_deaths_per_mill = df_bar_deaths_per_mill\
    .sort_values('deaths_per_mill', ascending = False).iloc[:11,:].reset_index(drop=True)

    # Create subplots data
    cols = ['cases','deaths','cases_per_mill','deaths_per_mill']
    df_bar = pd.concat([df_bar_cases,df_bar_deaths,df_bar_cases_per_mill,df_bar_deaths_per_mill],axis=0)
    df_bar = df_bar.assign(values=df_bar[cols].sum(1)).drop(cols, 1).reset_index(drop=True)


    return df_bar

#=============================================================================#
# ========== Geo-plot ==========

def get_geo_data(df):

    df_geo = df[df.year != 2019].drop(['day','year','month','geoId','popData2019'], axis = 1)
    df_geo = df_geo.reset_index(drop = True)
    df_geo.columns = ['day','cases','deaths','country','iso_alpha','continent']
    df_geo['day'] = pd.to_datetime(df_geo['day'], format = '%Y-%m-%d')
    df_geo = df_geo.sort_values(['day','country']).reset_index(drop = True)

    df_geo['daily_cases'] = df_geo[['country', 'day','cases']]\
    .groupby(['country', 'day']).sum()\
    .reset_index()\
    .sort_values(['day','country'])\
    .reset_index(drop = True).cases

    df_geo['cases'] = df_geo[['country', 'day','cases']]\
    .groupby(['country', 'day']).sum()\
    .groupby(level = 0).cumsum().reset_index()\
    .sort_values(['day','country'])\
    .reset_index(drop = True).cases

    df_geo['daily_deaths'] = df_geo[['country', 'day','deaths']]\
    .groupby(['country', 'day']).sum()\
    .reset_index()\
    .sort_values(['day','country'])\
    .reset_index(drop = True).deaths

    df_geo['deaths'] = df_geo[['country', 'day','deaths']]\
    .groupby(['country', 'day']).sum()\
    .groupby(level = 0).cumsum().reset_index()\
    .sort_values(['day','country'])\
    .reset_index(drop = True).deaths

    df_geo.daily_cases[df_geo.daily_cases < 0] = 0
    df_geo.daily_deaths[df_geo.daily_deaths < 0] = 0
    df_geo.day = df_geo.day.dt.strftime("%b %d").astype(str)

    return df_geo

#=============================================================================#
#                                                                             #
#                         Data preparation - Spain                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Load Covid19 data ==========

def get_covid_data_Spain():

    #urls = 'https://covid19.isciii.es/resources/serie_historica_acumulados.csv'

    url_cases = 'https://github.com/datadista/datasets/raw/master/COVID%2019/'+\
                'ccaa_covid19_datos_isciii_nueva_serie.csv'

    url_hospitals = 'https://github.com/datadista/datasets/raw/master/COVID%2019/'+\
                    'ccaa_ingresos_camas_convencionales_uci.csv'

    url_deaths = 'https://github.com/datadista/datasets/raw/master/COVID%2019/'+\
                 'ccaa_covid19_fallecidos_por_fecha_defuncion_nueva_serie_original.csv'

    sc = requests.get(url_cases).content
    sh = requests.get(url_hospitals).content
    sd = requests.get(url_deaths).content

    df_cases = pd.read_csv(io.StringIO(sc.decode('utf-8')),error_bad_lines=False)
    df_cases = df_cases.rename(columns={"fecha": "Fecha", "ccaa": "CCAA"})
    df_cases = df_cases.drop(['num_casos_prueba_pcr','num_casos_prueba_test_ac',
                              'num_casos_prueba_otras','num_casos_prueba_desconocida'],axis=1)
    df_hospitals = pd.read_csv(io.StringIO(sh.decode('utf-8')),error_bad_lines=False)
    df_deaths = pd.read_csv(io.StringIO(sd.decode('utf-8')),error_bad_lines=False)

    df_spain = pd.merge(df_cases, df_deaths, on=['Fecha', 'CCAA'])
    df_spain = pd.merge(df_spain, df_hospitals, how='left', on=['Fecha', 'CCAA'])
    df_spain = df_spain.drop(['cod_ine_x','cod_ine_y'],axis=1)

    df_spain.columns = ['Date', 'Region_Name', 'Cases', 'Deaths', 'Hospitalised',
                        'Hospital_Beds', 'ICU', 'ICU_Beds', 'Hospitalised_24h',
                        'Recovered_24h']

    df_spain = df_spain.replace({'Region_Name':
                                 {'Andalucía':'Andalusia',
                                  'Aragón':'Aragon',
                                  'Baleares':'Balearic Islands',
                                  'C. Valenciana':'Valencia',
                                  'Canarias':'Canary Islands',
                                  'Castilla La Mancha':'Castille-La Mancha',
                                  'Castilla y León':'Castille and Leon',
                                  'Cataluña':'Catalonia',
                                  'País Vasco':'Basque Country'}})

    df_spain = df_spain.query('Region_Name != ["Ceuta","Melilla"]')
    #df_spain['Recovered'] = df_spain['Cases'] - (df_spain['Deaths'] + df_spain['Hospitalised'] + df_spain['ICU'] )
    df_spain.Date = pd.to_datetime(df_spain.Date, format = '%Y/%m/%d')

    df_spain['Daily_Cases'] = df_spain['Cases']
    df_spain['Daily_Deaths'] = df_spain['Deaths']
    df_spain['Cases'] = df_spain.groupby(['Region_Name']).cumsum().Cases
    df_spain['Deaths'] = df_spain.groupby(['Region_Name']).cumsum().Deaths

    from elements import SPAIN_GEOLOCATIONS
    df_loc = pd.DataFrame(SPAIN_GEOLOCATIONS)

    df_spain = pd.merge(df_spain, df_loc, how='inner', on='Region_Name')

    return df_spain

#=============================================================================#
# ========== Line Graph ==========


#=============================================================================#
# ========== Spain Map ==========

def get_geo_Spain_data(df_spain):

    df_geo_spain = df_spain[['Date','Region_Name','Long','Lat','Cases','Daily_Cases','Deaths','Daily_Deaths']].fillna(0)
    df_geo_spain = df_geo_spain.sort_values('Date')
    df_geo_spain.Date = df_geo_spain.Date.dt.strftime("%b %d").astype(str)

    return df_geo_spain
#=============================================================================#
# ========== Sunburst chart ==========

def get_sunburst_data(df_spain):
    df_sunburst = df_spain[['Date', 'Region_Name', 'Cases', 'Deaths','Hospitalised','ICU','Recovered_24h']]

    dd = df_spain[['Date','Recovered_24h']].dropna().loc[df_spain[['Date','Recovered_24h']].dropna().Date.idxmin()].Date
    d = df_spain[df_spain.Date == dd]
    d['Recovered'] = d.Cases - d.Deaths - d.Hospitalised - d.ICU
    df_sunburst.loc[(df_sunburst.Date == dd),'Recovered_24h'] = d.Recovered * 2
    df_sunburst['Recovered'] = df_sunburst.groupby(['Region_Name']).cumsum().Recovered_24h

    df_sunburst = df_sunburst.drop('Recovered_24h',axis=1)
    df_sunburst['Active'] = df_sunburst['Cases'] - df_sunburst['Deaths'] - df_sunburst['Recovered']
    df_sunburst['Home_Isolation'] = df_sunburst['Active'] - df_sunburst['Hospitalised'] - df_sunburst['ICU']
    df_sunburst = df_sunburst.fillna(method='ffill')
    df_sunburst = df_sunburst.loc[df_sunburst.groupby('Region_Name').Date.idxmax()]
    df_sunburst['Country'] = 'Spain'

    df_sunburst = df_sunburst.drop('Date',axis=1).melt(id_vars=['Country','Region_Name'],
                                                       value_vars=['Deaths','Hospitalised','ICU','Home_Isolation',
                                                                   'Recovered'])
    df_sunburst['Cases'] = 'Active Cases'
    df_sunburst[(df_sunburst.variable == 'Deaths') | (df_sunburst.variable == 'Recovered')] = \
    df_sunburst.query("variable == ['Deaths','Recovered']").replace('Active Cases','Closed Cases')

    return df_sunburst

#=============================================================================#
#=============================================================================#
