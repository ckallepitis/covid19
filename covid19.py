###############################################################################
###                                                                         ###
###                           Covid19 Dashboard                             ###
###                                                                         ###
###############################################################################

# Perform imports here:
import pandas as pd
import numpy as np
import io
import requests
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px

# Launch the application:

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

###############################################################################
###                                                                         ###
###                       Data preparation - Worldwide                      ###
###                                                                         ###
###############################################################################

###############################################################################
### Load Covid19 data

url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
df.sort_values(['countryterritoryCode',
                'year','month','day']).reset_index(drop=True)
df = df[['dateRep','day','month','year','cases','deaths',
         'countriesAndTerritories', 'geoId','countryterritoryCode']]
df['dateRep'] = pd.to_datetime(df['dateRep'], format = '%d/%m/%Y').dt.date

###############################################################################
### Cases

df_pivot_c= df.pivot(index='dateRep', columns='countriesAndTerritories',
                                                                 values='cases')
df_pivot_c_cumsum = df_pivot_c.cumsum()

df_norm_100case = []
for column in df_pivot_c_cumsum.columns:
    df_norm_100case.append(df_pivot_c_cumsum[column]\
                                      [df_pivot_c_cumsum[column] >= 100].values)

df_norm_100case = pd.DataFrame(df_norm_100case).T

df_norm_100case.columns = df_pivot_c_cumsum.columns

df_norm_100case = df_norm_100case.dropna(how='all',axis = 1)

y_cutoff = 700000
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

Lines_c = pd.DataFrame(every_1day,columns=['every 1 day'],dtype=np.float64)
Lines_c['every 2 days'] = pd.DataFrame(every_2days)
Lines_c['every 3 days'] = pd.DataFrame(every_3days)
Lines_c['every 4 days'] = pd.DataFrame(every_4days)
Lines_c['every week'] = pd.DataFrame(every_week)

###############################################################################
### Deaths

df_pivot_d = df.pivot(index='dateRep', columns='countriesAndTerritories',
                                                                values='deaths')
df_pivot_d_cumsum = df_pivot_d.cumsum()

df_norm_10death = []
for column in df_pivot_d_cumsum.columns:
    df_norm_10death.append(df_pivot_d_cumsum[column]\
                                       [df_pivot_d_cumsum[column] >= 10].values)

df_norm_10death = pd.DataFrame(df_norm_10death).T

df_norm_10death.columns = df_pivot_d_cumsum.columns

df_norm_10death = df_norm_10death.dropna(how='all',axis = 1)

y_cutoff = 50000
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

Lines_d = pd.DataFrame(every_1day,columns=['every 1 day'],dtype=np.float64)
Lines_d['every 2 days'] = pd.DataFrame(every_2days)
Lines_d['every 3 days'] = pd.DataFrame(every_3days)
Lines_d['every week'] = pd.DataFrame(every_week)

###############################################################################
### Deaths - Daily

countries = df_pivot_d.agg(max) > 3
countries = (countries[countries == True].index.values)
df_pivot_3death = df_pivot_d[countries]

df_norm_3death = []
for column in df_pivot_3death.columns:
    temp = df_pivot_3death[column].reset_index(drop=True)
    df_norm_3death.append(df_pivot_3death[column].iloc[temp[temp > 3]\
                                                             .index[0]:].values)

df_norm_3death = pd.DataFrame(df_norm_3death).T
df_norm_3death.columns = df_pivot_3death.columns
df_norm_3death = df_norm_3death.rolling(7).mean().iloc[6:]\
                                                         .reset_index(drop=True)
df_norm_3death = df_norm_3death.dropna(how='all',axis = 1)
df_norm_3death = df_norm_3death.drop(['Algeria','Iraq','Philippines',
                                      'San_Marino','Slovenia'],axis=1)

###############################################################################
### Geo-plot

df_geo = df[df.year != 2019].drop(['day','year','month','geoId'],axis=1)
df_geo = df_geo[df_geo.countriesAndTerritories != \
            'Cases_on_an_international_conveyance_Japan'].reset_index(drop=True)
df_geo.columns = ['day','cases','deaths','country','iso_alpha']
df_geo['continent'] = 0
df_geo['day'] = pd.to_datetime(df_geo['day'], format = '%Y/%m/%d').dt.dayofyear
df_geo = df_geo.sort_values(['day','country']).reset_index(drop = True)
df_geo['cases'] = df_geo[['country', 'day','cases']]\
                                    .groupby(['country', 'day']).sum()\
                                    .groupby(level=0).cumsum().reset_index()\
                                    .sort_values(['day','country'])\
                                    .reset_index(drop = True).cases
df_geo['deaths'] = df_geo[['country', 'day','deaths']]\
                                    .groupby(['country', 'day']).sum()\
                                    .groupby(level=0).cumsum().reset_index()\
                                    .sort_values(['day','country'])\
                                    .reset_index(drop = True).deaths

Africa = ['Algeria', 'Mauritania', 'Burundi', 'Malawi', 'Guinea',
          'Guinea_Bissau', 'Gambia', 'Djibouti',  'Gabon', 'Benin',
          'Burkina_Faso', 'Liberia', 'Libya', 'Cameroon', 'Canada',
          'Cote_dIvoire', 'Uganda', 'Namibia', 'Central_African_Republic',
          'Chad', 'Congo', 'Togo', 'Somalia', 'Sudan', 'Sierra_Leone',
          'Democratic_Republic_of_the_Congo', 'Egypt', 'Ghana', 'Kenya',
          'Botswana', 'Mali', 'Mauritius', 'Morocco', 'Nigeria', 'Senegal',
          'South_Africa', 'Zambia', 'Zimbabwe', 'Tunisia', 'Angola',
          'United_Republic_of_Tanzania', 'Rwanda', 'Cape_Verde',
          'Equatorial_Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Madagascar',
          'Mozambique', 'Niger', 'Seychelles']

Americas = ['Argentina', 'Montserrat', 'Maldives', 'Guyana', 'Haiti',
            'British_Virgin_Islands', 'Bonaire, Saint Eustatius and Saba',
            'Nicaragua', 'Bolivia', 'Sint_Maarten', 'Saint_Lucia',
            'Saint_Vincent_and_the_Grenadines', 'Brazil', 'Canada', 'Chile',
            'Barbados', 'Belize', 'Bermuda', 'Paraguay', 'Greenland',
            'Colombia', 'Costa_Rica', 'Cuba', 'Dominican_Republic',
            'Turks_and_Caicos_islands', 'Ecuador', 'Honduras', 'Mexico',
            'Panama', 'Peru', 'Suriname', 'United_States_Virgin_Islands',
            'Puerto_Rico', 'Trinidad_and_Tobago', 'United_States_of_America',
            'Jamaica', 'Saint_Barthelemy', 'Saint_Kitts_and_Nevis', 'Uruguay',
            'Venezuela', 'Anguilla', 'Antigua_and_Barbuda', 'Aruba', 'Bahamas',
            'Guatemala', 'Grenada', 'Cayman_Islands', 'Curaçao', 'Dominica',
            'El_Salvador', 'Falkland_Islands_(Malvinas)']

Asia = ['Afghanistan', 'Algeria', 'Armenia', 'Azerbaijan',
        'Bahrain','Timor_Leste', 'Guam', 'Brunei_Darussalam', 'Cambodia',
        'Bangladesh', 'Bhutan', 'Syria', 'Nepal', 'Laos',
        'Cases_on_an_international_conveyance_Japan', 'China', 'India',
        'Mongolia', 'Myanmar', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan',
        'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Lebanon', 'Malaysia',
        'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Russia',
        'Saudi_Arabia', 'Singapore', 'South_Korea', 'Sri_Lanka', 'Taiwan',
        'Thailand', 'Fiji', 'French_Polynesia', 'Tunisia', 'Turkey',
        'United_Arab_Emirates', 'Uzbekistan', 'Vietnam']

Europe = ['Albania', 'Holy_See', 'Andorra', 'Austria', 'Belarus', 'Belgium',
          'Monaco', 'Bosnia_and_Herzegovina', 'Bulgaria',  'Croatia', 'Cyprus',
          'Czech_Republic', 'Denmark', 'Estonia', 'Finland', 'France',
          'Liechtenstein', 'Faroe_Islands', 'Germany', 'Greece', 'Guernsey',
          'Hungary', 'Iceland', 'Ireland', 'Isle_of_Man', 'Italy', 'Jersey',
          'Kosovo', 'Latvia', 'Lithuania', 'Luxembourg',  'Malta', 'Moldova',
          'Montenegro', 'Netherlands', 'North_Macedonia', 'Norway', 'Poland',
          'Portugal',  'Romania', 'Russia', 'San_Marino', 'Serbia', 'Slovakia',
          'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
          'United_Kingdom', 'Georgia', 'Gibraltar']

Oceania = ['Australia','New_Zealand','New_Caledonia','Northern_Mariana_Islands',
           'Papua_New_Guinea']

for i, country in enumerate(df_geo.country):
    if country in Africa:
        df_geo['continent'].iloc[i] = 'Africa'
    elif country in Americas:
        df_geo['continent'].iloc[i] = 'Americas'
    elif country in Asia:
        df_geo['continent'].iloc[i] = 'Asia'
    elif country in Europe:
        df_geo['continent'].iloc[i] = 'Europe'
    elif country in Oceania:
        df_geo['continent'].iloc[i] = 'Oceania'

###############################################################################
###                                                                         ###
###                       Data preparation - Spain                          ###
###                                                                         ###
###############################################################################

###############################################################################
### Load Covid19 data
urls = "https://covid19.isciii.es/resources/serie_historica_acumulados.csv"
sp = requests.get(urls).content
dfs = pd.read_csv(io.StringIO(sp.decode('latin1')))
df_loc = pd.DataFrame([
    ('AN','Andalusia',-4.5000000, 37.6000000),
    ('AR','Aragon',-1.0000000, 41.0000000),
    ('AS','Asturias', -6.0000000, 43.3333333),
    ('IB','Balearic Islands', 2.4334717, 39.2226789),
    ('PV','Basque Country', -2.7500000, 43.0000000),
    ('CN','Canary Islands', -15.5000000, 28.0000000),
    ('CB','Cantabria', -4.0000000, 43.3333333),
    ('CL','Castille and Leon', -4.2500000, 41.6666667),
    ('CM','Castille-La Mancha', -3.0000000, 39.5000000),
    ('CT','Catalonia', 1.8676758, 41.8204551),
    ('EX','Extremadura', -6.0000000, 39.0000000),
    ('GA','Galicia', -7.8662109, 42.7550796),
    ('RI','La Rioja', -2.5000000, 42.2500000),
    ('MD','Madrid', -3.6906338, 40.4252581),
    ('MC','Murcia', -1.5000000, 38.0000000),
    ('NC','Navarre', -1.6666667, 42.7500000),
    ('VC','Valencia', -0.7500000, 39.5000000)],
   columns=['region','region_name','long','lat'])

dfs = dfs.iloc[:-2,:]
dfs.columns = ['region','date','cases','hospitalised','ICU','deaths','recovered']
dfs.date = pd.to_datetime(dfs.date, format='%d/%m/%Y')
dfs = dfs.sort_values(['region','date'])
dfs = pd.merge(dfs, df_loc[['region','region_name']], how='inner', on='region')

###############################################################################
#### Line Graph

coloursp = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
'#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96',
'#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF']

###############################################################################
### Cases

dfs_pivot_c= dfs.pivot(index='date', columns='region_name',values='cases')

dfs_norm_10case = []
for column in dfs_pivot_c.columns:
    dfs_norm_10case.append(dfs_pivot_c[column][dfs_pivot_c[column] >= 10].values)

dfs_norm_10case = pd.DataFrame(dfs_norm_10case).T

dfs_norm_10case.columns = dfs_pivot_c.columns

dfs_norm_10case = dfs_norm_10case.dropna(how='all',axis = 1)

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

Lines_cs = pd.DataFrame(every_1day,columns=['every 1 day'],dtype=np.float64)
Lines_cs['every 2 days'] = pd.DataFrame(every_2days)
Lines_cs['every 3 days'] = pd.DataFrame(every_3days)
Lines_cs['every 4 days'] = pd.DataFrame(every_4days)
Lines_cs['every week'] = pd.DataFrame(every_week)

###############################################################################
### Deaths

dfs_pivot_d = dfs.pivot(index='date', columns='region_name',values='deaths')

dfs_norm_3death = []
for column in dfs_pivot_d.columns:
    dfs_norm_3death.append(dfs_pivot_d[column][dfs_pivot_d[column] >= 3].values)

dfs_norm_3death = pd.DataFrame(dfs_norm_3death).T

dfs_norm_3death.columns = dfs_pivot_d.columns

dfs_norm_3death = dfs_norm_3death.dropna(how='all',axis = 1)

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

Lines_ds = pd.DataFrame(every_1day,columns=['every 1 day'],dtype=np.float64)
Lines_ds['every 2 days'] = pd.DataFrame(every_2days)
Lines_ds['every 3 days'] = pd.DataFrame(every_3days)
Lines_ds['every week'] = pd.DataFrame(every_week)

###############################################################################
### Deaths - Daily

dfs_pivot_d_daily = dfs_pivot_d.diff()
dfs_pivot_d_daily.loc['2020-03-08'] = dfs_pivot_d.loc['2020-03-08']

regions = dfs_pivot_d_daily.agg(max) > 3
regions = (regions[regions == True].index.values)
dfs_pivot_3death = dfs_pivot_d_daily[regions]

dfs_norm_3death_roll = []
for column in dfs_pivot_3death.columns:
    temp = dfs_pivot_3death[column].reset_index(drop=True)
    dfs_norm_3death_roll.append(dfs_pivot_3death[column].iloc[temp[temp > 3]\
                                                             .index[0]:].values)

dfs_norm_3death_roll = pd.DataFrame(dfs_norm_3death_roll).T
dfs_norm_3death_roll.columns = dfs_pivot_3death.columns
dfs_norm_3death_roll = dfs_norm_3death_roll.rolling(7).mean().iloc[6:]\
                                                         .reset_index(drop=True)
dfs_norm_3death_roll = dfs_norm_3death_roll.dropna(how='all',axis = 1)

###############################################################################
### Spain Map

df_geos = pd.merge(dfs, df_loc.drop('region_name',axis=1),\
                                             how='inner', on='region').fillna(0)
datas = df_geos[df_geos.date == df_geos.date.unique()[-1]]

fig_spain_map = go.Figure(go.Scattergeo(
                                        lon = datas['long'],
                                        lat = datas['lat'],
                                        text = datas['region_name']+': '\
                                                    +datas['cases'].astype(str),
                                        mode = 'markers',
                                        marker = dict(size = datas['cases']/40,
                                                    color = coloursp,
                                                    line_width=0.5,
                                                    sizemode = 'area'),
                                        )
                           )

fig_spain_map.add_trace(go.Scattergeo(
                                        lon = datas['long'],
                                        lat = datas['lat'],
                                        text = datas['region'],
                                        mode = 'text',
                                        marker = dict(size = 0,
                                                    color = coloursp,
                                                    line_width=0.0,
                                                    sizemode = 'area'),
                                        hoverinfo='skip',
                                        ))

fig_spain_map.update_layout(
                    width = 600,
                    height = 400,
                    margin = dict(t=0, l=0, r=0, b=0),
                    showlegend=False
                 )

fig_spain_map.update_geos(fitbounds="locations")
###############################################################################
### Pie chart

data_sun = datas[['region','cases','hospitalised',
                  'ICU','deaths','recovered']].copy()
data_sun['active'] = pd.Series(data_sun.loc[:,'cases'] -\
                              ( data_sun.loc[:,'deaths'] \
                              + data_sun.loc[:,'recovered']))
data_sun['home_isolation'] = pd.Series(data_sun.loc[:,'active']\
                                       - ( data_sun.loc[:,'hospitalised']))
data_sun = data_sun.reset_index(drop=True)
data_sun['home_isolation'][7] = data_sun.loc[:,'home_isolation'][7] * (-1)
data_sun['home_isolation'][14] = data_sun.loc[:,'home_isolation'][14] * (-1)

data_sun_total = data_sun[['region','cases']]
data_sun_total = pd.melt(data_sun_total, id_vars=['region'],
                                         value_vars=['cases'])

data_sun_cases = pd.melt(data_sun, id_vars=['region'],
                                   value_vars=['active','deaths','recovered'])
data_sun_cases = data_sun_cases.sort_values('region')

data_sun_active = pd.melt(data_sun, id_vars=['active'],
                                    value_vars=['hospitalised','ICU',
                                                'home_isolation'])
data_sun_active['active'] = data_sun_active['active'].apply(str)
data_sun_active = data_sun_active.sort_values('active')

for r in df_loc.region:
    data_sun_cases[data_sun_cases.region == r] = \
    data_sun_cases[data_sun_cases.region == r]\
    .replace('active','active ('+r+')')

a = [ 'active ('+ i +')' for i in list(df_loc.region)] * 3
a.sort()

labels = list(data_sun_total.region) + list(data_sun_cases.variable) \
                                     + list(data_sun_active.variable)
parents = ['Spain'] * len(data_sun_total.region) \
        + list(data_sun_cases['region']) + a
values= list(data_sun_total['value']) + list(data_sun_cases['value']) \
                                      + list(data_sun_active['value'])

fig_pie_chart =go.Figure(go.Sunburst(labels = labels,
                                     parents = parents,
                                     values= values,
                                     outsidetextfont = {"size": 20,
                                                        "color": "#377eb8"},
                                     insidetextfont = {"size": 10},))

fig_pie_chart.update_layout(width = 400,height = 400,
                            margin = dict(t=0, l=0, r=0, b=0))

###############################################################################
###                                                                         ###
###                               Layout                                    ###
###                                                                         ###
###############################################################################

app.layout = dbc.Container([
###############################################################################
### Row 1
    dbc.Row([
            ###################################################################
            ### Title
            html.Div([
                    html.H1('Coronavirus COVID-19 Dashboard',
                            style={'font-family': 'Helvetica',
                                   'margin-top': '25', 'margin-bottom': '0'}),
                    html.P(\
                'Data normalised to allow comparison between countries/regions',
                            style={'font-family': 'Helvetica',
                                   'font-size': '100%', 'width': '80%'})
                     ]),#Div
            ###################################################################
            ### Select Coutnries - Multi-Select Dropdown
            html.Div([
                ###############################################################
                ### Worldwide Title
                html.Hr(),
                html.H2('Worldwide',
                       style={'font-family': 'Helvetica',
                              'margin-top': '25', 'margin-bottom': '20'}),
                html.P('Select Countries:'),

                dcc.RadioItems(
                    id = 'country_set',
                    options=[
                        {'label': ' !', 'value': '!'},
                        {'label': ' All', 'value': 'All'},
                        {'label': ' Africa', 'value': 'Africa'},
                        {'label': ' Americas', 'value': 'Americas'},
                        {'label': ' Asia', 'value': 'Asia'},
                        {'label': ' Europe', 'value': 'Europe'},
                        {'label': ' Oceania', 'value': 'Oceania'},
                        {'label': ' Strands', 'value': 'Strands'} ],
                    value='!',
                    labelStyle={'display': 'inline-block','margin': '5px'}),

                dcc.Dropdown(
                    id = 'Selected_Countries',
                    multi=True ),
                ],#Div
                style={'font-family': 'Helvetica','margin-top': '10',
                'font-size': '100%', 'width': '100%'}),
            ],align='center'),#Row
###############################################################################
### Row 2
    dbc.Row([
        #######################################################################
        ### Col1
        dbc.Col([
            dbc.Card([
                dbc.FormGroup([
                        #######################################################
                        #### Metric Selector
                        html.Div([
                        dbc.Label('Metric:'),
                        dcc.RadioItems(\
                           id = 'yaxis_scale',
                           options=[{'label': ' Linear', 'value': 'lin'},
                                    {'label': ' Logarithmic', 'value': 'log'} ],
                           value='log',
                           labelStyle={'display': 'inline-block',
                                       'margin': '5px'}
                           ),#RadioItems
                           ],style={'font-family': 'Helvetica','margin-top': '10',
                                  'font-size': '100%', 'width': 150})#Div
                              ]),#FormGroup
                dbc.FormGroup([
                        #######################################################
                        #### Data Selector
                        html.Div([
                        dbc.Label('Cases:'),
                        dcc.RadioItems(id = 'Data_to_show',
                                       options=[
                           {'label': ' Confirmed', 'value': 'cases'},
                           {'label': ' Deaths', 'value': 'deaths'},
                           {'label': ' Daily Deaths', 'value': 'daily_deaths'}],
                                       value='cases',
                                       labelStyle={'display': 'inline-block',
                                                   'margin': '5px'}),#RadioItems
                                 ],
                           style={'font-family': 'Helvetica','margin-top': '10',
                                  'font-size': '100%', 'width': 150})#Div,
                              ]),#FormGroup
                     ], body=True)#Card
                ], md=3),#Col
        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
                ###############################################################
                #### Line Graph
                dcc.Graph(id='Line_Graph',
                          #hoverData={'points': [{'customdata': 'Japan'}]}
                          ),
                     ],style={'font-family': 'Helvetica','margin-top': '10',
                            'font-size': '100%', 'width': '80%'}),#Div
               ], md=8),#Col
            ], align='center'),#Row

###############################################################################
### Row 3
    dbc.Row([
        #######################################################################
        ### Col1
        dbc.Col([
            dbc.Card([
                dbc.FormGroup([
                    ###########################################################
                    #### Map Data Selector
                    html.Div([
                    dbc.Label('Cases:'),
                    dcc.RadioItems(id = 'Map_cases',
                                   options=[
                                      {'label': ' Confirmed', 'value': 'cases'},
                                       {'label': ' Deaths', 'value': 'deaths'}],
                                   value='cases',
                                   labelStyle={'display': 'inline-block',
                                               'margin': '5px'}),#RadioItems
                             ],style={'font-family': 'Helvetica',
                                      'margin-top': '10',
                                      'font-size': '100%', 'width': 150})#Div
                ]),#FormGroup
            ],body=True),#Card
        ], md=3),#Col
        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
                ###############################################################
                #### Map Graph
                dcc.Graph(id = 'Maps')
                ], style={'font-family': 'Helvetica','margin-top': '10',
                          'font-size': '100%'}),
        ], md=8),#Col
    ], align='center'),#Row

###############################################################################
###                                 Spain                                   ###
###############################################################################

###############################################################################
### Row 4
    dbc.Row([
        html.Div([
           ####################################################################
           ### Spain Title
           html.Hr(),
           html.H2('Spain',
                   style={'font-family': 'Helvetica',
                          'margin-top': '25', 'margin-bottom': '20'}),

           ###################################################################
           ### Select Regions - Multi-Select Dropdown

           html.P('Select Region:'),
           dcc.RadioItems(
               id = 'region_set',
               options=[
                   {'label': ' MD,CT', 'value': 'MD,CT'},
                   {'label': ' All', 'value': 'All'},],
               value='All',
               labelStyle={'display': 'inline-block','margin': '5px'}),
           dcc.Dropdown(id = 'Selected_Regions', multi=True ,
                        value = ['Madrid','Catalonia']),
                  ],#Div
                  style={'font-family': 'Helvetica','margin-top': '10',
                  'font-size': '100%', 'width': '100%'}),
             ],align='center'),#Row

###############################################################################
### Row 5
    dbc.Row([
        #######################################################################
        ### Col1
        dbc.Col([
            dbc.Card([
                dbc.FormGroup([
                        #######################################################
                        #### Metric Selector
                        html.Div([
                        dbc.Label('Metric:'),
                        dcc.RadioItems(\
                           id = 'yaxis_scale_s',
                           options=[{'label': ' Linear', 'value': 'lin'},
                                    {'label': ' Logarithmic', 'value': 'log'} ],
                           value='log',labelStyle={'display': 'inline-block',
                                       'margin': '5px'}),#RadioItems
                               ],style={'font-family': 'Helvetica',
                                        'margin-top': '10',
                                        'font-size': '100%', 'width': 150})#Div
                              ]),#FormGroup
                dbc.FormGroup([
                        #######################################################
                        #### Data Selector
                        html.Div([
                        dbc.Label('Cases:'),
                        dcc.RadioItems(id = 'Data_to_show_s',
                                       options=[
                           {'label': ' Confirmed', 'value': 'cases'},
                           {'label': ' Deaths', 'value': 'deaths'},
                           {'label': ' Daily Deaths', 'value': 'daily_deaths'}],
                                       value='cases',
                                       labelStyle={'display': 'inline-block',
                                                   'margin': '5px'}),#RadioItems
                                ],style={'font-family': 'Helvetica',
                                         'margin-top': '10',
                                         'font-size': '100%', 'width': 150})#Div
                              ]),#FormGroup
                     ], body=True)#Card
                ], md=3),#Col
        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
                ###############################################################
                #### Line Graph
                dcc.Graph(id='Line_Graph_s'),
                     ],style= {'width': '80%'}),#Div
               ], md=8),#Col
            ], align='center'),#Row

###############################################################################
### Row 6
    dbc.Row([
        #######################################################################
        ### Col1
        dbc.Col([
            html.Div([
                ###############################################################
                #### Pie Chart
                dbc.Label('Click on regions for more info:'),
                dcc.Graph(id = 'Pie_Chart',figure=fig_pie_chart),
                ],
      style={'font-family': 'Helvetica','margin-top':'10',
             'font-size': '100%'}),

        ], width=5),#Col

        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
               ################################################################
               #### Map Graph
               dbc.Label('Confirmed Cases per Region:'),
               dcc.Graph(id = 'Spain_Map',figure=fig_spain_map),
                ],
      style={'font-family': 'Helvetica','margin-top': '10',
             'font-size': '100%', 'width': '100%'}),
            #html.Div([
            #   dcc.Slider(min=0,
            #              max=len(df_geos.date.dt.date.astype(str).unique()),
            #              marks={i:'{}'.format(date) for i,date in \
            #              enumerate(df_geos.date.dt.date.astype(str).unique())},
            #              value=len(df_geos.date.dt.date.astype(str).unique()),
            #              updatemode='drag',
            #              )
            #         ],
            #         style ={'font-family': 'Helvetica','margin-top': '10',
            #                 #'writing-mode': 'vertical-lr',
            #                 'text-orientation': 'sideways'})
        ],width=5),#Col
    ], justify="between"),#Row

###############################################################################
### Row 7
    dbc.Row([
        html.Div([
           ####################################################################
           ### Epilogue
           html.P('Source: European Centre for Disease Prevention and Control & Instituto de Salud Carlos III'),
           html.P('data updated daily',
                    style={'font-size': '80%'})
            ],style={'font-family': 'Helvetica','font-size': '80%',
                                                            'margin-top': '10'})
            ],align='center'),#Row

])#Container

###############################################################################
###                                                                         ###
###                         Callbacks & Functions                           ###
###                                                                         ###
###############################################################################

###############################################################################
#### Map Selector
@app.callback(Output('Maps', 'figure'),
             [Input('Map_cases', 'value')])
def callback_Map_cases(Map_cases_value):

    if Map_cases_value == 'cases':
        fig = px.scatter_geo(df_geo, locations="iso_alpha", color="continent",
                                     hover_name="country", size="cases",
                                     animation_frame="day",size_max=100,
                                     projection="natural earth")

    elif Map_cases_value == 'deaths':
        fig = px.scatter_geo(df_geo, locations="iso_alpha", color="continent",
                                     hover_name="country", size="deaths",
                                     animation_frame="day",size_max=50,
                                     projection="natural earth")

    fig.update_layout(  width = 800,
                        height = 500,
                        margin = dict(t=0, l=0, r=0, b=0)
                     )
    return fig

###############################################################################
#### Region Selector
@app.callback(Output('Selected_Regions', 'options'),
             [Input('Data_to_show_s', 'value')])
def callback_Data_to_show_s(Data_to_show_s_value):

    if Data_to_show_s_value == 'cases':
        list = [{'label':i,'value':i} for i in dfs_norm_10case.columns]
        return list
    elif Data_to_show_s_value == 'deaths':
        list = [{'label':i,'value':i} for i in dfs_norm_3death.columns]
        return list
    elif Data_to_show_s_value == 'daily_deaths':
        list = [{'label':i,'value':i} for i in dfs_norm_3death_roll.columns]
        return list

@app.callback(Output('Selected_Regions', 'value'),
             [Input('region_set', 'value')])
def callback_region_set(region_set_value):

    if region_set_value == 'All':
        list = dfs_norm_10case.columns
        return list
    elif region_set_value == 'MD,CT':
        list = ['Madrid','Catalonia']
        return list
    return

###############################################################################
#### Country Selector
@app.callback(Output('Selected_Countries', 'options'),
             [Input('Data_to_show', 'value')])
def callback_Data_to_show(Data_to_show_value):

    if Data_to_show_value == 'cases':
        list = [{'label':i,'value':i}for i in df_norm_100case.columns]
        return list
    elif Data_to_show_value == 'deaths':
        list = [{'label':i,'value':i}for i in df_norm_10death.columns]
        return list
    elif Data_to_show_value == 'daily_deaths':
        list = [{'label':i,'value':i}for i in df_norm_3death.columns]
        return list

@app.callback(Output('Selected_Countries', 'value'),
             [Input('country_set', 'value')])
def callback_country_set(country_set_value):

    if country_set_value == 'All':
        list = df_norm_100case.columns
        return list
    elif country_set_value == '!':
        list = ['Spain', 'Italy', 'United_Kingdom', 'China', 'South_Korea',
          'Japan', 'Taiwan', 'Germany', 'United_States_of_America']
        return list
    elif country_set_value == 'Africa':
        list = ['Algeria', 'Burkina_Faso', 'Cameroon', 'Canada', 'Cote_dIvoire',
          'Democratic_Republic_of_the_Congo', 'Egypt', 'Ghana', 'Kenya',
          'Mauritius', 'Morocco', 'Nigeria', 'Senegal', 'South_Africa',
          'Tunisia']
        return list
    elif country_set_value == 'Americas':
        list = ['Argentina', 'Bolivia', 'Brazil', 'Canada', 'Chile',
            'Colombia', 'Costa_Rica', 'Cuba', 'Dominican_Republic',
            'Ecuador', 'Honduras', 'Mexico', 'Panama', 'Peru',
            'Puerto_Rico', 'Trinidad_and_Tobago', 'United_States_of_America',
            'Uruguay',  'Venezuela']
        return list
    elif country_set_value == 'Asia':
        list = ['Afghanistan', 'Algeria', 'Armenia', 'Azerbaijan', 'Bahrain',
        'Brunei_Darussalam', 'Cambodia',
        'Cases_on_an_international_conveyance_Japan', 'China', 'India',
        'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan',
        'Kuwait', 'Kyrgyzstan',  'Lebanon', 'Malaysia', 'Oman', 'Pakistan',
        'Palestine', 'Philippines',  'Qatar', 'Russia','Saudi_Arabia',
        'Singapore', 'South_Korea', 'Sri_Lanka', 'Taiwan', 'Thailand',
        'Tunisia', 'Turkey', 'United_Arab_Emirates','Uzbekistan', 'Vietnam']
        return list
    elif country_set_value == 'Europe':
        list = ['Albania',  'Andorra', 'Austria', 'Belarus', 'Belgium',
          'Bosnia_and_Herzegovina', 'Bulgaria',  'Croatia', 'Cyprus',
          'Czech_Republic', 'Denmark', 'Estonia', 'Finland', 'France',
          'Faroe_Islands','Germany', 'Greece', 'Guernsey', 'Hungary',
          'Iceland', 'Ireland', 'Isle_of_Man', 'Italy', 'Jersey', 'Kosovo',
          'Latvia', 'Lithuania', 'Luxembourg',  'Malta', 'Moldova',
          'Montenegro',  'Netherlands', 'North_Macedonia', 'Norway', 'Poland',
          'Portugal',  'Romania', 'Russia', 'San_Marino', 'Serbia', 'Slovakia',
          'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
          'United_Kingdom']
        return list
    elif country_set_value == 'Oceania':
        list = ['Australia','New_Zealand']
        return list
    elif country_set_value == 'Strands':
        list = ['Argentina','Malaysia','Spain','United_States_of_America']
        return list
    return

###############################################################################
#### Line Graph

colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
'#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96',
'#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
'#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
'#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
'#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA',
'#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
'#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
'#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B',
'#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
'#FECB52', '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
'#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96',
'#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
'#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
'#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
'#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#636EFA',
'#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
'#FF97FF', '#FECB52', '#636EFA', '#EF553B']

# Reference line color
color_balck = 'rgb(67,67,67)'

@app.callback(Output('Line_Graph', 'figure'),
              [Input('Selected_Countries', 'value'),
              Input('yaxis_scale', 'value'),
              Input('Data_to_show', 'value')])

def callback_Line_Graph(Selected_Countries_value,
                        yaxis_scale_value,
                        Data_to_show_value):

    ###########################################################################
    #### Confirmed
    if Data_to_show_value == 'cases':
        Columns = list(df_norm_100case.columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_norm_100case[Countries]
        lines = Lines_c
        title = \
      'Comulative number of confirmed cases, by number of days since 100th case'
        x_title = 'Number of days since 100th case'
        y_title = 'Comulative number of cases'
        range_x = [0,80]
        range_y = [90,400000]
        range_logy = [np.log10(90),np.log10(1e6)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': colours[i]},
                           opacity=0.6,
                           name=column) )

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': colours[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':12},
                                    showarrow=False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
                           hovertemplate = 'Doubles',
                           mode='lines',
                           opacity=0.5,
                           line={'color': color_balck,
                                 'dash':'dash'},
                           name=column) )

    ###########################################################################
    #### Deaths
    elif Data_to_show_value == 'deaths':
        Columns = list(df_norm_10death.columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_norm_10death[Countries]
        lines = Lines_d
        title = \
        'Comulative number of deaths, by number of days since 10th death'
        x_title = 'Number of days since 10th death'
        y_title = 'Comulative number of deaths'
        range_x = [0,80]
        range_y = [9,2e4]
        range_logy = [np.log10(9),np.log10(1e5)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': colours[i]},
                           opacity=0.6,
                           name=column) )

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': colours[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':12},
                                    showarrow=False))

        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
                           hovertemplate = 'Doubles',
                           mode='lines',
                           opacity=0.5,
                           line={'color': color_balck,
                                 'dash':'dash'},
                           name=column) )

    ###########################################################################
    #### Daily Deaths
    elif Data_to_show_value == 'daily_deaths':
        Columns = list(df_norm_3death.columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_norm_3death[Countries]
        title = 'Daily deaths'
        x_title = 'Number of days since 3 daily deaths first recorded'
        y_title = 'Number of daily deaths (7 day rolling average)'
        range_x = [0,80]
        range_y = [1,1500]
        range_logy = [np.log10(1),np.log10(3e3)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': colours[i]},
                           opacity=0.6,
                           name=column) )

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': colours[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':12},
                                    showarrow=False))

    if yaxis_scale_value == 'log':
        for i in annotations:
            i['y'] = np.log10(i['y'])
    elif yaxis_scale_value == 'linear':
        annotations = annotations

    return {
        'data': plot_data,
        'layout': go.Layout(
            title = title,
            width = 900,
            height = 500,
            xaxis={'title': x_title,
                   'range': range_x
                   },
            yaxis={'title': y_title,
                 'range': range_logy if yaxis_scale_value == 'log' else range_y,
                   'type': 'log' if yaxis_scale_value == 'log' else 'linear'},
            hovermode='x',
            annotations = annotations
            )
           }

###############################################################################
###                            Spain Graph                                  ###
###############################################################################

@app.callback(Output('Line_Graph_s', 'figure'),
              [Input('Selected_Regions', 'value'),
              Input('yaxis_scale_s', 'value'),
              Input('Data_to_show_s', 'value')])

def callback_Line_Graph(Selected_Regions_value,
                        yaxis_scale_s_value,
                        Data_to_show_s_value):

    ###########################################################################
    #### Confirmed
    if Data_to_show_s_value == 'cases':

        Columns = list(dfs_norm_10case.columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = dfs_norm_10case[Regions]
        lines = Lines_cs
        title = 'Comulative number of confirmed cases, by number of days since 10th case'
        x_title = 'Number of days since 10th case'
        y_title = 'Comulative number of cases'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': coloursp[i]},
                           opacity=0.6,
                           name= column ))

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': coloursp[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=df_loc.set_index('region_name')\
                                    .region.loc[column],
                                    font={'family':'Arial','size':12},
                                    showarrow=False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
                           hovertemplate = 'Doubles',
                           mode='lines',
                           opacity=0.5,
                           line={'color': color_balck,
                                 'dash':'dash'},
                           name=column) )

    ###########################################################################
    #### Deaths
    elif Data_to_show_s_value == 'deaths':

        Columns = list(dfs_norm_3death.columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = dfs_norm_3death[Regions]
        lines = Lines_ds
        title = 'Comulative number of deaths, by number of days since 3rd death'
        x_title = 'Number of days since 3rd death'
        y_title = 'Comulative number of deaths'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': coloursp[i]},
                           opacity=0.6,
                           name=column) )

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': coloursp[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=df_loc.set_index('region_name')\
                                    .region.loc[column],
                                    font={'family':'Arial','size':12},
                                    showarrow=False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
                           hovertemplate = 'Doubles',
                           mode='lines',
                           opacity=0.5,
                           line={'color': color_balck,
                                 'dash':'dash'},
                           name=column) )

    ###########################################################################
    #### Daily Deaths
    elif Data_to_show_s_value == 'daily_deaths':

        Columns = list(dfs_norm_3death_roll.columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = dfs_norm_3death_roll[Regions]
        title = 'Daily deaths'
        x_title = 'Number of days since 3 daily deaths first recorded'
        y_title = 'Number of daily deaths (7 day rolling average)'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           hovertemplate = '%{y:.0f}',
                           mode='lines',
                           line = {'color': coloursp[i]},
                           opacity=0.6,
                           name=column) )

            plot_data.append(
                go.Scatter(x = [data[column].dropna().index[-1]],
                           y = [list(data[column].dropna())[-1]],
                           mode='markers',
                           marker = {'color': coloursp[i],
                                     'size' : 5},
                           showlegend=False,
                           opacity=0.6,
                           hoverinfo='skip',) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=df_loc.set_index('region_name')\
                                    .region.loc[column],
                                    font={'family':'Arial','size':12},
                                    showarrow=False))

    for i in range(0,plot_data.__len__()):
        fig.add_trace(plot_data[i])

    if yaxis_scale_s_value == 'log':
        for i in annotations:
            i['y'] = np.log10(i['y'])
    elif yaxis_scale_s_value == 'linear':
        annotations = annotations

    fig.update_layout(  title = title,
                    width = 900,
                    height = 500,
                    xaxis={'title': x_title,},
                    yaxis={'title': y_title,
                   'type': 'log' if yaxis_scale_s_value == 'log' else 'linear'},
                    hovermode='x',
                    annotations = annotations,
                    template = 'plotly_white'
                 )
    return fig

###############################################################################

if __name__ == '__main__':
    app.run_server()
