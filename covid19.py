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
###                           Load DataFrame                                ###
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
                        'Data normalised to allow comparison between countries',
                            style={'font-family': 'Helvetica',
                                   'font-size': '100%', 'width': '80%'})
                     ]),#Div
            ###################################################################
            ### Select Coutnries - Multi-Select Dropdown
            html.Div([
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
                    style={'font-family': 'Helvetica','margin-top': '10'}),
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
                        dbc.Label('Metric:'),
                        dcc.RadioItems(\
                           id = 'yaxis_scale',
                           options=[{'label': ' Linear', 'value': 'lin'},
                                    {'label': ' Logarithmic', 'value': 'log'} ],
                           value='log' ),#RadioItems
                              ]),#FormGroup
                dbc.FormGroup([
                        #######################################################
                        #### Data Selector
                        dbc.Label('Cases:'),
                        dcc.RadioItems(id = 'Data_to_show',
                                       options=[
                           {'label': ' Confirmed', 'value': 'cases'},
                           {'label': ' Deaths', 'value': 'deaths'},
                           {'label': ' Daily Deaths', 'value': 'daily_deaths'}],
                                       value='cases' ),#RadioItems
                              ]),#FormGroup
                     ], body=True)#Card
                ], md=2),#Col
        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
                ###############################################################
                #### Line Graph
                dcc.Graph(id='Line_Graph',
                          hoverData={'points': [{'customdata': 'Japan'}]}),
                     ]),#Div
               ], md=9),#Col
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
                    dbc.Label('Cases:'),
                    dcc.RadioItems(id = 'Map_cases',
                                   options=[
                                      {'label': ' Confirmed', 'value': 'cases'},
                                       {'label': ' Deaths', 'value': 'deaths'}],
                                   value='cases' ),#RadioItems
                ]),#FormGroup
            ],body=True),#Card
        ], md=2),#Col
        #######################################################################
        ### Col2
        dbc.Col([
            html.Div([
                ###############################################################
                #### Map Graph
                dcc.Graph(id = 'Maps')
                ],
      style={'font-family': 'Helvetica','font-size': '80%','margin-top': '10'}),
        ], md=9),#Col
    ], align='center'),#Row

###############################################################################
### Row 4
    dbc.Row([
        html.Div([
           ####################################################################
           ### Epilogue
           html.P('Source: European Centre for Disease Prevention and Control'),
           html.P('data updates every day at 23:59 CET',
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
                                     animation_frame="day",
                                     projection="natural earth")

    elif Map_cases_value == 'deaths':
        fig = px.scatter_geo(df_geo, locations="iso_alpha", color="continent",
                                     hover_name="country", size="deaths",
                                     animation_frame="day",
                                     projection="natural earth")
    return fig

###############################################################################
#### Country Selector
@app.callback(Output('Selected_Countries', 'options'),
             [Input('Data_to_show', 'value')])
def callback_country_set(Data_to_show_value):

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
        list = df_norm_10death.columns
        return list
    elif country_set_value == '!':
        list = ['Spain', 'Italy', 'United_Kingdom', 'China', 'South_Korea',
          'Japan', 'Taiwan','France','Germany','United_States_of_America']
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
          'Portugal',  'Romania', 'Russia','San_Marino', 'Serbia', 'Slovakia',
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

def callback_Line_Graph(Selected_Countries_value,yaxis_scale_value,
                                                            Data_to_show_value):
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

if __name__ == '__main__':
    app.run_server(debug=True)
