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
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Launch the application:
app = dash.Dash()
server = app.server

###############################################################################
###                                                                         ###
###                           Load DataFrame                                ###
###                                                                         ###
###############################################################################

###############################################################################
### Load Covid19 data

url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
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

y_cutoff = 600000
x_start = 100
Double_1day = [x_start]
Double_2days = [x_start]
Double_3days = [x_start]
Double_4days = [x_start]
Double_week = [x_start]
for i in range(1,77):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        Double_1day.append(x)
    else: Double_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        Double_2days.append(y)
    else: Double_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        Double_3days.append(z)
    else: Double_3days.append(z)
    v = x_start * ((2**(1/4)) ** i)
    if v > y_cutoff:
        v = None
        Double_4days.append(v)
    else: Double_4days.append(v)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        Double_week.append(w)
    else: Double_week.append(w)

Lines_c = pd.DataFrame(Double_1day,columns=['Double_1day'],dtype=np.float64)
Lines_c['Double_2days'] = pd.DataFrame(Double_2days)
Lines_c['Double_3days'] = pd.DataFrame(Double_3days)
Lines_c['Double_4days'] = pd.DataFrame(Double_4days)
Lines_c['Double_week'] = pd.DataFrame(Double_week)

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

y_cutoff = 21000
x_start = 10
Double_1day = [x_start]
Double_2days = [x_start]
Double_3days = [x_start]
Double_week = [x_start]
for i in range(1,77):
    x = x_start * (2 ** i)
    if x > y_cutoff:
        x = None
        Double_1day.append(x)
    else: Double_1day.append(x)
    y = x_start * ((2 ** (1/2)) ** i)
    if y > y_cutoff:
        y = None
        Double_2days.append(y)
    else: Double_2days.append(y)
    z = x_start * ((2**(1/3)) ** i)
    if z > y_cutoff:
        z = None
        Double_3days.append(z)
    else: Double_3days.append(z)
    w = x_start * ((2**(1/7)) ** i)
    if w > y_cutoff:
        w = None
        Double_week.append(w)
    else: Double_week.append(w)

Lines_d = pd.DataFrame(Double_1day,columns=['Double_1day'],dtype=np.float64)
Lines_d['Double_2days'] = pd.DataFrame(Double_2days)
Lines_d['Double_3days'] = pd.DataFrame(Double_3days)
Lines_d['Double_week'] = pd.DataFrame(Double_week)

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
###                                                                         ###
###                               Layout                                    ###
###                                                                         ###
###############################################################################

app.layout = html.Div([
    ###########################################################################
    ### Title
    html.Div([
            html.H1('Coronavirus COVID-19 Dashboard',
                    style={'font-family': 'Helvetica',
                           "margin-top": "25", "margin-bottom": "0"}),
            html.P('Data normalised to allow comparison between countries',
                    style={'font-family': 'Helvetica',
                           "font-size": "120%", "width": "80%"})
             ]),

    ###########################################################################
    #### Selectors
    html.Div([
            ###################################################################
            ### Select Coutnries - Multi-Select Dropdown
            html.Div([
                    html.P('Select Countries:'),
                    dcc.Dropdown(
                        id = 'Selected_Countries',
                        multi=True )
                    ],
                    style={'margin-top': '10'}),

            ###################################################################
            #### Country Sets - Radio Item
            html.Div([
                    dcc.RadioItems(
                        id = 'country_set',
                        options=[
                            {'label': 'All', 'value': 'All'},
                            {'label': 'Africa', 'value': 'Africa'},
                            {'label': 'Americas', 'value': 'Americas'},
                            {'label': 'Asia', 'value': 'Asia'},
                            {'label': 'Europe', 'value': 'Europe'},
                            {'label': 'Oceania', 'value': 'Oceania'},
                            {'label': 'Strands', 'value': 'Strands'} ],
                        value='Strands' ),
                    ],
                    style={'margin-top': '10'}),

            ###################################################################
            #### Y axis scale - Radio Item
            html.Div([
                    html.P('Y axis scale:'),
                    dcc.RadioItems(
                        id = 'yaxis_scale',
                        options=[
                            {'label': 'Linear', 'value': 'lin'},
                            {'label': 'Log', 'value': 'log'} ],
                        value='log' )
                    ],
                    style={'margin-top': '10'}),

            ###################################################################
            #### Cases/Deaths - Radio Item
            html.Div([
                    html.P('Data to show:'),
                    dcc.RadioItems(
                        id = 'Data_to_show',
                        options=[
                            {'label': 'Confirmed Cases', 'value': 'cases'},
                            {'label': 'Deaths', 'value': 'deaths'},
                            {'label': 'Daily Deaths', 'value': 'daily_deaths'}],
                        value='cases' )
                    ],
                    style={'margin-top': '10'})
            ]),

    ###########################################################################
    #### Graphs
    html.Div([
            ###################################################################
            #### Line Graph
            dcc.Graph(id="Line_Graph",
                      hoverData={'points': [{'customdata': 'Japan'}]})

            ]),
])

###############################################################################
###                                                                         ###
###                         Callbacks & Functions                           ###
###                                                                         ###
###############################################################################

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
        title = 'Comulative number of confirmed cases, by number of days since 100th case'
        x_title = 'Number of days since 100th case'
        y_title = 'Comulative number of cases'
        range_x = [0,80]
        range_y = [0,300000]
        range_logy = [np.log10(0),np.log10(600000)]

        plot_data = []
        annotations = []
        for column in data.columns:
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           #text={'Country': column,
                           #     'Confirmed cases': data[column].max()},
                           mode='lines+markers',
                           opacity=0.7,
                           marker={'size': 5,
                                   'opacity': 0.7,},
                           name=column) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':10},
                                    showarrow=False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
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
        title = 'Comulative number of deaths, by number of days since 10th death'
        x_title = 'Number of days since 10th death'
        y_title = 'Comulative number of deaths'
        range_x = [0,40]
        range_y = [0,15000]
        range_logy = [np.log10(0),np.log10(15000)]

        plot_data = []
        annotations = []
        for column in data.columns:
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           #text={'Country': column,
                           #      'Confirmed cases': data[column].max()},
                           mode='lines+markers',
                           opacity=0.7,
                           marker={'size': 5,
                                   'opacity': 0.7,},
                           name=column) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':10},
                                    showarrow=False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(x=list(lines.index.values),
                           y=lines[column],
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
        range_x = [0,75]
        range_y = [0,1500]
        range_logy = [np.log10(0),np.log10(3000)]

        plot_data = []
        annotations = []
        for column in data.columns:
            plot_data.append(
                go.Scatter(x=list(data.index.values),
                           y=data[column],
                           #text=,
                           mode='lines+markers',
                           opacity=0.7,
                           marker={'size': 5,
                                   'opacity': 0.7,},
                           name=column) )

            annotations.append(dict(xref='x',yref='y',
                                    x=data[column].dropna().index[-1],
                                    y=list(data[column].dropna())[-1],
                                    xanchor='left', yanchor='middle',
                                    text=column,
                                    font={'family':'Arial','size':10},
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
