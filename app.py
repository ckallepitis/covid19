#=============================================================================#
#                                                                             #
#                             Covid19 Dashboard                               #
#                                                                             #
#=============================================================================#

# Perform imports here:
import pandas as pd
pd.options.mode.chained_assignment = None  # default  = 'warn'
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

from elements import COLOURS, COLOURS_SPAIN
from data import df_traj, df_geo, df_bar
from data import df_traj_spain, df_geo_spain, df_sunburst

# Launch the application:
BS = 'https://stackpath.bootstrapcdn.com/'\
     'bootswatch/4.4.1/litera/bootstrap.min.css'
app = dash.Dash(external_stylesheets = [BS])
#app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server

#=============================================================================#
#                                                                             #
#                               Plots - Worldwide                             #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Bar-plot ==========

margin_l = 75

fig_bar_total_c = px.bar(
    df_bar['cases'],
    x = 'cases',
    y = 'country',
    text = 'cases',
    color = 'country',
    orientation = 'h',
    hover_data = ['country', 'cases'],
)
fig_bar_total_c.update_layout(
    margin = dict(t = 0, l = margin_l, r = 0, b = 0),
    showlegend = False,
    xaxis = {'title': ''},
    yaxis = {'title': ''},
    template = 'plotly_white',
)
fig_bar_total_d = px.bar(
    df_bar['deaths'],
    x = 'deaths',
    y = 'country',
    text = 'deaths',
    color = 'country',
    orientation = 'h',
    hover_data = ['country', 'deaths'],
)
fig_bar_total_d.update_layout(
    margin = dict(t = 0, l = margin_l, r = 0, b = 0),
    showlegend = False,
    xaxis = {'title': ''},
    yaxis = {'title': ''},
    template = 'plotly_white',
)

fig_bar_permill_c = px.bar(
    df_bar['cases_per_mill'],
    x = 'cases_per_mill',
    y = 'country',
    text = 'cases_per_mill',
    color = 'country',
    orientation = 'h',
    hover_data = ['country', 'cases_per_mill'],
)
fig_bar_permill_c.update_layout(
    margin = dict(t = 0, l = margin_l, r = 0, b = 0),
    showlegend = False,
    xaxis = {'title': ''},
    yaxis = {'title': ''},
    template = 'plotly_white',
)
fig_bar_permill_d = px.bar(
    df_bar['deaths_per_mill'],
    x = 'deaths_per_mill',
    y = 'country',
    text = 'deaths_per_mill',
    color = 'country',
    orientation = 'h',
    hover_data = ['country','deaths_per_mill'],
)
fig_bar_permill_d.update_layout(
    margin = dict(t = 0, l = margin_l, r = 0, b = 0),
    showlegend = False,
    xaxis = {'title': ''},
    yaxis = {'title': ''},
    template = 'plotly_white',
)

#=============================================================================#
#                                                                             #
#                               Plots - Spain                                 #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Spain Map ==========

# Create figure
fig_spain_map = go.Figure()

# Add traces, one for each slider step
for i in range(0,df_geo_spain.date.unique().__len__()):
    datas = df_geo_spain[df_geo_spain.date == df_geo_spain.date.unique()[i]]

    fig_spain_map.add_trace(
        go.Scattergeo(
            lon = datas['long'],
            lat = datas['lat'],
            text = datas['region_name']+': '+datas['cases'].astype(str),
            mode = 'markers',
            name = '',
            marker = dict(
            size = datas['cases']/40,
            color = COLOURS_SPAIN,
            line_width = 0.5,
            sizemode = 'area',
            )
        )
    )

    fig_spain_map.add_trace(
        go.Scattergeo(
            lon = datas['long'],
            lat = datas['lat'],
            text = datas['region'],
            textfont = {'size': i+1 if i < 12 else 12},
            mode = 'text',
            hoverinfo = 'skip',
        )
    )


# Create and add slider
steps = []
j = 0
for i in range(0,len(fig_spain_map.data),2):
    step = dict(
        label =  str(df_geo_spain.date.dt.month[j]) +'-'+ \
        str(df_geo_spain.date.dt.day[j]),
        method = 'restyle',
        args = ['visible', [False] * len(fig_spain_map.data)],)
    step['args'][1][i] = True  # Toggle i'th trace to 'visible'
    step['args'][1][i+1] = True
    steps.append(step)
    j += 1

sliders = [
    dict(
        #active = len(fig_spain_map.data)/2,
        currentvalue = {'prefix': 'Date: '},
        pad = {'t': 50},
        steps = steps,
    ),
]

fig_spain_map.update_layout(
    width = 600,
    height = 400,
    margin = dict(t = 0, l = 4, r = 0, b = 0),
    showlegend = False,
    sliders = sliders,
)

fig_spain_map.update_geos(fitbounds = 'locations')

#=============================================================================#
# ========== Sunburst plot ==========

fig_sunburst = go.Figure(
    go.Sunburst(
        labels = df_sunburst['labels'],
        parents = df_sunburst['parents'],
        values = df_sunburst['values'],
        outsidetextfont = {'size': 20, 'color': '#377eb8'},
        insidetextfont = {'size': 10},
    )
)

fig_sunburst.update_layout(
    width = 400,
    height = 400,
    margin = dict(t = 0, l = 0, r = 0, b = 0),
)

#=============================================================================#
#                                                                             #
#                                 Layout                                      #
#                                                                             #
#=============================================================================#

app.layout = dbc.Container([
    #=========================================================================#
    #=== Row 1
    dbc.Row([
        #=====================================================================#
        #=== Title
        html.Div([
            html.H1('Coronavirus COVID-19 Dashboard',
                    style = {
                        'font-family': 'Helvetica',
                        'margin-top': '25',
                        'margin-bottom': '0',
                    },
                   ),#H1
            html.P('Data normalised to allow '\
                   'comparison between countries/regions',
                   style = {
                       'font-family': 'Helvetica',
                       'font-size': '100%',
                       'width': '80%',
                   },
                  ),#P
        ]),#Div

        #=====================================================================#
        #=== Select Coutnries - Multi-Select Dropdown
        html.Div([
            ##================================================================#
            #=== Worldwide Title
            html.Hr(),
            html.H2('Worldwide',
                    style = {
                        'font-family': 'Helvetica',
                        'margin-top': '25',
                        'margin-bottom': '20',
                    },
                   ),#H2
            html.P('Select Countries:'),
            dcc.RadioItems(
                id = 'country_set',
                options =
                [
                    {
                        'label': ' !',
                        'value': '!',
                    },
                    {
                        'label': ' All',
                        'value': 'All',
                    },
                    {
                        'label': ' Africa',
                        'value': 'Africa'},
                    {
                        'label': ' Americas',
                        'value': 'Americas',
                    },
                    {
                        'label': ' Asia',
                        'value': 'Asia',
                    },
                    {
                        'label': ' Europe',
                        'value': 'Europe',
                    },
                    {
                        'label': ' Oceania',
                        'value': 'Oceania',
                    },
                ],
                value = '!',
                labelStyle =
                {
                    'display': 'inline-block',
                    'margin': '5px',
                },
            ),#RadioItems
            dcc.Dropdown(
                id = 'Selected_Countries',
                multi = True,
            ),#Dropdown
        ],
            style = {
                'font-family': 'Helvetica',
                'margin-top': '10',
                'font-size': '100%',
                'width': '100%',
            }
        ),#Div
    ],
        align = 'center'
    ),#Row

    #=========================================================================#
    #=== Row 2
    dbc.Row(
        [
            #=================================================================#
            #=== Col1
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.FormGroup(
                                [
                                    #=========================================#
                                    #=== Metric Selector
                                    html.Div(
                                        [
                                            dbc.Label('Metric:'),
                                            dcc.RadioItems(
                                                id = 'yaxis_scale',
                                                options =
                                                [
                                                    {
                                                        'label': ' Linear',
                                                        'value': 'lin',
                                                    },
                                                    {
                                                        'label': ' Logarithmic',
                                                        'value': 'log',
                                                    },
                                                ],
                                                value = 'log',
                                                labelStyle =
                                                {
                                                    'display': 'inline-block',
                                                    'margin': '5px',
                                                },
                                            ),#RadioItems
                                        ],
                                        style =
                                        {
                                            'font-family': 'Helvetica',
                                            'margin-top': '10',
                                            'font-size': '100%',
                                            'width': 150,
                                        },
                                    ),#Div
                                ],
                            ),#FormGroup

                            dbc.FormGroup(
                                [
                                    #=========================================#
                                    #=== Data Selector
                                    html.Div(
                                        [
                                            dbc.Label('Cases:'),
                                            dcc.RadioItems(
                                                id = 'Data_to_show',
                                                options =
                                                [
                                                    {
                                                        'label': ' Confirmed',
                                                        'value': 'cases',
                                                    },
                                                    {
                                                        'label': ' Deaths',
                                                        'value': 'deaths',
                                                    },
                                                    {
                                                        'label':' Daily Deaths',
                                                        'value': 'daily_deaths',
                                                    },
                                                ],
                                                value= 'cases',
                                                labelStyle =
                                                {
                                                    'display': 'inline-block',
                                                    'margin': '5px',
                                                },
                                            ),#RadioItems
                                        ],
                                        style =
                                        {
                                            'font-family': 'Helvetica',
                                            'margin-top': '10',
                                            'font-size': '100%',
                                            'width': 150,
                                        },
                                    ),#Div,
                                ],
                            ),#FormGroup
                        ],
                        body = True,
                    ),#Card
                ],
                md = 3,
            ),#Col1

            #=================================================================#
            #=== Col2
            dbc.Col(
                [
                    html.Div(
                        [
                            #=================================================#
                            #=== Line Graph
                            dcc.Graph(id = 'Line_Graph'),
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                            'width': '80%',
                        },
                    ),#Div
                ],
                md = 8,
            ),#Col2
        ],
        align = 'center',
    ),#Row2

    #=========================================================================#
    #=== Row 3
    dbc.Row(
        [
            #=================================================================#
            #=== Col1
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.FormGroup(
                                [
                                    #=========================================#
                                    #=== Map Data Selector
                                    html.Div(
                                        [
                                            dbc.Label('Cases:'),
                                            dcc.RadioItems(
                                                id = 'Map_cases',
                                                options =
                                                [
                                                    {
                                                        'label': ' Confirmed',
                                                        'value': 'cases',
                                                    },
                                                    {'label': ' Deaths',
                                                     'value': 'deaths',
                                                    },
                                                ],
                                                value = 'cases',
                                                labelStyle =
                                                {
                                                    'display': 'inline-block',
                                                    'margin': '5px',
                                                },
                                            ),#RadioItems
                                        ],
                                        style =
                                        {
                                            'font-family': 'Helvetica',
                                            'margin-top': '10',
                                            'font-size': '100%',
                                            'width': 150,
                                        },
                                    ),#Div
                                ],
                            ),#FormGroup
                        ],
                        body = True,
                    ),#Card
                ],
                md = 3,
            ),#Col1

            #=================================================================#
            #=== Col2
            dbc.Col(
                [
                    html.Div(
                        [
                            #=================================================#
                            #=== Map Graph
                            dcc.Graph(id = 'Maps'),
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                md = 8,
            ),#Col2
        ],
        align = 'center',
    ),#Row3

    #=========================================================================#
    #=== Row 4
    dbc.Row(
        [
            #=================================================================#
            #=== Col1
            dbc.Col(
                [
                    html.Div(
                        [
                            dbc.Label('Confirmed Cases:'),
                            #=================================================#
                            #=== Bar plot - cases
                            dcc.Graph(
                                id = 'Bar_total_c',
                                figure = fig_bar_total_c,
                                config =
                                {
                                    'displayModeBar': False,
                                },
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                md = 3,
            ),#Col1

            #=================================================================#
            #=== Col2
            dbc.Col(
                [
                    html.Div(
                        [
                            dbc.Label('Deaths:'),
                            #=================================================#
                            #=== Bar plot - deaths
                            dcc.Graph(
                                id = 'Bar_total_d',
                                figure = fig_bar_total_d,
                                config =
                                {
                                    'displayModeBar': False,
                                },
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                md = 3,
            ),#Col2

            #=================================================================#
            #=== Col3
            dbc.Col(
                [
                    html.Div(
                        [
                            dbc.Label('Cases per million people:'),
                            #=================================================#
                            #=== Bar plot - cases per mill
                            dcc.Graph(
                                id = 'Bar_permill_c',
                                figure = fig_bar_permill_c,
                                config =
                                {
                                    'displayModeBar': False,
                                },
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                md = 3,
            ),#Col3

            #=================================================================#
            #=== Col4
            dbc.Col(
                [
                    html.Div(
                        [
                            dbc.Label('Deaths per million people:'),
                            #=================================================#
                            #=== Bar plot - deaths per mill
                            dcc.Graph(
                                id = 'Bar_permill_d',
                                figure = fig_bar_permill_d,
                                config =
                                {
                                    'displayModeBar': False
                                },
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                md = 3,
            ),#Col4
        ],
        align = 'center',
    ),#Row4

#=============================================================================#
#                                   Spain                                     #
#=============================================================================#

    #=========================================================================#
    #=== Row 5
    dbc.Row(
        [
            html.Div(
                [
                    #=========================================================#
                    #=== Spain Title
                    html.Hr(),
                    html.H2(
                        'Spain',
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '25',
                            'margin-bottom': '20',
                        },
                    ),#H2
                    #=========================================================#
                    #== Select Regions - Multi-Select Dropdown
                    html.P('Select Region:'),
                    dcc.RadioItems(
                        id = 'region_set',
                        options =
                        [
                            {
                                'label': ' MD,CT',
                                'value': 'MD,CT',
                            },
                            {
                                'label': ' All',
                                'value': 'All',
                            },
                        ],
                        value = 'All',
                        labelStyle =
                        {
                            'display': 'inline-block',
                            'margin': '5px',
                        },
                    ),#RadioItems
                    dcc.Dropdown(
                        id = 'Selected_Regions',
                        multi = True,
                        value =
                        [
                            'Madrid',
                            'Catalonia',
                        ],
                    ),#Dropdown
                ],#Div
                style =
                {
                    'font-family': 'Helvetica',
                    'margin-top': '10',
                    'font-size': '100%',
                    'width': '100%',
                },
            ),#Div
        ],
        align = 'center',
    ),#Row5

    #=========================================================================#
    #== Row 6
    dbc.Row(
        [
            #=================================================================#
            #== Col1
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.FormGroup(
                                [
                                    #=========================================#
                                    #=== Metric Selector
                                    html.Div(
                                        [
                                            dbc.Label('Metric:'),
                                            dcc.RadioItems(
                                                id = 'yaxis_scale_s',
                                                options =
                                                [
                                                    {
                                                        'label': ' Linear',
                                                        'value': 'lin',
                                                    },
                                                    {
                                                        'label': ' Logarithmic',
                                                        'value': 'log',
                                                    },
                                                ],
                                                value = 'log',
                                                labelStyle =
                                                {
                                                    'display': 'inline-block',
                                                    'margin': '5px',
                                                },
                                            ),#RadioItems
                                        ],
                                        style =
                                        {
                                            'font-family': 'Helvetica',
                                            'margin-top': '10',
                                            'font-size': '100%',
                                            'width': 150,
                                        },
                                    ),#Div
                                ],
                            ),#FormGroup
                            dbc.FormGroup(
                                [
                                    #=========================================#
                                    #=== Data Selector
                                    html.Div(
                                        [
                                            dbc.Label('Cases:'),
                                            dcc.RadioItems(
                                                id = 'Data_to_show_s',
                                                options =
                                                [
                                                    {
                                                        'label': ' Confirmed',
                                                        'value': 'cases',
                                                    },
                                                    {
                                                        'label': ' Deaths',
                                                        'value': 'deaths',
                                                    },
                                                    {
                                                        'label':' Daily Deaths',
                                                        'value': 'daily_deaths',
                                                    },
                                                ],
                                                value = 'cases',
                                                labelStyle =
                                                {
                                                    'display': 'inline-block',
                                                    'margin': '5px',
                                                },
                                            ),#RadioItems
                                        ],
                                        style =
                                        {
                                            'font-family': 'Helvetica',
                                            'margin-top': '10',
                                            'font-size': '100%',
                                            'width': 150,
                                        },
                                    )#Div
                                ],
                            ),#FormGroup
                        ],
                        body = True,
                    ),#Card
                ],
                md = 3,
            ),#Col1
            #=================================================================#
            #=== Col2
            dbc.Col(
                [
                    html.Div(
                        [
                            #=================================================#
                            #=== Line Graph
                            dcc.Graph(id = 'Line_Graph_s'),
                        ],
                        style =
                        {
                            'width': '80%',
                        },
                    ),#Div
                ],
                md = 8,
            ),#Col2
        ],
        align = 'center',
    ),#Row6
    #=========================================================================#
    #=== Row 7
    dbc.Row(
        [
            #=================================================================#
            #=== Col1
            dbc.Col(
                [
                    html.Div(
                        [
                            #=================================================#
                            #=== Sunburst Plot
                            dbc.Label('Click on regions for more info:'),
                            dcc.Graph(
                                id = 'Sunburst',
                                figure = fig_sunburst
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top':'10',
                            'font-size': '100%',
                        },
                    ),#Div
                ],
                width = 5,
            ),#Col1
            #=================================================================#
            #=== Col2
            dbc.Col(
                [
                    html.Div(
                        [
                            #=================================================#
                            #=== Map Graph
                            dbc.Label('Confirmed Cases per Region:'),
                            dcc.Graph(
                                id = 'Spain_Map',
                                figure = fig_spain_map,
                            ),#Graph
                        ],
                        style =
                        {
                            'font-family': 'Helvetica',
                            'margin-top': '10',
                            'font-size': '100%',
                            'width': '100%',
                        },
                    ),#Div
                ],
                width = 5,
            ),#Col2
        ],
        justify = 'between',
    ),#Row7
    #=========================================================================#
    #=== Row 8
    dbc.Row(
        [
            html.Div(
                [
                    #=========================================================#
                    #=== Epilogue
                    html.P(
                        'Source: European Centre for Disease Prevention ' +
                        'and Control & Instituto de Salud Carlos III'
                    ),
                    html.P(
                        'data updated daily',
                        style =
                        {
                            'font-size': '80%',
                        },
                    ),
                ],
                style =
                {
                    'font-family': 'Helvetica',
                    'font-size': '80%',
                    'margin-top': '10',
                },
            ),#Div
        ],
        align = 'center',
    ),#Row8
])#Container

#=============================================================================#
#                                                                             #
#                           Callbacks & Functions                             #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Map Selector ==========
@app.callback(Output('Maps', 'figure'),
             [Input('Map_cases', 'value')])
def callback_Map_cases(Map_cases_value):

    if Map_cases_value == 'cases':
        fig = px.scatter_geo(
            df_geo, locations = 'iso_alpha',
            color = 'continent',
            hover_name = 'country',
            size = 'cases',
            animation_frame = 'day',
            size_max = 100,
            projection = 'natural earth',
        )

    elif Map_cases_value == 'deaths':
        fig = px.scatter_geo(
            df_geo,
            locations = 'iso_alpha',
            color = 'continent',
            hover_name = 'country',
            size = 'deaths',
            projection = 'natural earth',
        )

    fig.update_layout(
        width = 800,
        margin = dict(t = 0, l = 0, r = 0, b = 0),
    )

    return fig

#=============================================================================#
# ========== Region Selector ==========
@app.callback(Output('Selected_Regions', 'options'),
             [Input('Data_to_show_s', 'value')])
def callback_Data_to_show_s(Data_to_show_s_value):

    if Data_to_show_s_value == 'cases':
        list = [{'label':i,'value':i} for i in df_traj_spain['cases'].columns]
        return list
    elif Data_to_show_s_value == 'deaths':
        list = [{'label':i,'value':i} for i in df_traj_spain['deaths'].columns]
        return list
    elif Data_to_show_s_value == 'daily_deaths':
        list = [{'label':i,'value':i} for i in \
                df_traj_spain['daily_deaths'].columns]
        return list

@app.callback(Output('Selected_Regions', 'value'),
             [Input('region_set', 'value')])
def callback_region_set(region_set_value):

    if region_set_value == 'All':
        list = df_traj_spain['cases'].columns
        return list
    elif region_set_value == 'MD,CT':
        list = ['Madrid','Catalonia']
        return list
    return

#=============================================================================#
# ========== Country Selector ==========
@app.callback(Output('Selected_Countries', 'options'),
             [Input('Data_to_show', 'value')])
def callback_Data_to_show(Data_to_show_value):

    if Data_to_show_value == 'cases':
        list = [{'label':i,'value':i}for i in df_traj['cases'].columns]
        return list
    elif Data_to_show_value == 'deaths':
        list = [{'label':i,'value':i}for i in df_traj['deaths'].columns]
        return list
    elif Data_to_show_value == 'daily_deaths':
        list = [{'label':i,'value':i}for i in df_traj['daily_deaths'].columns]
        return list

@app.callback(Output('Selected_Countries', 'value'),
             [Input('country_set', 'value')])
def callback_country_set(country_set_value):

    if country_set_value == 'All':
        list = df_traj['cases'].columns
        return list
    elif country_set_value == '!':
        list = [
            'Spain', 'Italy', 'UK', 'China', 'South_Korea',
            'Japan', 'Taiwan', 'Germany', 'USA',
        ]
        return list
    elif country_set_value == 'Africa':
        list = [
            'Algeria', 'Burkina_Faso', 'Cameroon', 'Canada', 'Cote_dIvoire',
            'Democratic_Republic_of_the_Congo', 'Egypt', 'Ghana', 'Kenya',
            'Mauritius', 'Morocco', 'Nigeria', 'Senegal', 'South_Africa',
            'Tunisia',
        ]
        return list
    elif country_set_value == 'Americas':
        list = [
            'Argentina', 'Bolivia', 'Brazil', 'Canada', 'Chile',
            'Colombia', 'Costa_Rica', 'Cuba', 'Dominican_Republic',
            'Ecuador', 'Honduras', 'Mexico', 'Panama', 'Peru',
            'Puerto_Rico', 'Trinidad_and_Tobago', 'USA',
            'Uruguay',  'Venezuela',
        ]
        return list
    elif country_set_value == 'Asia':
        list = [
            'Afghanistan', 'Algeria', 'Armenia', 'Azerbaijan', 'Bahrain',
            'Brunei_Darussalam', 'Cambodia',
            'Cases_on_an_international_conveyance_Japan', 'China', 'India',
            'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan',
            'Kazakhstan', 'Kuwait', 'Kyrgyzstan',  'Lebanon', 'Malaysia',
            'Oman', 'Pakistan', 'Palestine', 'Philippines',  'Qatar', 'Russia',
            'Saudi_Arabia', 'Singapore', 'South_Korea', 'Sri_Lanka', 'Taiwan',
            'Thailand', 'Tunisia', 'Turkey', 'United_Arab_Emirates',
            'Uzbekistan', 'Vietnam',
        ]
        return list
    elif country_set_value == 'Europe':
        list = [
            'Albania',  'Andorra', 'Austria', 'Belarus', 'Belgium',
            'Bosnia_and_Herzegovina', 'Bulgaria',  'Croatia', 'Cyprus',
            'Czech_Republic', 'Denmark', 'Estonia', 'Finland', 'France',
            'Faroe_Islands','Germany', 'Greece', 'Guernsey', 'Hungary',
            'Iceland', 'Ireland', 'Isle_of_Man', 'Italy', 'Jersey', 'Kosovo',
            'Latvia', 'Lithuania', 'Luxembourg',  'Malta', 'Moldova',
            'Montenegro',  'Netherlands', 'North_Macedonia', 'Norway', 'Poland',
            'Portugal',  'Romania', 'Russia', 'San_Marino', 'Serbia',
            'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
            'UK',
        ]
        return list
    elif country_set_value == 'Oceania':
        list = [
            'Australia','New_Zealand',
        ]
        return list
    return

#=============================================================================#
# ========== Line Graph ==========

# Reference line color
color_balck = 'rgb(67,67,67)'

@app.callback(Output('Line_Graph', 'figure'),
              [Input('Selected_Countries', 'value'),
              Input('yaxis_scale', 'value'),
              Input('Data_to_show', 'value')])

def callback_Line_Graph(Selected_Countries_value,
                        yaxis_scale_value,
                        Data_to_show_value):

    #=========================================================================#
    #=== Confirmed
    if Data_to_show_value == 'cases':
        Columns = list(df_traj['cases'].columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_traj['cases'][Countries]
        lines = df_traj['lines_cases']
        title = 'Infection Trajectories, Growth of Outbreak by Country'
        x_title = 'Number of days since 100th case'
        y_title = 'Comulative number of cases'
        rx = []
        for column in data.columns:
            rx.append(np.count_nonzero(~np.isnan(data[column])))
        if not rx:
            rx = 0
        else:
            rx = max(rx) + 10
        range_x = [0,rx]
        range_y = [90,data.max().max() * 1.05]
        range_logy = [np.log10(90),np.log10(data.max().max() * 1.05)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                        'color': COLOURS[i],
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                        'color': COLOURS[i],
                        'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left',
                    yanchor = 'middle',
                    text = column,
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False
                )
            )

        for column in lines.columns:
            plot_data.append(
                go.Scatter(
                    x = list(lines.index.values),
                    y = lines[column],
                    hovertemplate = 'Doubles',
                    mode = 'lines',
                    opacity = 0.5,
                    line = {
                        'color': color_balck,
                        'dash':'dash',
                    },
                    name = column,
                )
            )

    #=========================================================================#
    #=== Deaths
    elif Data_to_show_value == 'deaths':
        Columns = list(df_traj['deaths'].columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_traj['deaths'][Countries]
        lines = df_traj['lines_deaths']
        title = 'Infection Trajectories, Growth of Outbreak by Country'
        x_title = 'Number of days since 10th death'
        y_title = 'Comulative number of deaths'
        rx = []
        for column in data.columns:
            rx.append(np.count_nonzero(~np.isnan(data[column])))
        if not rx:
            rx = 0
        else:
            rx = max(rx) + 10
        range_x = [0,rx]
        range_y = [9,data.max().max() * 1.05]
        range_logy = [np.log10(9),np.log10(data.max().max() * 1.05)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                       'color': COLOURS[i],
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                        'color': COLOURS[i],
                        'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left',
                    yanchor = 'middle',
                    text = column,
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False
                )
            )

        for column in lines.columns:
            plot_data.append(
                go.Scatter(
                    x = list(lines.index.values),
                    y = lines[column],
                    hovertemplate = 'Doubles',
                    mode = 'lines',
                    opacity = 0.5,
                    line = {
                       'color': color_balck,
                       'dash':'dash',
                    },
                    name = column,
                )
            )

    #=========================================================================#
    #=== Daily Deaths
    elif Data_to_show_value == 'daily_deaths':
        Columns = list(df_traj['daily_deaths'].columns)
        Countries = [c for c in Selected_Countries_value if c in Columns]

        data = df_traj['daily_deaths'][Countries]
        title = 'Daily deaths'
        x_title = 'Number of days since 3 daily deaths first recorded'
        y_title = 'Number of daily deaths (7 day rolling average)'
        rx = []
        for column in data.columns:
            rx.append(np.count_nonzero(~np.isnan(data[column])))
        if not rx:
            rx = 0
        else:
            rx = max(rx) + 10
        range_x = [0,rx]
        range_y = [1,data.max().max() * 1.05]
        range_logy = [np.log10(1),np.log10(data.max().max() * 1.05)]

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                       'color': COLOURS[i]
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                        'color': COLOURS[i],
                        'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left',
                    yanchor = 'middle',
                    text = column,
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False,
                )
            )

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
            xaxis = {
                'title': x_title,
                'range': range_x,
            },
            yaxis = {
                'title': y_title,
                'range': range_logy if yaxis_scale_value == 'log' else range_y,
                'type': 'log' if yaxis_scale_value == 'log' else 'linear'
            },
            hovermode = 'x',
            annotations = annotations
        ),
    }

#=============================================================================#
#                              Spain Graph                                    #
#=============================================================================#

@app.callback(Output('Line_Graph_s', 'figure'),
              [Input('Selected_Regions', 'value'),
              Input('yaxis_scale_s', 'value'),
              Input('Data_to_show_s', 'value')])

def callback_Line_Graph(Selected_Regions_value,
                        yaxis_scale_s_value,
                        Data_to_show_s_value):

    #=========================================================================#
    #=== Confirmed
    if Data_to_show_s_value == 'cases':

        Columns = list(df_traj_spain['cases'].columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = df_traj_spain['cases'][Regions]
        lines = df_traj_spain['lines_cases']
        title = 'Infection Trajectories, Growth of Outbreak by Region'
        x_title = 'Number of days since 10th case'
        y_title = 'Comulative number of cases'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                        'color': COLOURS_SPAIN[i],
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                       'color': COLOURS_SPAIN[i],
                       'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left', yanchor = 'middle',
                    text = df_geo_spain[['region','region_name']]\
                           .drop_duplicates().set_index('region_name')\
                           .region.loc[column],
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False))
        for column in lines.columns:
            plot_data.append(
                go.Scatter(
                    x = list(lines.index.values),
                    y = lines[column],
                    hovertemplate = 'Doubles',
                    mode = 'lines',
                    opacity = 0.5,
                    line = {
                       'color': color_balck,
                       'dash':'dash',
                    },
                    name = column
                )
            )

    #=========================================================================#
    #=== Deaths
    elif Data_to_show_s_value == 'deaths':

        Columns = list(df_traj_spain['deaths'].columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = df_traj_spain['deaths'][Regions]
        lines = df_traj_spain['lines_deaths']
        title = 'Infection Trajectories, Growth of Outbreak by Region'
        x_title = 'Number of days since 3rd death'
        y_title = 'Comulative number of deaths'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                        'color': COLOURS_SPAIN[i],
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                        'color': COLOURS_SPAIN[i],
                        'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left',
                    yanchor = 'middle',
                    text = df_geo_spain[['region','region_name']]\
                           .drop_duplicates().set_index('region_name')\
                           .region.loc[column],
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False,
                )
            )
        for column in lines.columns:
            plot_data.append(
                go.Scatter(
                    x = list(lines.index.values),
                    y = lines[column],
                    hovertemplate = 'Doubles',
                    mode = 'lines',
                    opacity = 0.5,
                    line = {
                        'color': color_balck,
                        'dash':'dash',
                    },
                    name = column,
                )
            )

    #=========================================================================#
    #=== Daily Deaths
    elif Data_to_show_s_value == 'daily_deaths':

        Columns = list(df_traj_spain['daily_deaths'].columns)
        Regions = [c for c in Selected_Regions_value if c in Columns]

        data = df_traj_spain['daily_deaths'][Regions]
        title = 'Daily deaths'
        x_title = 'Number of days since 3 daily deaths first recorded'
        y_title = 'Number of daily deaths (7 day rolling average)'

        fig = go.Figure()

        plot_data = []
        annotations = []
        for i, column in enumerate(data.columns):
            plot_data.append(
                go.Scatter(
                    x = list(data.index.values),
                    y = data[column],
                    hovertemplate = '%{y:.0f}',
                    mode = 'lines',
                    line = {
                        'color': COLOURS_SPAIN[i],
                    },
                    opacity = 0.6,
                    name = column,
                )
            )

            plot_data.append(
                go.Scatter(
                    x = [data[column].dropna().index[-1]],
                    y = [list(data[column].dropna())[-1]],
                    mode = 'markers',
                    marker = {
                        'color': COLOURS_SPAIN[i],
                        'size' : 5,
                    },
                    showlegend = False,
                    opacity = 0.6,
                    hoverinfo = 'skip',
                )
            )

            annotations.append(
                dict(
                    xref = 'x',
                    yref = 'y',
                    x = data[column].dropna().index[-1],
                    y = list(data[column].dropna())[-1],
                    xanchor = 'left',
                    yanchor = 'middle',
                    text = df_geo_spain[['region','region_name']]\
                           .drop_duplicates().set_index('region_name')\
                           .region.loc[column],
                    font = {
                        'family':'Arial',
                        'size':12,
                    },
                    showarrow = False,
                )
            )

    for i in range(0,plot_data.__len__()):
        fig.add_trace(plot_data[i])

    if yaxis_scale_s_value == 'log':
        for i in annotations:
            i['y'] = np.log10(i['y'])
    elif yaxis_scale_s_value == 'linear':
        annotations = annotations

    fig.update_layout(
        title = title,
        width = 900,
        height = 500,
        xaxis = {'title': x_title,},
        yaxis = {
            'title': y_title,
            'type': 'log' if yaxis_scale_s_value == 'log' else 'linear',
        },
        hovermode = 'x',
        annotations = annotations,
        template = 'plotly_white',
    )
    return fig

#=============================================================================#
#=============================================================================#

if __name__ == '__main__':
    #app.run_server()
    app.run_server(debug = True)
