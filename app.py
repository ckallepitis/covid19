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
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.express as px

from elements import COLOURS, COLOURS_SPAIN, SPAIN_GEOLOCATIONS
from data import get_covid_data, get_line_graph_data
from data import get_bar_plot_data, get_geo_data
from data import get_covid_data_Spain, get_geo_Spain_data, get_sunburst_data

#from rq import Queue
#from worker import conn

#q = Queue(connection=conn)

# Launch the application:
app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])
app.config.suppress_callback_exceptions = True
server = app.server

# World Data
#df = q.enqueue(get_covid_data,
# 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
df = get_covid_data()
df_line_data = get_line_graph_data(df)
df_geo = get_geo_data(df)
df_bar = get_bar_plot_data(df)

# Spain Data
df_spain = get_covid_data_Spain()
df_spain_line_data = df_spain[['Date', 'Cases',
                               'Daily Cases; 7-day rolling average', 'Deaths',
                               'Daily Deaths; 7-day rolling average', 'Region',
                               'Region_Name']]
df_geo_spain = get_geo_Spain_data(df_spain)
df_sunburst = get_sunburst_data(df_spain)


#=============================================================================#
#                                                                             #
#                               Plots - World                                 #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Bar-plot ==========

fig_bar = px.bar(
    df_bar,
    x = 'values',
    y = 'country',
    text = 'values',
    color = 'country',
    facet_col ='cat',
    facet_col_spacing = 0.1,
    hover_name = 'country',
    hover_data = {'cat':False,
                  'country':False,
                  'values':False,
                 },
)

fig_bar.update_yaxes(matches=None)
fig_bar.update_xaxes(matches=None)
fig_bar.update_yaxes(showticklabels=True)
fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
fig_bar.update_yaxes(categoryorder='total ascending')
fig_bar.update_xaxes(title = '')

fig_bar.update_layout(
    showlegend = False,
    yaxis = {'title': ''},
    template = 'plotly_white',
)

#=============================================================================#
#                                                                             #
#                               Plots - Spain                                 #
#                                                                             #
#=============================================================================#

#=============================================================================#
# ========== Sunburst plot ==========

fig_sunburst = px.sunburst(df_sunburst,
                           path=['Country','Region_Name','Cases','variable'],
                           values='value')

#=============================================================================#
#                                                                             #
#                                 Layout                                      #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                                   Sidebar                                   #
#=============================================================================#

sidebar = html.Div(
    [
        html.Div('Covid-19 Dashboard',
        style={'color':'#2fa4e7', 'fontSize':'4vw'}),
        html.Hr(),
        html.P(
            'Data following the Covid-19 pandemic', className='lead'
        ),
        dbc.Nav(
            [
                dbc.NavLink('World', href='/page-1', id='page-1-link'),
                dbc.NavLink('Spain', href='/page-2', id='page-2-link'),
                dbc.NavLink('Cyprus', href='/page-3', id='page-3-link'),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style= {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '20%',
        'padding': '2rem 1rem',
        'background-color': '#f8f9fa',
    },
)

content = html.Div(
    id='page-content',
    style={
        'margin-left': '25%',
    }
)

#=============================================================================#
#                                   World                                     #
#=============================================================================#

world = dbc.Container([
    #=========================================================================#
    #=== Row 1
    dbc.Row(html.H1('World')),
    dbc.Row([
        html.Div([
            dbc.Label('Select Countries:'),
            dcc.RadioItems(
                id = 'country_set',
                options = [
                    {'label': ' !', 'value': '!'},
                    {'label': ' World', 'value': 'World'},
                    {'label': ' Africa', 'value': 'Africa'},
                    {'label': ' America', 'value': 'America'},
                    {'label': ' Asia', 'value': 'Asia'},
                    {'label': ' Europe', 'value': 'Europe'},
                    {'label': ' Oceania','value': 'Oceania'},
                ],
                value = '!',
                labelStyle =
                {
                    'display': 'inline-block',
                    'margin': '5px',
                }
            ),#RadioItems
            dcc.Dropdown(
                id = 'Selected_Countries',
                multi = True,
            ),
        ],
            style = {
                'width': '100%',
            }
        ),#Div
    ]),#Row 1

    #=========================================================================#
    #=== Row 2
    dbc.Row([
        #=====================================================================#
        #=== Col1
        dbc.Col([
            dbc.Card([
                dbc.FormGroup([
                    #=========================================================#
                    #=== Metric Selector
                    dbc.Label('Metric:'),
                    dbc.RadioItems(
                        id = 'yaxis_scale',
                        options =[
                            {'label': 'Linear','value': False},
                            {'label': 'Logarithmic','value': True},
                        ],
                        value = False,
                    ),#RadioItems
                ]),#FormGroup
                dbc.FormGroup([
                    #=========================================================#
                    #=== Data Selector
                    dbc.RadioItems(
                        id = 'Data_to_show',
                        options =[
                            {'label': 'Confirmed','value': 'cases'},
                            {'label': 'Daily Confirmed',
                             'value': 'daily cases; 7-day rolling average'},
                            {'label': 'Deaths',
                             'value': 'deaths'},
                            {'label': 'Daily Deaths',
                            'value': 'daily deaths; 7-day rolling average'},
                        ],
                        value = 'daily cases; 7-day rolling average',
                    ),#RadioItems
                ]),#FormGroup
            ],
                style = {
                    'width': '11rem',
                },
                body = True
            ),#Card
        ],
            md = 2,
        ),#Col1

        #=====================================================================#
        #=== Col2
        dbc.Col([
            dcc.Graph(id = 'Line_Graph')
        ],
            md = 10,
        ),#Col2
    ],
        align = 'center',
    ),#Row2

    #=========================================================================#
    #=== Row 3
    dbc.Row([
        #=====================================================================#
        #=== Col1
        dbc.Col([
            dbc.Card([
                html.Div([
                    dbc.Row([
                        daq.BooleanSwitch(id='boolean_switch',on=False),
                        dbc.Label('Animate Map')
                    ]),
                ],
                    style = {
                        'padding': '0.5rem 0.5rem',
                    }
                ),
                dbc.FormGroup([
                    #=========================================================#
                    #=== Map Data Selector
                    dbc.RadioItems(
                        id = 'Map_Data_to_show',
                        options =[
                            {'label': 'Confirmed','value': 'cases'},
                            {'label': 'Daily Confirmed','value': 'daily_cases'},
                            {'label': 'Deaths','value': 'deaths'},
                            {'label': 'Daily Deaths','value': 'daily_deaths'},
                        ],
                        value = 'cases',
                    ),#RadioItems
                ],
                ),#FormGroup
            ],
                style = {
                    'width': '11rem',
                },
                body = True,
            ),#Card
        ],
            md = 2,
        ),#Col1

        #=====================================================================#
        #=== Col2
        dbc.Col([
            dcc.Graph(id = 'Map')
        ],
            md = 10,
        ),#Col2
    ],
        align = 'center',
    ),#Row3

    #=========================================================================#
    #=== Row 4
    dbc.Row([
        #=====================================================================#
        #=== Bar plot
        dbc.Col([
            dcc.Graph(
                id = 'Bar_plot',
                figure = fig_bar,
                config = {
                    'displayModeBar': False,
                },
            ),
        ],
            align = 'center',
            md = 12,
        ),#Col
    ]
    ),#Row4

    #=========================================================================#
    #=== Row 5
    dbc.Row([
        dbc.Label('Source: European Centre for Disease Prevention ;'),
        html.A(' Data',
            href='https://github.com/datadista/datasets/raw/master/COVID%2019/',
            target='_blank'),
    ]),
])#Container world

#=============================================================================#
#                                   Spain                                     #
#=============================================================================#

spain = dbc.Container([
    #=========================================================================#
    #=== Row 1
    dbc.Row(html.H1('Spain')),
    dbc.Row([
        html.Div([
            dbc.Label('Select Regions:'),
            dcc.RadioItems(
                id = 'region_set',
                options = [
                    {'label': ' Spain', 'value': 'Spain'},
                    {'label': ' Madrid', 'value': 'Madrid'},
                    {'label': ' Catalonia', 'value': 'Catalonia'},
                ],
                value = 'Catalonia',
                labelStyle =
                {
                    'display': 'inline-block',
                    'margin': '5px',
                }
            ),#RadioItems
            dcc.Dropdown(
                id = 'Selected_Regions',
                multi = True,
            ),
        ],
            style = {
                'width': '100%',
            }
        ),#Div
    ]),#Row 1

    #=========================================================================#
    #=== Row 2
    dbc.Row([
        #=====================================================================#
        #=== Col1
        dbc.Col([
            dbc.Card([
                dbc.FormGroup([
                    #=========================================================#
                    #=== Metric Selector
                    dbc.Label('Metric:'),
                    dbc.RadioItems(
                        id = 'yaxis_scale_s',
                        options =[
                            {'label': 'Linear','value': False},
                            {'label': 'Logarithmic','value': True},
                        ],
                        value = False,
                    ),#RadioItems
                ]),#FormGroup
                dbc.FormGroup([
                    #=========================================================#
                    #=== Data Selector
                    dbc.RadioItems(
                        id = 'Spain_Data_to_show',
                        options =[
                            {'label': 'Confirmed','value': 'Cases'},
                            {'label': 'Daily Confirmed',
                             'value': 'Daily Cases; 7-day rolling average'},
                            {'label': 'Deaths','value': 'Deaths'},
                            {'label': 'Daily Deaths',
                             'value': 'Daily Deaths; 7-day rolling average'},
                        ],
                        value = 'Daily Cases; 7-day rolling average',
                    ),#RadioItems
                ]),#FormGroup
            ],
                style = {
                    'width': '11rem',
                },
                body = True
            ),#Card
        ],
            md = 2,
        ),#Col1

        #=====================================================================#
        #=== Col2
        dbc.Col([
            dcc.Graph(id = 'Spain_Line_Graph')
        ],
            md = 10,
        ),#Col2
    ],
        align = 'center',
    ),#Row2

    #=========================================================================#
    #=== Row 3
    dbc.Row([
        #=====================================================================#
        #=== Col1
        dbc.Col([
            dbc.Card([
                html.Div([
                    dbc.Row([
                        daq.BooleanSwitch(id='boolean_switch_spain',on=False),
                        dbc.Label('Animate Map')
                    ]),
                ],
                    style = {
                        'padding': '0.5rem 0.5rem',
                    }
                ),
                dbc.FormGroup([
                    #=========================================================#
                    #=== Map Data Selector
                    dbc.RadioItems(
                        id = 'Spain_Map_Data_to_show',
                        options =[
                            {'label': 'Confirmed','value': 'Cases'},
                            {'label': 'Daily Confirmed',
                            'value': 'Daily Cases'},
                            {'label': 'Deaths','value': 'Deaths'},
                            {'label': 'Daily Deaths',
                            'value': 'Daily Deaths'},
                        ],
                        value = 'Cases',
                    ),#RadioItems
                ],
                ),#FormGroup
            ],
                style = {
                    'width': '11rem',
                },
                body = True,
            ),#Card
        ],
            md = 2,
        ),#Col1

        #=====================================================================#
        #=== Col2
        dbc.Col([
            dcc.Graph(id = 'Spain_Map')
        ],
            md = 10,
        ),#Col2
    ],
        align = 'center',
    ),#Row3

    #=========================================================================#
    #=== Row 4
    dbc.Row([
        #=====================================================================#
        #=== Sunburst Plot
        dbc.Label('Click on regions for more info:'),
        dcc.Graph(
            id = 'Sunburst',
            figure = fig_sunburst
        ),#Graph
    ],
        align = 'center',
    ),#Row4
    #=========================================================================#
    #=== Row 5
    dbc.Row([
        dbc.Label('Source: Datadista.com ;'),
        html.A(' Data',
               href='https://github.com/datadista/datasets/raw/master/COVID%2019/',
               target='_blank'),
    ]),
])#Container Spain


cyprus = dbc.Container([
    dbc.Row([
        html.A('University of Cyprus Covid-19 Dashboard',
               href='https://covid19.ucy.ac.cy/',
               target='_blank')
    ],
        justify='center',
        align='center',
        className='h-50',
    )
],
    style={'height': '100vh'}
)


app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    content,
])

#=============================================================================#
#                                                                             #
#                           Callbacks & Functions                             #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                                 Sidebar                                     #
#=============================================================================#
# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f'page-{i}-link', 'active') for i in range(1, 4)],
    [Input('url', 'pathname')],
)
def toggle_active_links(pathname):
    if pathname == '/':
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f'/page-{i}' for i in range(1, 4)]


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname in ['/', '/page-1']:

        return world
    elif pathname == '/page-2':

        return spain
    elif pathname == '/page-3':

        return cyprus
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1('404: Not found', className='text-danger'),
            html.Hr(),
            html.P(f'The pathname {pathname} was not recognised...'),
        ]
    )
#=============================================================================#
#                                   World                                     #
#=============================================================================#

#=============================================================================#
# ========== Dropdown Country list ==========

@app.callback(Output('Selected_Countries', 'options'),
             [Input('Data_to_show', 'value')])
def callback_country_list(Data_to_show_value):
    return [{'label':i,'value':i} for i in df.country.unique()]

#=============================================================================#
# ========== Country Selector ==========

@app.callback(Output('Selected_Countries', 'value'),
             [Input('country_set', 'value')])
def callback_country_set(country_set_value):

    if country_set_value == '!':
        country_list = df.query("country == ['Cyprus','France','Guatemala','Italy',"+\
             "'Netherlands','Spain','Sweden','UK']").country.unique().tolist()
        return country_list
    if country_set_value == 'World':
        country_list = df.query("popData2019 > 0").country.unique().tolist()
        return country_list
    if country_set_value == 'Africa':
        country_list = df.query("continentExp == 'Africa'").country.unique().tolist()
        return country_list
    if country_set_value == 'America':
        country_list = df.query("continentExp == 'America'").country.unique().tolist()
        return country_list
    if country_set_value == 'Asia':
        country_list = df.query("continentExp == 'Asia'").country.unique().tolist()
        return country_list
    if country_set_value == 'Europe':
        country_list = df.query("continentExp == 'Europe'").country.unique().tolist()
        return country_list
    if country_set_value == 'Oceania':
        country_list = df.query("continentExp == 'Oceania'").country.unique().tolist()
        return country_list

    return country_list

#=============================================================================#
# ========== Line Graph ==========

@app.callback(Output('Line_Graph', 'figure'),
              [Input('Selected_Countries', 'value'),
              Input('yaxis_scale', 'value'),
              Input('Data_to_show', 'value')])

def callback_Line_Graph(Selected_Countries_value,
                        yaxis_scale_value,
                        Data_to_show_value):

    fig = px.line(
        df_line_data.query("country == "+str(Selected_Countries_value)),
        x = 'day',
        y = Data_to_show_value,
        color = 'continent' if Selected_Countries_value == "popData2019 > 0" else 'country',
        log_y = yaxis_scale_value
    )

    fig.update_traces(hovertemplate=None)

    fig.update_layout(
        template = 'plotly_white',
        hovermode='x',
    )

    return fig

#=============================================================================#
# ========== Map Selector ==========

@app.callback(Output('Map', 'figure'),
             [Input('Map_Data_to_show', 'value'),
             Input('boolean_switch', 'on')])
def callback_Map(Map_Data_to_show_value,boolean_switch_on):

    df_geo_o = df_geo

    if not boolean_switch_on:
        df_geo_o = df_geo.loc[df_geo.groupby('country').cases.idxmax()]

    fig = px.scatter_geo(
        df_geo_o,
        locations = 'iso_alpha',
        color = 'continent',
        hover_name = 'country',
        hover_data = {
            'day':True,
            'iso_alpha':False,
            'continent':False,
        },
        size = Map_Data_to_show_value,
        animation_frame = 'day' if boolean_switch_on else None,
        projection = 'natural earth',
    )

    fig.update_layout(
        margin = dict(t = 0, l = 0, r = 0, b = 0),
        template = 'plotly_white',
    )

    fig.update_geos(fitbounds='locations',showcountries = True)

    if boolean_switch_on:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10

    return fig

#=============================================================================#
#                                   Spain                                     #
#=============================================================================#

#=============================================================================#
# ========== Dropdown Region list ==========

@app.callback(Output('Selected_Regions', 'options'),
             [Input('Spain_Data_to_show', 'value')])
def callback_region_list(Spain_Data_to_show):
    return [{'label':i,'value':i} for i in df_spain.Region_Name.unique()]

#=============================================================================#
# ========== Region Selector ==========

@app.callback(Output('Selected_Regions', 'value'),
             [Input('region_set', 'value')])
def callback_region_set(region_set_value):

    if region_set_value == 'Spain':
        region_list = df_spain.query("Cases > -1").Region_Name.unique().tolist()
        return region_list
    if region_set_value == 'Madrid':
        region_list = df_spain.query("Region_Name == 'Madrid'").Region_Name.unique().tolist()
        return region_list
    if region_set_value == 'Catalonia':
        region_list = df_spain.query("Region_Name == 'Catalonia'").Region_Name.unique().tolist()
        return region_list


    return region_list

#=============================================================================#
# ========== Spain Line Graph ==========

@app.callback(Output('Spain_Line_Graph', 'figure'),
              [Input('Selected_Regions', 'value'),
              Input('yaxis_scale_s', 'value'),
              Input('Spain_Data_to_show', 'value')])

def callback_Spain_Line_Graph(Selected_Regions_value,
                              yaxis_scale_s_value,
                              Spain_Data_to_show_value):

    fig = px.line(
        df_spain_line_data.query("Region_Name == "+str(Selected_Regions_value)),
        x = 'Date',
        y = Spain_Data_to_show_value,
        color = 'Region_Name',
        log_y = yaxis_scale_s_value
    )

    fig.update_traces(hovertemplate=None)

    fig.update_layout(
        template = 'plotly_white',
        hovermode='x',
    )

    return fig

#=============================================================================#
# ========== Spain Map Selector ==========

@app.callback(Output('Spain_Map', 'figure'),
             [Input('Spain_Map_Data_to_show', 'value'),
             Input('boolean_switch_spain', 'on')])
def callback_Spain_Map(Spain_Map_Data_to_show_value,
                      boolean_switch_spain_on):

    df_geo_spain_o = df_geo_spain

    if not boolean_switch_spain_on:
        df_geo_spain_o = df_geo_spain.loc[df_geo_spain.groupby('Region_Name')\
                                                    .Cases.idxmax()]

    fig = px.scatter_geo(
        df_geo_spain_o,
        lat = 'Lat',
        lon = 'Long',
        color = 'Region_Name',
        hover_name = 'Region_Name',
        hover_data = {
            'Region_Name':False,
            'Date':True,
            'Lat':False,
            'Long':False,
        },
        size = Spain_Map_Data_to_show_value,
        animation_frame = 'Date' if boolean_switch_spain_on else None,
        projection = 'natural earth',
    )

    fig.update_layout(
        margin = dict(t = 0, l = 0, r = 0, b = 0),
        template = 'plotly_white',
    )

    fig.update_geos(fitbounds='locations',showcountries = True)

    if boolean_switch_spain_on:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10

    return fig

#=============================================================================#
#=============================================================================#

if __name__ == '__main__':
    app.run_server()
