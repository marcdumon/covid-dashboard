# --------------------------------------------------------------------------------------------------------
# 2020/05/15
# src - covid_dashboard.py
# md
# --------------------------------------------------------------------------------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.graph_objs import Figure

from component_factory import generate_graph_componet
from data_process import load_ecdc

import numpy as np
import pandas as pd

import plotly.express as px
import chart_studio.plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.figure_factory as ff
from my_tools.plotly_discrete_colorscale import plotly_discrete_colorscale

from data_visualisation import create_fire_heatmap

# DATA
ecdc = load_ecdc()
# Remove 'Cases_on_an_international_conveyance_Japan'
ecdc = ecdc.loc[ecdc['region'] != 'Other']
# todo: Save this to file and allow saving of new countrylists
# todo: Map ecdc country names to readable version

region_location = ecdc[['region', 'location']]
europe_countries = region_location[region_location['region'] == 'Europe']['location'].unique().tolist()
asia_countries = region_location[region_location['region'] == 'Asia']['location'].unique().tolist()
all_countries = ecdc['location'].unique().tolist()
group1 = ['Belgium', 'Netherlands', 'France', 'Germany', 'Italy', 'Spain', 'United_Kingdom', 'Sweden', 'Norway', 'Singapore', 'Taiwan', 'Vietnam', 'United_States_of_America']

country_groups_options = [
    {'label': 'Europe', 'value': 'Europe'},
    {'label': 'Asia', 'value': 'Asia'},
    {'label': 'All', 'value': 'All'},
    {'label': 'Group1', 'value': 'Group1'}
]
# selected_countries = []

# HTML
# app = dash.Dash(__name__, external_stylesheets=['my_css.css', 'bootstrap.css'])
app = dash.Dash(__name__, )
# app.config.suppress_callback_exceptions = True # todo: What is this

header = html.Div(
    [dbc.Row(
        [
            dbc.Col(html.H1('COVID-19 Dashboard'), style={'text-align': 'left'}, width=3),
            dbc.Col(dbc.Button('Select Countries', id='open', outline=True, color='secondary', style={'margin': '0rem 1rem 0rem 0rem', })
                    , width=3),
            dbc.Col(html.P('y ' * 10), width=3),
            dbc.Col(html.P('z ' * 10), width=3),

            dbc.Modal(
                [
                    dbc.ModalHeader('Header'),
                    dbc.ModalBody(html.Div([
                        dcc.Checklist(id='checklist_country_groups',
                                      options=country_groups_options,
                                      # value=['Europe'],
                                      labelStyle={'display': 'inline-block'},
                                      persistence=True
                                      ),
                        dcc.Dropdown(id='dropdown_countries',
                                     options=[{'label': c, 'value': c} for c in all_countries],
                                     # value=['Belgium'],
                                     multi=True,
                                     persistence=True)])),
                    dbc.ModalFooter(dbc.Button('Close', id='close', className='ml-auto')),
                ],
                id='modal',
                size='lg',
                centered=True),
        ],
        # justify='center',
        align='center',
        # no_gutters=False,
    )
    ], style={
        'width': '100%',
        # 'margin-top': '10px',
        # 'margin-bottom': '25px',
        'margin-left': '10px',
        'top': '0px',
        'overflow': 'hidden',
        'position': 'fixed',
        'z-index': '100',
        'background-color': 'white',
        'border-bottom': '1px solid lightgray'
    },
)

graphs = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div(
                generate_graph_componet('fire_heatmap', 'Fire Heatmap'),
                # , style={'border': '1px black solid'}
                id='div_fire_heatmap'
            ), width=5),

            # dbc.Col(html.Img(id='image'), width=2),
            # dbc.Col(dcc.Graph(id='test_graph2'), width=2)
            dbc.Col(html.Div(
                generate_graph_componet('test_graph2', 'test graph2'),
                # , style={'border': '1px black solid'}
                id='div_test_graph2'
            ), width=5)
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.P('cxcczxczxczx'),
                dcc.Graph(id='test_graph4', config={'displayModeBar': False})], width=4),
            dbc.Col(dcc.Graph(id='test_graph5'), width=4),
            dbc.Col(dcc.Graph(id='test_graph6'), width=4)
        ]),
],
)

store = dcc.Store(id='tmp_countries')

app.layout = html.Div([header,
                       store,

                       dcc.Tabs([
                           dcc.Tab(graphs,
                                   label="WORLD"
                                   ),
                           dcc.Tab(None,
                                   label='BELGIUM'
                                   )
                       ])
                       ],
                      style={'margin-top': '100px', }
                      )


@app.callback(Output('dropdown_countries', 'value'),
              [Input('checklist_country_groups', 'value')],
              )
def add_country_group(country_groups):
    if country_groups is None:
        pass
        raise PreventUpdate
    countries = []
    if country_groups:
        for country_group in country_groups:
            if country_group == 'Europe': countries += europe_countries
            if country_group == 'Asia': countries += asia_countries
            if country_group == 'All': countries += all_countries
            if country_group == 'Group1': countries += group1
        countries = sorted(list(set(countries)))

        return countries


#  GRAPHS CALLBACKS
@app.callback([Output('fire_heatmap', 'figure'), Output('fire_heatmap_expanded', 'figure')],
              [Input('dropdown_countries', 'value')])
def generate_fire_heatmap(countries):
    print('Countries: ', countries)
    if countries is None:
        raise PreventUpdate

    e = ecdc[ecdc['location'].isin(countries)]
    df = e.copy()
    # Set negative cases to 0
    df['cases'][df['cases'] < 0] = 0

    # Add rolling mean
    rolling = 7
    df['rolling_cases'] = df.groupby(['location'])['cases'].rolling(rolling).mean().reset_index(0)['cases']
    df.fillna(0, inplace=True)
    # Sort x-axis
    n = 60
    sort_column = 'cases'
    from_date = max(df['date']) - pd.DateOffset(n)
    sort = df[df['date'] > from_date].groupby(['location'])[sort_column].sum().sort_values(ascending=False).index.values
    df['sort'] = pd.Categorical(df['location'], categories=sort, ordered=True)  # See https://stackoverflow.com/questions/26707171/sort-pandas-dataframe-based-on-list
    df.sort_values('sort', inplace=True)
    df.drop('sort', axis=1, inplace=True)
    # fig = create_fire_heatmap(df, column='rolling_cases', showscale=True)
    fig = create_fire_heatmap(df, column='cases', showscale=False)

    fig.update_layout(
        margin=dict(t=2, r=2, b=2, l=2),
        # margin=dict(t=2, r=2, l=2),
        # showlegend=False,
        # width=1300,
        # height=900,
        autosize=True,
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_fixedrange=True, yaxis_fixedrange=True,  # disable zoom
        xaxis_showspikes=True, yaxis_showspikes=True,
        # displayModeBar=False,
        plot_bgcolor='#66DD66')
    return fig, fig


@app.callback(
    Output('modal', 'is_open'),
    [Input('open', 'n_clicks'), Input('close', 'n_clicks')],
    [State('modal', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


for name in ['fire_heatmap', 'test_graph2']:
    @app.callback(
        Output(f'{name}_modal', 'is_open'),
        # [Input('fire_heatmap_expand_button', 'n_clicks')],
        [Input(f'div_{name}', 'n_clicks')],
        [State(f'{name}_modal', 'is_open')],
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

# @app.callback(
#     Output('fire_heatmap_modal', 'is_open'),
#     # [Input('fire_heatmap_expand_button', 'n_clicks')],
#     [Input('div_xxx', 'n_clicks')],
#     [State('fire_heatmap_modal', 'is_open')],
# )
# def toggle_modal(n1, is_open):
#     if n1:
#         return not is_open
#     return is_open
#

if __name__ == '__main__':
    pass

    # app.run_server(debug=True, host='0.0.0.0', port=8050,
    app.run_server(debug=True, host='192.168.1.200', port=8050,
                   dev_tools_hot_reload=True,
                   dev_tools_hot_reload_interval=1,
                   )
