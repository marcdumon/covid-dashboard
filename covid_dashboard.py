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
from data_process import load_ecdc

import numpy as np
import pandas as pd

import plotly.express as px
import chart_studio.plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.figure_factory as ff
from my_tools.plotly_discrete_colorscale import plotly_discrete_colorscale

# DATA
from data_visualisation import create_fire_heatmap

ecdc = load_ecdc()
region_location = ecdc[['region', 'location']]
europe_countries = region_location[region_location['region'] == 'Europe']['location'].unique().tolist()
asia_countries = region_location[region_location['region'] == 'Asia']['location'].unique().tolist()
other_countries = region_location[region_location['region'] == 'Other']['location'].unique().tolist()
all_countries = ecdc['location'].unique().tolist()

country_groups_options = [{'label': 'Europe', 'value': 'Europe'}, {'label': 'Asia', 'value': 'Asia'}, {'label': 'Other', 'value': 'Other'}, {'label': 'All', 'value': 'All'}]


selected_countries = []

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

header = html.Div([
    dbc.Row(dbc.Col(html.Div(html.H1('COVID-19 Dashboard')), width=5),
            justify='center'),
    dbc.Row(
        [
            dbc.Col('x ' * 1000, width=2),
            dbc.Col(html.P('y ' * 1000), width=2),
            dbc.Col(html.P('z ' * 1000), width=2),
        ],
        justify='center',
        no_gutters=True)
])

country_selection = html.Div([
    dbc.Row(dbc.Col(
        [
            dbc.Button("Select countries", id="open"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Header"),
                    dbc.ModalBody(html.Div([
                        dcc.Checklist(id='checklist_country_groups',
                                      options=country_groups_options,
                                      value=['Europe'],
                                      labelStyle={'display': 'inline-block'}),
                        dcc.Dropdown(id='dropdown_countries',
                                     options=[{'label': c, 'value': c} for c in all_countries],
                                     multi=True)])),
                    dbc.ModalFooter(dbc.Button("Close", id="close", className="ml-auto")),
                ],
                id="modal",
                size="lg",
                centered=True)
        ]
    ), justify='center')
])

graphs = html.Div([
    dbc.Row(
        [
            dbc.Col([
                dbc.Button('Expand graph', id='expand_button1'),
                dcc.Graph(id='test_graph', config={'displayModeBar': False})], width=4),
            dbc.Col(dcc.Graph(id='test_graph1'), width=4),
            dbc.Col(dcc.Graph(id='test_graph2'), width=4)
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.P('cxcczxczxczx'),
                dcc.Graph(id='test_graph4', config={'displayModeBar': False})], width=4),
            dbc.Col(dcc.Graph(id='test_graph5'), width=4),
            dbc.Col(dcc.Graph(id='test_graph6'), width=4)
        ]),
])



expanded_graph = dbc.Modal(
    [dbc.ModalHeader(country_selection),
     dbc.ModalBody(
         dcc.Graph(id='test_graph_expanded',
                   config={'displayModeBar': False},
                   style={"height": "80vh"})),
     dbc.ModalFooter(dbc.Button("Close", id="close2"))],
    id="modal2",
    centered=True,
    style={"max-width": "none", "width": "90%"}
)

app.layout = dbc.Container(
    [
        header,
        # country_selection,
        graphs,
        expanded_graph

    ],
    fluid=True, )


@app.callback(Output('dropdown_countries', 'value'),
              [Input('checklist_country_groups', 'value')])
def add_country_group(country_groups):
    countries = []
    for country_group in country_groups:
        if country_group == 'Europe': countries += europe_countries
        if country_group == 'Asia': countries += asia_countries
        if country_group == 'Other': countries += other_countries
        if country_group == 'All': countries += all_countries
    countries = sorted(list(set(countries)))
    return countries


@app.callback([Output('test_graph', 'figure'), Output('test_graph_expanded', 'figure')],
              [Input('dropdown_countries', 'value')])
def generate_test_graph(countries):
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

    return (fig, fig)


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal2", "is_open"),
    [Input("expand_button1", "n_clicks"), Input("close2", "n_clicks")],
    [State("modal2", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
