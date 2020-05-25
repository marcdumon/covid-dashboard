# --------------------------------------------------------------------------------------------------------
# 2020/05/15
# src - covid_dashboard.py
# md
# --------------------------------------------------------------------------------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.graph_objs import Figure

from html_factory import generate_graph_component
from data_process import load_ecdc

import numpy as np
import pandas as pd

import plotly.express as px
import chart_studio.plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.figure_factory as ff
from my_tools.plotly_discrete_colorscale import plotly_discrete_colorscale

from data_visualisation import create_fire_heatmap, add_new_mean_line

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

good_countries = ['Australia', 'Austria', 'China', 'Croatia', 'Estonia', 'Greece', 'Iceland', 'Jordan', 'Lebanon', 'Luxembourg', 'Mauritius', 'New_Zeland', 'Norway',
                  'Slovakia', 'Slovenia', 'South_Korea', 'Taiwan', 'Thailand', 'Vietnam']
middle_countries = ['Azerbaijan', 'Belgium', 'Costa_Rica', 'Cyprus', 'Czechia', 'Denmark', 'France', 'Germany', 'Iran', 'Israel', 'Italy', 'Japan', 'Kyrgystan', 'Malaysia',
                    'Netherlands',
                    'Portugal', 'Spain', 'Switzerland', 'Tunisia', 'Turkey', 'Uzbekistan']
bad_countries = ['Argentina', 'Bahrain', 'Bangladesh', 'Belarus', 'Brazil', 'Canada', 'Chile', 'Ecuador', 'Egypt', 'Finland', 'Hungary', 'India', 'Indonesia', 'Iraq', 'Mali',
                 'Mexico',
                 'Panama', 'Peru', 'Philippines', 'Poland', 'Qatar', 'Romania', 'Russia', 'Singapore', 'Somalia', 'Sweden', 'United_Arab_Emirates', 'Ukraine', 'United_Kingdom',
                 'United_States_of_America']

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

country_groups_options = [
    {'label': 'Europe', 'value': 'Europe'},
    {'label': 'Asia', 'value': 'Asia'},
    {'label': 'All', 'value': 'All'},
    {'label': 'Good', 'value': 'Good'},
    {'label': 'Middle', 'value': 'Middle'},
    {'label': 'Bad', 'value': 'Bad'},

    {'label': 'Group1', 'value': 'Group1'}]

graph_names = ['fire_heatmap', 'deaths_per_cases', 'recent_total_cases']

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# HTML
app = dash.Dash(__name__, )
# app = dash.Dash(__name__, external_stylesheets=['10_bootstrap.css','20_my_css.css'])
# app.config.suppress_callback_exceptions = True #

header = html.Div([
    dbc.Row(
        [
            dbc.Col(html.H1('COVID-19 Dashboard'), style={'text-align': 'left'}, width=3),
            dbc.Col(dbc.Button('Select Countries', id='open', outline=True, color='secondary', style={'margin': '0rem 1rem 0rem 0rem', }), width=3),
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
                id='modal_countries',
                size='lg',
                centered=True),
        ],
        align='center',
    )], style={'width': '100%',
               'top': '0px',
               'overflow': 'hidden',
               'position': 'fixed',
               'z-index': '100',
               'background-color': 'white',
               'border-bottom': '1px solid lightgray'
               })

graphs = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div(
                generate_graph_component('fire_heatmap', 'Fire Heatmap'),
                id='div_fire_heatmap'), width=12),
            dbc.Modal([
                dbc.ModalHeader(),
                dbc.ModalBody([html.Div([
                    daq.Slider(
                        id='config_fire_heatmap_slider_rolling',
                        min=1, max=100, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                    html.Div(id='config_fire_heatmap_div_rolling', style={'padding': '0px 0px 25px'})
                ]),
                    html.Div([daq.Slider(
                        id='config_fire_heatmap_slider_sorting',
                        min=1, max=100, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                        html.Div(id='config_fire_heatmap_div_sorting')])
                ])],
                id='modal_config_fire_heatmap'),

        ]),
    dbc.Row(
        [
            dbc.Col(html.Div(
                generate_graph_component('recent_total_cases', 'total cases last {n} days / total confirmed cases'),
                id='div_recent_total_cases'),
                width=6),
            dbc.Modal([
                dbc.ModalHeader(),
                dbc.ModalBody([html.Div([
                    daq.Slider(
                        id='config_recent_total_cases_slider_last_days',
                        min=1, max=100, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                    html.Div(id='config_recent_total_cases_div_last_days', style={'padding': '0px 0px 25px'})
                ]),
                    html.Div([daq.Slider(
                        id='config_recent_total_cases_slider_rolling',
                        min=1, max=150, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                        html.Div(id='config_recent_total_cases_div_rolling')])
                ])
            ],
                id='modal_config_recent_total_cases'),

            # dbc.Col(width=1),
            dbc.Col(html.Div(
                generate_graph_component('deaths_per_cases', 'total deaths last {n} days / total confirmed cases last {n} days'),
                id='div_deaths_per_cases'),
                width=6),
            dbc.Modal([
                dbc.ModalHeader(),
                dbc.ModalBody([html.Div([
                    daq.Slider(
                        id='config_deaths_per_cases_slider_last_days',
                        min=1, max=100, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                    html.Div(id='config_deaths_per_cases_div_last_days', style={'padding': '0px 0px 25px'})
                ]),
                    html.Div([daq.Slider(
                        id='config_deaths_per_cases_slider_rolling',
                        min=1, max=150, step=1, value=7,
                        dots=False, updatemode='drag', persistence=True),
                        html.Div(id='config_deaths_per_cases_div_rolling')])
                ])],

                id='modal_config_deaths_per_cases')
        ])
])

app.layout = html.Div(
    [
        header,
        graphs
    ],
    style={'margin-top': '100px', 'margin-left': '25px', }
)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# CALLBACKS
@app.callback(Output('dropdown_countries', 'value'),

              [Input('checklist_country_groups', 'value')])
def add_country_group(country_groups):
    if country_groups is None:
        raise PreventUpdate
    else:
        countries = []
        for country_group in country_groups:
            if country_group == 'Europe': countries += europe_countries
            if country_group == 'Asia': countries += asia_countries
            if country_group == 'All': countries += all_countries
            if country_group == 'Group1': countries += group1
            if country_group == 'Good': countries += good_countries
            if country_group == 'Middle': countries += middle_countries
            if country_group == 'Bad': countries += bad_countries

        countries = sorted(list(set(countries)))
        return countries


@app.callback(
    Output('modal_countries', 'is_open'),
    [Input('open', 'n_clicks'), Input('close', 'n_clicks')],
    [State('modal_countries', 'is_open')],
)
def modal_countries(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


#  GRAPHS CALLBACKS

@app.callback([Output('fire_heatmap', 'figure'), Output('expanded_fire_heatmap', 'figure')],
              [Input('dropdown_countries', 'value'), Input('config_fire_heatmap_slider_rolling', 'value'), Input('config_fire_heatmap_slider_sorting', 'value')],
              )
def generate_fire_heatmap(countries, rolling, sorting):
    if not countries:
        raise PreventUpdate

    e = ecdc[ecdc['location'].isin(countries)]
    df = e.copy()
    # Set negative cases to 0
    df['cases'][df['cases'] < 0] = 0

    # Add rolling mean
    # rolling = 100
    df['rolling_cases'] = df.groupby(['location'])['cases'].rolling(rolling).mean().reset_index(0)['cases']
    df.fillna(0, inplace=True)
    # Sort x-axis
    # n = 7
    sort_column = 'cases'
    from_date = max(df['date']) - pd.DateOffset(sorting)
    sort = df[df['date'] > from_date].groupby(['location'])[sort_column].sum().sort_values(ascending=False).index.values
    df['sort'] = pd.Categorical(df['location'], categories=sort, ordered=True)  # See https://stackoverflow.com/questions/26707171/sort-pandas-dataframe-based-on-list
    df.sort_values('sort', inplace=True)
    df.drop('sort', axis=1, inplace=True)
    fig = create_fire_heatmap(df, column='rolling_cases', showscale=False)
    # fig = create_fire_heatmap(df, column='cases', showscale=False)

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


@app.callback(Output('config_fire_heatmap_div_rolling', 'children'),
              [Input('config_fire_heatmap_slider_rolling', 'value')])
def fire_heatmap_config_rolling(rolling):
    return f'Rolling mean = {rolling}'


@app.callback(Output('config_fire_heatmap_div_sorting', 'children'),
              [Input('config_fire_heatmap_slider_sorting', 'value')])
def fire_heatmap_config_rolling(sorting):
    return f'Nr of days for sorting = {sorting}'


@app.callback([Output('deaths_per_cases', 'figure'), Output('expanded_deaths_per_cases', 'figure')],
              [Input('dropdown_countries', 'value'), Input('config_deaths_per_cases_slider_last_days', 'value'), Input('config_deaths_per_cases_slider_rolling', 'value')])
def generate_deaths_per_cases(countries, last_days, rolling):
    fig = go.Figure()
    for country in countries:
        # ecdc_country = ecdc.loc[(ecdc['date'] >= ds.value[0]) & (ecdc['date'] <= ds.value[1]) & (ecdc['country'] == country), :]
        ecdc_country = ecdc.loc[ecdc['location'] == country, :]
        ecdc_country_10 = ecdc_country.loc[:, ['cases', 'deaths']].rolling(last_days).sum()

        add_new_mean_line(fig, ecdc_country_10['cases'], ecdc_country_10['deaths'], rolling, name=country)
    fig.update_xaxes(
        # range=[0, 5.8],
        title_text=f'total confirmed cases last {last_days}d')
    fig.update_yaxes(
        # range=[0, 4.8],
        title_text=f'total deaths last {last_days}d')
    fig.add_shape(type="line", x0=10000, y0=10, x1=1000000, y1=1000, line=dict(color="MediumPurple", width=1, dash='dash'))
    annotations = [dict(x=4.9, y=2, xref='x', yref='y', text='case fatality rate 0.1% (flue)', ax=0, ay=0, textangle=-44)]
    fig.update_layout(yaxis_type="log", xaxis_type='log')
    fig.update_layout(annotations=annotations)
    fig.update_layout(
        margin=dict(t=2, r=2, b=2, l=2),
        # showlegend=False,
        autosize=True,
        xaxis_showgrid=True, yaxis_showgrid=True,
        xaxis_gridcolor='#d3d3d3', yaxis_gridcolor='#d3d3d3',
        xaxis_fixedrange=True, yaxis_fixedrange=True,  # disable zoom
        xaxis_showspikes=True, yaxis_showspikes=True,
        # displayModeBar=False,
        plot_bgcolor='#ffffff'
    )
    return fig, fig


@app.callback(Output('config_deaths_per_cases_div_last_days', 'children'),
              [Input('config_deaths_per_cases_slider_last_days', 'value')])
def deaths_per_cases_config_last_days(last_days):
    return f'Last days = {last_days}'


@app.callback(Output('config_deaths_per_cases_div_rolling', 'children'),
              [Input('config_deaths_per_cases_slider_rolling', 'value')])
def deaths_per_cases_config_rolling(rolling):
    return f'Rolling mean = {rolling}'


@app.callback([Output('recent_total_cases', 'figure'), Output('expanded_recent_total_cases', 'figure')],
              [Input('dropdown_countries', 'value'), Input('config_recent_total_cases_slider_last_days', 'value'), Input('config_recent_total_cases_slider_rolling', 'value')])
def generate_recent_total_cases(countries, last_days, rolling):
    fig = go.Figure()
    n = last_days
    m = rolling
    for country in countries:
        # Todo: Refactor to make much faster + save for reuse
        ecdc.loc[ecdc['location'] == country, 'total_cases'] = ecdc.loc[ecdc['location'] == country, 'cases'].cumsum()
        ecdc.loc[ecdc['location'] == country, 'total_deaths'] = ecdc.loc[ecdc['location'] == country, 'deaths'].cumsum()

        # ecdc_country = ecdc.loc[(ecdc['date'] >= ds.value[0]) & (ecdc['date'] <= ds.value[1]) & (ecdc['country'] ==
        ecdc_country = ecdc.loc[ecdc['location'] == country, :]
        ecdc_country_n = ecdc_country.loc[:, ['cases', 'deaths']].rolling(n).sum().rename(columns={'cases': f'total_cases_{n}', 'deaths': f'total_deaths_{n}'})
        ecdc_country = pd.concat([ecdc_country, ecdc_country_n], axis=1)
        add_new_mean_line(fig, ecdc_country[f'total_cases'], ecdc_country[f'total_cases_{n}'], m, name=country, mode='lines')
    fig.update_xaxes(range=[2, 6.5], title_text=f'total cases')
    fig.update_yaxes(range=[1.5, 6], title_text=f'total cases last {n} days')

    # fig.update_layout( title=f'total cases last {n} days / total cases')
    # fig.update_layout(annotations=annotations)
    fig.update_layout(xaxis_type='log', yaxis_type='log')
    fig.update_layout(
        margin=dict(t=2, r=2, b=2, l=2),
        # showlegend=False,
        autosize=True,
        xaxis_showgrid=True, yaxis_showgrid=True,
        xaxis_gridcolor='#d3d3d3', yaxis_gridcolor='#d3d3d3',
        xaxis_fixedrange=True, yaxis_fixedrange=True,  # disable zoom
        xaxis_showspikes=True, yaxis_showspikes=True,
        # displayModeBar=False,
        plot_bgcolor='#ffffff'
    )

    return fig, fig


@app.callback(Output('config_recent_total_cases_div_last_days', 'children'),
              [Input('config_recent_total_cases_slider_last_days', 'value')])
def recent_total_cases_config_last_days(last_days):
    return f'Last days = {last_days}'


@app.callback(Output('config_recent_total_cases_div_rolling', 'children'),
              [Input('config_recent_total_cases_slider_rolling', 'value')])
def recent_total_cases_config_rolling(rolling):
    return f'Rolling mean = {rolling}'


for name in graph_names:
    @app.callback(Output(f'modal_expanded_{name}', 'is_open'),
                  [Input(f'div_{name}', 'n_clicks')],
                  [State(f'modal_expanded_{name}', 'is_open')])
    def toggle_modal_expanded(n1, is_open):
        if n1: return not is_open
        return is_open


    @app.callback(Output(f'modal_config_{name}', 'is_open'),
                  [Input(f'config_{name}', 'n_clicks')],
                  [State(f'modal_config_{name}', 'is_open')])
    def toggle_modal_config(n1, is_open):
        if n1: return not is_open
        return is_open

if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8050)
    # app.run_server(debug=True, threaded=True, port=10450)
    # app.server.run(debug=True)
    app.run_server(debug=True, host='192.168.1.200', port=8050,
                   dev_tools_hot_reload=True,
                   dev_tools_hot_reload_interval=1,
                   threaded=True, )
