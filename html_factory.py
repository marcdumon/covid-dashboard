# --------------------------------------------------------------------------------------------------------
# 2020/05/20
# src - html_factory.py
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


def generate_graph_component(name, titel=''):
    """
    Generates html code for a graph, including expand button and expand modal
    :param name:
    :param title:
    :return:
    """
    graph_html = [
        dcc.Graph(id=name, config={'displayModeBar': False}),
        dbc.Modal([
            dbc.ModalHeader(titel, style={'margin': '0'}),
            dbc.ModalBody([dcc.Graph(id=f'expanded_{name}',
                                     config={'displayModeBar': False},
                                     style={'height': '75vh'}),
                           html.Img(src='assets/images/config.png', id=f'config_{name}',
                           className='config-graph')
                           ]),

            # dbc.ModalFooter(),
        ],
            id=f'modal_expanded_{name}',
            centered=True,
            style={"max-width": "none", "width": "90%"}
        ),


    ]
    return graph_html


if __name__ == '__main__':
    print(generate_graph_component(('new_fantastic_graph')))
