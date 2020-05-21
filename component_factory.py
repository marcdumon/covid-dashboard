# --------------------------------------------------------------------------------------------------------
# 2020/05/20
# src - component_factory.py
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


def generate_graph_componet(name, title=''):
    """
    Generates html code for a graph, including expand button and expand modal
    :param name:
    :param title:
    :return:
    """
    graph_html = [dbc.Button('Expand graph', id=f'{name}_expand_button', color='secondary', outline=True, style={'width': '100%'}),
                  dcc.Graph(id=name, config={'displayModeBar': False}),
                  dbc.Modal([dbc.ModalHeader(title),
                             dbc.ModalBody(dcc.Graph(id=f'{name}_expanded',
                                                     config={'displayModeBar': False},
                                                     style={'height': '80vh'})),
                             # dbc.ModalFooter(dbc.Button("Close", id=f'{name}_close_button'))
                             ],
                            id=f'{name}_modal',
                            centered=True,
                            style={"max-width": "none", "width": "90%"}
                            ),
                  ]

    return graph_html


if __name__ == '__main__':
    print(generate_graph_componet(('new_fantastic_graph')))
