# --------------------------------------------------------------------------------------------------------
# 2020/04/06
# src - chart_factory.py
# md
# --------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import plotly.express as px
import chart_studio.plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.figure_factory as ff
from my_tools.plotly_discrete_colorscale import plotly_discrete_colorscale


def add_new_line(fig, x, y, name, mode='lines+markers'):
    trace = go.Scatter(x=x, y=y, mode=mode, name=name)
    fig.add_trace(trace)
    return fig


def add_new_mean_line(fig, x, y, n, name, mode='lines+markers'):
    trace = go.Scatter(x=x, y=y.rolling(n).mean(), mode=mode, name=f'{name}_ma{n}', marker=dict(size=3))
    fig.add_trace(trace)
    return fig


def add_3d_line(fig, x, y, z, name, mode='lines+markers'):
    pass


def add_mean_3d_line(fig, x, y, z, n, name, mode='lines+markers'):
    pass


def create_fire_heatmap(df, column='cases', showscale=True):
    """
    Creates a flame heatmap like Yaneer Bar-Yam
    Example: https://twitter.com/yaneerbaryam/status/1258167125870555137/photo/1
    df: Dataframe: MultiIndex dataframe with date and location and 1 value colum.  Location is used for x-axis and date for y-axis. The value colums is used as z-axis (color)

    Returns: plotly fig object
    Input Example

        	date	location	region	        cases	rolling_cases
    0	2020-03-01	Antwerpen	Flanders	    1	    NaN
    1	2020-03-02	Antwerpen	Flanders	    1	    NaN
    2	2020-03-03	Antwerpen	Flanders	    5	    NaN
    3	2020-03-04	Antwerpen	Flanders	    6	    NaN
    4	2020-03-05	Antwerpen	Flanders	    11	    NaN
    ...	...	...	...	...	...
    755	2020-05-06	WestVlaanderen	Flanders	48	    53.000000
    756	2020-05-07	WestVlaanderen	Flanders	66	    55.142857
    757	2020-05-08	WestVlaanderen	Flanders	52	    59.000000
    758	2020-05-09	WestVlaanderen	Flanders	15	    56.714286
    759	2020-05-10	WestVlaanderen	Flanders	2	    54.857143
    """
    locations = list(df['location'])
    dates = list(df['date'])
    max_z = max(df[column])
    # create discrete colors
    bvals = [0, 1, 11, 31, 100, max_z]
    colors = ['#00FF00', '#FFFF00', '#EBB400', '#BD4C00', '#FF0000']
    if max_z > 1000:
        bvals.insert(-1, 1000)
        colors.append('#660066')
        # colors.append('#990099')
    dcolorscale = plotly_discrete_colorscale(bvals, colors)


    fig = go.Figure(data=go.Heatmap(x=locations, y=dates,
                                    # z=df[df.columns[0]],  # the only value column (usualy 'cases')
                                    z=df[column],
                                    # connectgaps=False,  # If True then plotly adds a wrong z-value when country has no data for that date
                                    # type='heatmap',
                                    colorscale=dcolorscale,
                                    showscale=showscale,
                                    xgap=.5, ygap=.5,
                                    ))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['xaxis']['dtick'] = 1
    fig['layout']['yaxis']['dtick'] = 7 * 86400000  # 1 day in ms

    return fig


def create_fire_heatmap_old(df, column='cases', showscale=True):
    """
    Creates a flame heatmap like Yaneer Bar-Yam
    Example: https://twitter.com/yaneerbaryam/status/1258167125870555137/photo/1
    df: Dataframe: MultiIndex dataframe with date and location and 1 value colum.  Location is used for x-axis and date for y-axis. The value colums is used as z-axis (color)

    Returns: plotly fig object
    Example
                                 cases
    date	province
    2020-04-29	VlaamsBrabant	    3
                Antwerpen	        1
    2020-04-28	WestVlaanderen	    25
                VlaamsBrabant	    24
                OostVlaanderen	    18

    """
    l = df.index.get_level_values(1)
    d = df.index.get_level_values(0)
    max_z = df.values.max()

    # create discrete colors
    bvals = [0, 1, 11, 31, 100, max_z]
    colors = ['#00FF00', '#FFFF00', '#EBB400', '#BD4C00', '#FF0000']
    if max_z > 1000:
        bvals.insert(-1, 1000)
        colors.append('#660066')

    dcolorscale = plotly_discrete_colorscale(bvals, colors)

    fig = go.Figure(data=go.Heatmap(x=l, y=d,
                                    # z=df[df.columns[0]],  # the only value column (usualy 'cases')
                                    z=df['cases'],
                                    # connectgaps=False,  # If True then plotly adds a wrong z-value when country has no data for that date
                                    # type='heatmap',
                                    colorscale=dcolorscale,
                                    showscale=showscale,
                                    xgap=.5, ygap=.5,
                                    ))
    # print(df[df.columns[0]].values)
    # print(d)
    # print(l)
    #
    # fig = ff.create_annotated_heatmap(x=l, y=d,
    #                                   z=df[df.columns[0]].values,  # the only value column (usualy 'cases'),
    #                                   annotation_text=df[df.columns[0]].values,
    #                                   colorscale=dcolorscale,
    #                                   showscale=showscale,
    #                                   )
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['dtick'] = 7 * 86400000  # 1 day in ms

    return fig


if __name__ == '__main__':
    pass
