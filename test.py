import dash
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

check_options = [{'label': c, 'value': c} for c in ['ch_a', 'ch_b']]
ch_a, ch_b = ['a1', 'a2'], ['b1', 'b2']

drop_options = [{'label': c, 'value': c} for c in ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3']]

check = dcc.Checklist(id='check', options=check_options, persistence=False)
drop = dcc.Dropdown(id='drop', options=drop_options, multi=True, persistence=False)

app.layout = html.Div([
    dcc.Store(id='store', storage_type='local'),
    check,
    drop,

])


@app.callback(
    Output('store', 'data'),
    [
        # Input('check', 'value'),
        Input('drop', 'value')]
)
def store_data(drp):
    # print(chk)
    print('Dropdown data: ', drp)
    print('=' * 100)
    store_data = []
    if drp: store_data += drp
    print('Saved: ', store_data)
    return store_data


#
@app.callback(
    Output('drop', 'value'),
    [Input('check', 'value')],
    [State('store', 'data')]
)
def load_data(chk, data):
    if data is None:
        raise PreventUpdate

    # chk = chk or []
    store_data = data

    print('Loaded: ', data, store_data)

    for c in chk:
        if c == 'ch_a': store_data += ch_a
        if c == 'ch_b': store_data += ch_b
    return store_data


#
# @app.callback(
#     Output('drop', 'value'),
#     [Input('check', 'value')],
#     [State('store','data')]
# )
# def add_check(chk, st):
#     print(st)
#     print('-'*80)
#     countries = [] if st is None else st
#     if chk:
#         for c in chk:
#             if c == 'ch_a':
#                 countries += ch_a
#             elif c == 'ch_b':
#                 countries += ch_b
#     countries=sorted(list(set(countries)))
#     # print(countries)
#     return countries
#
#
# @app.callback(
#     Output('store', 'data'),
#     [Input('drop', 'value')]
# )
# def store_drop(drp):
#     print(drp)
#     print('='*80)
#     return drp

# @app.callback(
#     Output('drop','value'),
#     [Input('store','data')]
# )
#
# def load_drop(drp):
#     return drp

#
#
# # create a layout with two multi-select dropdowns
# def get_dropdown(n, value=None):
#     value = [] if value is None else value
#     return html.Div(
#         [dcc.Dropdown(
#             id='dropdown' + n,
#             options=[
#                 {'label': 'New York City', 'value': 'NYC'},
#                 {'label': 'Montréal', 'value': 'MTL'},
#                 {'label': 'San Francisco', 'value': 'SF'}
#             ],
#             multi=True,
#             value=value
#         )],
#         id='dropdown-container' + n,
#
#     )
#
#
# app.layout = html.Div([
#     get_dropdown('1'),
#     get_dropdown('2'),
#     html.Div([], id='previously-selected', style={'display': 'none'})
# ])
#
#
# # Callback one recreates the other dropdown using new values
# #   and updates the previously-selected value
# @app.callback(
#     [Output('dropdown-container2', 'children'), Output('previously-selected', 'children')],
#     [Input('dropdown1', 'value')],
#     [State('previously-selected', 'children')]
# )
# def update_via_children(value, prev_selected):
#     print('callback one')
#
#     if sorted(value) == sorted(prev_selected):
#         raise PreventUpdate
#
#     return get_dropdown('2', value=value), value
#
#
# # Callback two updates the value of dropdown1 directly and so could be used to alter something
# #   complicated like a Graph, hopefully. Does not update previously-selected, so callback one
# #   will be called again, recreating dropdown2, triggering callback one a second time, but
# #   this time previously-selected == value
# @app.callback(
#     Output('dropdown1', 'value'),
#     [Input('dropdown2', 'value')],
#     [State('previously-selected', 'children')]
# )
# def update_directly(value, prev_selected):
#     print('callback two')
#     if sorted(value) == sorted(prev_selected):
#         raise PreventUpdate
#     return value

app.run_server(debug=True)

