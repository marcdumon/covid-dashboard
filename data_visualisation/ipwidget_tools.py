# --------------------------------------------------------------------------------------------------------
# 2020/04/10
# src - ipwidget_tools.py
# md
# --------------------------------------------------------------------------------------------------------

import difflib
import random

import requests
import pandas as pd
import ipywidgets as widgets


def multi_checkbox_widget(descriptions):
    """
        Source: https://gist.github.com/pbugnion/5bb7878ff212a0116f0f1fbc9f431a5c
        Widget with a search field and lots of checkboxes
    """
    search_widget = widgets.Text()
    options_dict = {description: widgets.Checkbox(description=description, value=False) for description in descriptions}
    options = [options_dict[description] for description in descriptions]
    # options_widget = widgets.VBox(options, layout={'overflow': 'scroll'})
    options_widget = widgets.VBox(options, layout=widgets.Layout(flex_flow='row wrap'))
    multi_select = widgets.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = [options_dict[description] for description in descriptions]
        else:
            # Filter by search field using difflib.
            close_matches = difflib.get_close_matches(search_input, descriptions, n=10, cutoff=0.0)
            new_options = [options_dict[description] for description in close_matches]
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    return multi_select


def date_slider(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d/%m/%y '), date) for date in dates]
    index = (0, len(options) - 1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '500px'}
    )
    return selection_range_slider


if __name__ == '__main__':
    pass