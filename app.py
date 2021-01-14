import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies  as dd
import dash_table
import plotly.express as px
import plotly.graph_objects as go

# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras import Sequential, Input, Model, layers

import io
import os
from base64 import decodebytes
import datetime
import re

import pandas as pd
import numpy as np

from flask import Flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', './assests/app.css']

server = Flask('mod4-dash')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# begin the code to set up the data for the application
# read in our data from  the native JSON 
dfTemp = pd.read_json('politifact_info.json', orient='', )
df = pd.DataFrame()
for column in dfTemp.columns:
  df[column] = dfTemp[column][0]

drop1 = df.loc[df['truth_value'] == 'full-flop'].index
drop2 = df.loc[df['truth_value'] == 'half-flip'].index
drop3 = df.loc[df['truth_value'] == 'no-flip'].index

dfFinal = df.drop(index = drop1.append(drop2).append(drop3))
dfFinalInitial = dfFinal.groupby('author').count()
dfFinalIntermediate = pd.get_dummies(data = dfFinal, columns=['truth_value'])
dfFinalEncodded = dfFinalIntermediate.groupby('author').sum()
dfFinalEncodded['total'] = dfFinalInitial['truth_value']
dfFinalEncodded = dfFinalEncodded.sort_values('total', axis = 0, ascending=False)

truthColumns = [
  'truth_value_pants-fire', 
  'truth_value_false', 
  'truth_value_barely-true',
  'truth_value_half-true',
  'truth_value_mostly-true',
  'truth_value_true']
fig1 = go.Figure(
    layout = {'barmode' : 'stack'}
)
for column in truthColumns:
  fig1.add_trace(go.Bar(
    x =dfFinalEncodded[:10][column], 
    y = dfFinalEncodded[:10].index,
    orientation = 'h',
  ))

app.layout = html.Div(
  children=[
    dcc.Dropdown(
        id='dropdown_1',
        options=[
            {'label': 'Raw Numbers', 'value': 'raw'},
            {'label': 'Percentages', 'value': 'norm'},
        ],
        value='raw'
    ),
    dcc.Input(
            id="input_1",
            type='search',
            placeholder="Search for an Author",
    ),
    dcc.Graph(
      id = 'graph_1',
      figure = fig1
    ),
    dcc.Slider(
      id='graph_1_slider',
      min = 5,
      max=50,
      value=10,
      step=1,
      updatemode='drag',
      marks = {
        5:'Top 5',
        10: 'Top 10',
        15: 'Top 15',
        20: 'Top 20',
        25: 'Top 25',
        30: 'Top 30',
        35: 'Top 35',
        40: 'Top 40',
        45: 'Top 45',
        50: 'Top 50'
      }
    ),

    dash_table.DataTable(
        id='datatable_1',
        style_cell={
          'whiteSpace': 'normal',
          'height': 'auto',
        },  
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
        persistence = True,
    ),
  ],
  className='app')

@app.callback(
  dd.Output('graph_1','figure'),
  [
    dd.Input('graph_1_slider', 'value'),
    dd.Input('dropdown_1', 'value'),
    dd.Input('input_1', 'value')
  ])
def update_graph_1(sliderValue, dropdownValue, inputValue):
  truthColumns = [
    'truth_value_pants-fire', 
    'truth_value_false', 
    'truth_value_barely-true',
    'truth_value_half-true',
    'truth_value_mostly-true',
    'truth_value_true'
  ]
  colorMap = {
    'truth_value_pants-fire' : 'maroon',
    'truth_value_false' : 'red',
    'truth_value_barely-true' : 'orange',
    'truth_value_half-true' : 'yellow',
    'truth_value_mostly-true' : 'lightgreen',
    'truth_value_true' : 'green'
  }
  fig = go.Figure(
      layout = {'barmode' : 'stack'}
  )
  if inputValue == '' or inputValue == None:
    if dropdownValue == 'raw':
      for column in truthColumns:
        fig.add_trace(go.Bar(
          x =dfFinalEncodded[:sliderValue][column], 
          y = dfFinalEncodded[:sliderValue].index,
          marker_color = colorMap[column],
          name = f'{column.split("_")[2]}',
          orientation = 'h',
      ))
      fig.update_layout(
        title_text = 'Quotes by Author: Raw Numbers',
      )
      return fig
    else:
      for column in truthColumns:
        fig.add_trace(go.Bar(
          x =dfFinalEncodded[:sliderValue][column]/dfFinalEncodded[:sliderValue]['total'], 
          y = dfFinalEncodded[:sliderValue].index,
          marker_color = colorMap[column],
          name = f'{column.split("_")[2]}',
          orientation = 'h',
      ))
      fig.update_layout(
        title_text = 'Quotes by Author: Percentages',
      )
      return fig
  else:
    if dropdownValue == 'raw':
      for column in truthColumns:
        fig.add_trace(go.Bar(
          x = dfFinalEncodded.loc[dfFinalEncodded.index.str.contains(f'{inputValue.title()}')][column][:sliderValue], 
          y = dfFinalEncodded.loc[dfFinalEncodded.index.str.contains(f'{inputValue.title()}')].index,
          marker_color = colorMap[column],
          name = f'{column.split("_")[2]}',
          orientation = 'h',
      ))
      fig.update_layout(
        title_text = 'Quotes by Author: Raw Numbers',
      )
      return fig
    else:
      for column in truthColumns:
        fig.add_trace(go.Bar(
          x = dfFinalEncodded.loc[dfFinalEncodded.index.str.contains(f'{inputValue.title()}')][column][:sliderValue]/dfFinalEncodded.loc[dfFinalEncodded.index.str.contains(f'{inputValue.title()}')]['total'][:sliderValue], 
          y = dfFinalEncodded.loc[dfFinalEncodded.index.str.contains(f'{inputValue.title()}')].index,
          marker_color = colorMap[column],
          name = f'{column.split("_")[2]}',
          orientation = 'h',
      ))
      fig.update_layout(
        title_text = 'Quotes by Author: Percentages',
      )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)