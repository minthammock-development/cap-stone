import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
import dash.dependencies  as dd
import dash_table
import plotly.express as px
import plotly.graph_objects as go

from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pyLDAvis
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
import json
import string
import pandas as pd
import numpy as np

from flask import Flask

external_stylesheets = ['./assets/app.css']

server = Flask('capstone-dash-app')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# begin the code to set up the data for the application
# read in our data from  the native JSON 
dfTemp = pd.read_json('politifact_info.json', orient='', )
df = pd.DataFrame()
for column in dfTemp.columns:
  df[column] = dfTemp[column][0]

#There are some bad rows that we need to drop
drop1 = df.loc[df['truth_value'] == 'full-flop'].index
drop2 = df.loc[df['truth_value'] == 'half-flip'].index
drop3 = df.loc[df['truth_value'] == 'no-flip'].index

dfFinal = df.drop(index = drop1.append(drop2).append(drop3))
jsonFinal = dfFinal.to_json()

# Do some groupby methods to make pretty visuals
dfFinalInitial = dfFinal.groupby('author').count()
dfFinalIntermediate = pd.get_dummies(data = dfFinal, columns=['truth_value'])
dfFinalEncodded = dfFinalIntermediate.groupby('author').sum()
dfFinalEncodded['total'] = dfFinalInitial['truth_value']
dfFinalEncodded = dfFinalEncodded.sort_values('total', axis = 0, ascending=False)

# this is defined in global scobe because many function use it
truthColumns = [
  'truth_value_pants-fire', 
  'truth_value_false', 
  'truth_value_barely-true',
  'truth_value_half-true',
  'truth_value_mostly-true',
  'truth_value_true']

# create the figure that will be the default when loading the app
fig1 = go.Figure(
    layout = {'barmode' : 'stack'}
)
for column in truthColumns:
  fig1.add_trace(go.Bar(
    x =dfFinalEncodded[:10][column], 
    y = dfFinalEncodded[:10].index,
    orientation = 'h',
  ))

# def create_word_2_vec(df, textColumn, partitionColumn, stopwordsList = None, size = 100, window = 5, min_count = 1, workers = 1, epochs = 5, kwargs = {}):
#   def data_partition(df, partitionColumn):
#     uniqueValues = df[partitionColumn].unique()
#     partition = {}
#     for value in uniqueValues:
#       part = df.loc[df[partitionColumn] == value]
#       partition[value] = (part)
#     return partition
#   partitionList = data_partition(df, partitionColumn)
#   word2VecPartition = {}
#   for truthValue,part in partitionList.items():
#     lemmatizer = WordNetLemmatizer()
#     part_tokens = part[textColumn].map(word_tokenize)
#     part_tokens_lemmatized = []
#     for text in part_tokens:
#       temp = []
#       for word in text:
#         if word not in string.punctuation and word not in stopwordsList:
#           temp.append(lemmatizer.lemmatize(word.lower()))
#       part_tokens_lemmatized.append(temp)

#     wordToVec = Word2Vec(part_tokens_lemmatized, size = size, window = window, min_count = min_count, workers = workers, **kwargs)
#     wordToVec.train(part_tokens_lemmatized, total_examples=wordToVec.corpus_count, epochs = epochs)
#     word2VecPartition[truthValue] = wordToVec
#   lemmatizer = WordNetLemmatizer()
#   all_tokens = df[textColumn].map(word_tokenize)
#   all_tokens_lemmatized = []
#   for text in all_tokens:
#     temp = []
#     for word in text:
#       if word not in string.punctuation and word not in stopwordsList:
#         temp.append(lemmatizer.lemmatize(word.lower()))
#     all_tokens_lemmatized.append(temp)

#   wordToVec = Word2Vec(all_tokens_lemmatized, size = size, window = window, min_count = min_count, workers = workers, **kwargs)
#   wordToVec.train(all_tokens_lemmatized, total_examples=wordToVec.corpus_count, epochs = epochs)
#   word2VecPartition['all'] = wordToVec
#   return word2VecPartition

# word2VecList = create_word_2_vec(dfFinal, 'quote', 'truth_value',stopwordsList = [], size = 150, window = 5, min_count=1, workers=4, epochs = 5)


# begin the layout of the application
app.layout = html.Div(
  id = 'body_container',
  children=[
    dcc.Tabs(
      children=[
        dcc.Tab(
          id = 'tab-1',
          label = 'Tab 1',
          value = 'tab_1',
          children=[
            html.Div(
              className = 'graph_editors',
              children=[
                dcc.Dropdown(
                  # className = 'graph',
                  id='dropdown_1',
                  options=[
                    {'label': 'Raw Numbers', 'value': 'raw'},
                    {'label': 'Percentages', 'value': 'norm'},
                  ],
                  value='raw'
                ),
                dcc.Input(
                  # className = 'graph',
                  id="input_1",
                  type='search',
                  placeholder="Search for an Author",
                ),
              ]
            ),
            html.Div(
              className = 'level_1_container',
              children=[
                html.Div(
                  className = 'graph_container',
                  children = [
                    dcc.Graph(
                      className = 'graph',
                      id = 'graph_1',
                      figure = fig1
                    ),
                    dcc.Slider(
                      className = 'graph',
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
                  ]
                ),
                html.Div(
                  className = 'text_container',
                  children = [
                    html.H2(
                      children = [],
                      id = 'quote_display'
                    )
                  ]
                )
              ]
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
              filter_action="custom",
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
          ]
        ),
        dcc.Tab(
          id = 'tab-2',
          label = 'Tab 2',
          value = 'tab_2',
          children=[
            html.H1(
              'Data Understanding'
            ),
            html.P(
              '''
                The data used for this paper was pulled from Politifact.com. 
                Politifact is a lovely organization that takes the time to investigate viral claims that deal with hot or sensative topics. 
                Unsurprisingly, many of these claims are political in nature, but not strictly so. 
                Politifact investigates these claims from variosu sources and writes a small article summarizing their findings as to the validity of the content. 
                They do the hard work! If you find this paper at all interesting it might be worth sending a little money their way. 
                5 bucks people, I'm not asking for the moon. 
                Anyways, Politifact vets the claims and boils it down to six distinct categories: Pants on fire, false, barly-true, half-true and true. 
                Pants fire is a special category reserved for falsehoods that not only lack any semblence of truth, but are also particularly outlandish or detrimental. 
                A good example would be, "The democrates say they are going to blow up the moon." 
                The classifications are somewhat subjective but because professionals took the time to look into the truth and cite their sources, I hold the labels as gold standard when it comes to verifying the level of truth in a statement. 
                There's one problem with this approach. 
                Research is hard and Politifact has only done 19000 evaluations over the past 12 years or so. 
                Yes, that's a lot of investigations, but not when thinking about NLP neural networks which are often trained on bodies of work as large as all of wikipedia.
                Because of the raw data limitation, I'm skeptical of the ability to classify truth to say +80% accuracy. 
                Ultimately, the work done here will need to be supercharged with data in order to reach it's true potential. 
                For now, let's take a look at what we have. 
                An important consideration ever present in modeling is class imbalance. 
                Here's our given composition.
              '''
            ),
            html.Img(src = "https://drive.google.com/uc?id=1EjxE9yDNB9mj2lFiUfnypbdTDUOcZMKC"),
            html.P(
              '''
                The imbalence is not as severe as it could be, but this still merits adjusting for. 
                Sci-kit learn has a particular function which calculated the weights needed for each class in order to mimic balanced amounts of data. 
                This is the primary tool I am going to use to address the imbalance, please see the code sections for specifics on the implementation. 
                With imbalance addressed we will take a detour into the data itself. 
                Our dataset includes 18436 different records. 
                These records have three text columns. 
                Our target column is the truth_value column which is the basis of our supervised learning task for this paper. 
                In addition, we have a column of quotes which are consumed by our model in order to make predictions on the truth value. 
                The final column is the author column which routinely accounts for the person or organization responsible for the creation of the quote. 
                What do these columns look like? 
                The next visual represents the 10 most frequent authors and the breakdown of their quotes into the various truth values.
              '''
            ),
            html.Img(src = "https://drive.google.com/uc?id=1gL4xn0HS9aSG4sG4O38i6uVu8MV_Zqlh"),
            html.P(
              '''
                With the rise of social media, it makes sense that those at the top of the list are figures in the modern political scene. 
                In general, Politifact reviews quotes with implications for the real world. 
                With the prolification of mis-information as a political tool, much of their work has centered on uncovering the truth behind statements made in the political sphere. 
                Please note that there are lumped categories such as "Facebook posts" and "Viral image". 
                These categories include quotes that achieved viral status in spreading through social media sites. 
                You won't find a good definition of viral in this paper as the exact criteria used to determine what elevates something to viral status isn't exactly clear. 
                For a better sense of the nature of the quotes themselves, please see the hosted application for this paper. 
                Given the top authors are pulled mostly from the political realm, it follows that words common in modern political times are highly represented. 
                Check out the following two word clouds for a more intuitive idea about the nature of the quotes corpus. 
                This example excludes stop words from consideration (the, and, where, in, is, I, etc...).
              '''
            ),
            html.Img(src = "https://drive.google.com/uc?id=1aJmMaPmbhdrM2VB7OGGISv7sRhe61na4" ),
            html.P(
              '''
                Unsuprizingly, the word cloud that excludes stop words consists of what we might expect: Words about hot topic issues, words about the voting process, locations, individuals in politics, money, governmental institutions and the like. 
                Summary statistics are also worth considering. 
                While using the quotes to determine the truth value will ultimately be the job of our models, let's see if there's any differences we can spot that might seperate the classes at hand. 
              '''
            ),
            html.Img(src = "https://drive.google.com/uc?id=1pMjZ6eUHw2rFrXjsCD-ip3LTxS9xVG7Q"),
          ]
        ),
        dcc.Tab(
          id = 'tab-3',
          label = 'Tab 3',
          value = 'tab_3',
          children=[
            html.Div(
              className = 'graph_editors',
              children=[
                dcc.Dropdown(
                  # className = 'graph',
                  id='dropdown_2',
                  options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'True', 'value': 'true'},
                    {'label': 'Mostly-True', 'value': 'mostly-true'},
                    {'label': 'Half-True', 'value': 'half-true'},
                    {'label': 'Barely-True', 'value': 'barely-true'},
                    {'label': 'False', 'value': 'false'},
                    {'label': 'Pants-On-Fire', 'value': 'pants-fire'},
                  ],
                  value='all'
                ),
                dcc.Input(
                  # className = 'graph',
                  id="input_2",
                  type='search',
                  placeholder="Words Like",
                ),
                dcc.Input(
                  # className = 'graph',
                  id="input_3",
                  type='search',
                  placeholder="Words Not like",
                ),
                html.Button(
                  id = 'button_1',
                  value = 'Show Associations'
                ),
              ]
            ),
            cyto.Cytoscape(
              id='node_graph',
              layout={'name': 'preset'},
              style={'width': '90%', 'height': '1000px'},
              elements=[
                  {'data': {'id': 'one', 'label': 'Use the search bars to get started'}, 'position': {'x': 0, 'y': 0}},
              ]
            )
          ]
        ),
        dcc.Tab(
          id = 'tab-4',
          label = 'Tab 4',
          value = 'tab_4',
          children=[
            html.Iframe(
              id = 'lda_frame',
              src = 'assets/lda_model_all.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_true_frame',
              src = 'assets/lda_model_true.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_barely_true_frame',
              src = 'assets/lda_model_barely-true.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_half_true_frame',
              src = 'assets/lda_model_half-true.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_mostly_true_frame',
              src = 'assets/lda_model_mostly-true.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_false_frame',
              src = 'assets/lda_model_false.html',
              height = '1000px',
              width = '80%',
            ),
            html.Iframe(
              id = 'lda_pants_fire_frame',
              src = 'assets/lda_model_pants-fire.html',
              height = '1000px',
              width = '80%',
            )
          ]
        ),
      ]
    ),  
  ],
  className='app')



##############################################################################################################################################



@app.callback(
    dd.Output('datatable_1', "data"),
    dd.Input('datatable_1', "filter_query"))
def update_table(filter):
  operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]
  def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

  filtering_expressions = filter.split(' && ')
  dff = dfFinal
  for filter_part in filtering_expressions:
      col_name, operator, filter_value = split_filter_part(filter_part)

      if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
          # these operators match pandas series operator method names
          dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
      elif operator == 'contains':
          dff = dff.loc[dff[col_name].str.contains(filter_value)]
      elif operator == 'datestartswith':
          # this is a simplification of the front-end filtering logic,
          # only works with complete fields in standard format
          dff = dff.loc[dff[col_name].str.startswith(filter_value)]

  return dff.to_dict('records')

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
        xaxis = go.layout.XAxis(
          title = 'Number of Quotes',),
        yaxis = go.layout.YAxis(
          title = 'Author',),
        hoverlabel = go.layout.Hoverlabel(
          
        )
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
          xaxis = go.layout.XAxis(
            title = 'Percentage of Quotes',),
          yaxis = go.layout.YAxis(
            title = 'Author',),
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
          xaxis = go.layout.XAxis(
            title = 'Number of Quotes',),
          yaxis = go.layout.YAxis(
            title = 'Author',),
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
        xaxis = go.layout.XAxis(
          title = 'Percentage of Quotes',),
        yaxis = go.layout.YAxis(
          title = 'Author',),
      )
      return fig

@app.callback(
  dd.Output('quote_display', 'children'),
  dd.Input('graph_1', 'hoverData')
)
def get_quotes(hoverData):
  if hoverData == None:
    return ''
  truthColumns = [
    'pants-fire', 
    'false', 
    'barely-true',
    'half-true',
    'mostly-true',
    'true'
  ]
  curveNumber = hoverData['points'][0]['curveNumber']
  label = truthColumns[curveNumber]
  author = hoverData['points'][0]['y']
  hoverSlice = dfFinal.loc[dfFinal.author == author]
  hoverSlice = hoverSlice.loc[hoverSlice.truth_value == label]
  return html.Div(
    children = [
      # html.I(json.dumps(hoverData)),
      # html.P(label),
      html.I('\"' + hoverSlice.reset_index().iloc[0]['quote'] +'\"'),
      html.Br(),
      html.Br(),
      html.Strong(f'- {author}')
    ]
  )

@app.callback(
  dd.Output('datatable_1', 'filter_query'),
  dd.Input('graph_1', 'clickData')
)
def graph_select(clickData):
  if clickData == None:
    return ''
  truthColumns = [
    'pants-fire', 
    'false', 
    'barely-true',
    'half-true',
    'mostly-true',
    'true'
  ]
  curveNumber = clickData['points'][0]['curveNumber']
  truthValue = truthColumns[curveNumber]
  author = clickData['points'][0]['y']
  return f'author = {author.replace(" ", " ")} && truth_value = {truthValue}'

# @app.callback(
#   [
#     dd.Output('node_graph', 'elements'),
#     dd.Output('node_graph', 'stylesheet')
#   ],
#   [
#     dd.Input('dropdown_2', 'value'),
#     dd.Input('button_1', 'n_clicks'),
#     dd.State('input_2', 'value'),
#     dd.State('input_3', 'value')
#   ]
# )
# def update_cytoscape(dropdownValue,n_clicks, positiveInput, negativeInput):
#   if positiveInput == None or positiveInput == '':
#     potitiveInput = []
#   else:
#     positiveInput = positiveInput.split(',')
#   if negativeInput == None or negativeInput == '':
#     negativeInput = []
#   else:
#     negativeInput = negativeInput.split(',')
#   wv = word2VecList[dropdownValue].wv
#   topNWords = wv.most_similar(positive = positiveInput, negative = negativeInput)
#   data = []
#   main = {
#     'data' : {'id' : 'main', 'label' : ' '.join(positiveInput)},
#     'position' : {'x': 0, 'y': 500}
#   }
#   data.append(main)
#   for element in enumerate(topNWords):
#     dataPoint = {
#       'data' : {'id': element[1][0], 'label' : element[1][0].title()},
#       'position' : {'x' : 1000, 'y': 100*element[0]}
#     }
#     edge = {'data' : {'source' : 'main', 'target' : element[1][0], 'label' : f"{str(round(10*element[1][1],3))}"}}

#     data.append(dataPoint)
#     data.append(edge)
#     stylesheet = [
#       {
#         'selector': 'edge',
#         'style': {'label': 'data(label)'}
#       },
#       {
#         'selector' : 'node',
#         'style': {'label' : 'data(label)'}
#       }
#     ]
#   return data, stylesheet

if __name__ == '__main__':
    app.run_server(debug=True)