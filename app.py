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

# external_stylesheets = ['./assets/app.css']

server = Flask('capstone-dash-app')

app = dash.Dash(__name__, server=server)

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

#############################################################################################################################################################
# begin the layout of the application
#############################################################################################################################################################

app.layout = html.Div(
  id = 'body_container',
  children=[
    dcc.Tabs(
      value = 'tab_1',
      children=[
        dcc.Tab(
          id = 'tab_1',
          label = 'Project Overview',
          value = 'tab_1',
          className='tab',
          children=[
            html.Div(
              children=[ 
                dcc.Markdown(
                  className = 'markdown',
                  children = [
                    '''
                      # Abstract

                      This paper is a first attempt at using neural network technology to determine the truth value of short, headline like, statements. The statements used in this paper are political in nature. While an examination of lies in politics was not my direct motivation, ultimately, many of the damaging mis-information campaigns are centered around the political scene. The sole data source for this project reflects this political leaning, and as such, I will refrain from drawing conclusions about more general statements. Since the proliferation of social media as a primary means of spreading misinformation, society, worldwide, has been reeling from the effects. Here in the United States, misinformation has been used as a tool for undermining our most fundamental democratic institutions and is quickly eroding the moral cohesion of local communities. Since the beginning of the mis-information plague, there have yet to be any widely used tools which can combat the disease with reasonable surety. Perhaps, this is due to the fact that a lie takes only a moment to create, but hours to fully debunk. To be clear, this paper is an exploration of neural networks and their ability to discern truth with varying degrees of information. At this point, the models are not performing well enough to be used as a digital vaccine to the larger problem. It is my hope that the work in this paper will be improved upon by others to create a lasting light in the dark sea of lies we have all been swimming in.
                      It's worth defining the use of misinformation as it will be used in this paper. Misinformation is the presentation of factually incorrect information that is portrayed as being true. Whether or not the source/speaker of the information believes what was said was correct, is irrelevant. Specifically, the data from PolitiFact with labels of "pants-fire", "false," and to some extent "barely-true" are what I consider to be the primary examples of misinformation.
                      At the end of this technical paper, you will understand much about the nature of the quotes we examined and of the modeling approaches we investigated. As it stands the final/best model in this paper is not ready for production. More research, data and approaches will be needed to solve - to any realistic degree anyways - the plague of misinformation. This isn't to say no progress has been made. To be clear, most models in this paper have shown the ability to classify the truth of statements significantly better than a random guess. The metrics of the models show the training process has picked up on the nature of truth to varying degrees.
                      This paper is for all interested in misinformation and how modeling might be used as a tool to combat its use. More specifically, I am reaching out to social media companies, journalists and people of the political world to examine the trends found in this paper. The points I wish to leave with this audience are the following:

                      1.	Combatting misinformation is in everyone's best interests.
                      2.	Special resource allocation and attention should be given to information that is being generated and distributed during the high interest events of the American election cycle.
                      3.	People and groups of people are likely to be the subjects of misinformation.
                      4.	Portions of the outrageously false category - pants fire - use language that is particularly violent. Most of these quotes were generated on social media and it is my belief that they represent malicious content.

                      CRISP-DM was my research methodology of choice. Now to the matter at hand.

                      # Business Understanding

                      Given the intent of my work is to act as a defense against misinformation on social media, we will focus on business case for companies like Facebook, Instagram, twitter and the like. Other expansions are desired in the end, but for the sake of keeping the problem reasonably narrow, text versions of social media will they be the target. What does a defense against misinformation look like? The answer is multifaceted. A general model is a great start, but not the end all be all. No model will be perfect, and thus it is equally important to understand what lies look like. Models are useful tools, but not the ultimate solution. Any understanding of the structure of lies will be essential in both, curtailing the pervasive spread and efficiently debunking them in near real time.

                      ## Why Care About Misinformation?

                      It's worth asking, why would social media companies care about verifying the truth value of content posed to their applications? Any normal person would understand that the answer involves doing the right thing. For those of you not convinced I'll continue on. How about money? Everyone likes money, right? While the total cost of misinformation is difficult to ascertain and hard to quantify, CHEQ AI, a cybersecurity company that uses AI to protect others from mis-information, commissioned a study run by professor Roberto Cavazos from the University of Baltimore, to quantify the cost of mis-information on businesses in the US and abroad. You can find the whole article here, but the general conclusions are the following.

                      *	37 billion in losses on the stock market
                      *	17 billion in losses annually from false/misleading financial information
                      *	9.54 billion spent on reputation management for false information about companies.
                      *	3 billion has been already spent to combat fake news.
                      *	400 million spent by US public institutions defending against misinformation.
                      *	250 million spent on brand safety measure.

                      All in all, the report concluded that 78 billion dollars is the total spent as a consequence of and combatting against misinformation in the recent past. This is just the direct costs that could be found. The price tag for indirect consequences is unknown and assumed to be many times greater than what can be directly attributed.

                      Not all of this applies directly to social media companies, but as the host for much of the fake news, their share of the burden is significant. In 2018, Facebook's CEO claimed that the company's annual budget for defending against misinformation was more than 3 billion dollars. This includes over 30000 positions world-wide which are dedicated content monitors. While I'm not a legal scholar, the relationship social media has to fake news isn't completely detached. Several governments world-wide have entered into agreements with social media companies to fight versions of misinformation that may be deemed bad for the public or governmental institutions. This relationship is costly for the companies and often required in order to conduct business around the world.

                      Given this is paper is being written at the end of 2020, I've reserved a specific shout out for COVID-19. Misinformation has taken a toll of human life in the US in the year following the outbreak of COVID-19. The ensuing social and commercial restrictions have negatively impacted hundreds of thousands of individuals and businesses world-wide. As the primary income for most social media platforms is advertising dollars, the overall downturn of the global economy due to the virus has hit the pockets of social media companies. The handling of the virus has been negatively affected by the dissemination of fake news, whereby prolonging the economic damage. As the pandemic is still raging and vaccines have only just begin being distributed, the total cost is still unknown. But there can be little doubt it will be a very large number. I'm interested to see how much of the total cost will be attributed to misinformation when the dust settles.

                      ## What is the Business Objective?

                      Having made the case why social media companies should - and currently do - care about misinformation, let's move on to the more important question. What is the business solution to this complex question? There are several different versions of what monitoring for misinformation can take on. The scale ranges from simple identification to more advanced methods of targeted censorship, moderation and my personal favorite, providing truth and context. Any case begins with the ability to determine the truth value of statements posted to social media. As such, the primary business objective is to create a way to effectively categorize user generated text as to the level of truth it contains in a cost-effective manner. Ideally, this will be done with some level of specificity.

                      No model is perfect, and I take personal care when thinking about integrating AI into a system that makes decisions about people. In the interest of staying away from the messy world of censorship, this paper will focus on finding the truth value of statements and shedding light on the connection between truth and language such that social media companies will be able to find specific text with strong correlations to specific truth categories.                  
                      
                      # Data Understanding

                      The data used for this paper was pulled from Politifact.com. Politifact is a lovely organization that takes the time to investigate viral claims that deal with hot or sensative topics. Unsurprisingly, many of these claims are political in nature, but not strictly so. Politifact investigates these claims and writes a small article summarizing their findings as to the validity of the content in question. They do the hard work! If you find this paper at all interesting it might be worth sending a little money their way. 5 bucks people, I'm not asking for the moon. Anyways, Politifact vets the claims and boils it down to six distinct categories: Pants on fire, false, barly-true, half-true and true. Pants fire is a special category reserved for falsehoods that not only lack any semblance of truth but are also particularly outlandish or detrimental. A good example would be, "The democrates say they are going to blow up the moon." The classifications are somewhat subjective but because professionals took the time to look into the truth and cite their sources, I hold the labels as the gold standard when it comes to verifying the level of truth in a statement. 
                      There's one problem with this approach. Research is hard and Politifact has only done 19000 evaluations over the past 12 years or so. Yes, that's a lot of investigations, but not when thinking about NLP neural networks which are often trained on bodies of work as large as all of wikipedia.
                      Because of the raw data limitation, I'm skeptical of the ability to classify truth to say +40% accuracy. Ultimately, the work done here will need to be supercharged with data in order to reach its true potential. For now, let's take a look at what we have. 
                      An important consideration ever present in modeling is class imbalance. Here's our given composition.

                      ![pie_chart](https://drive.google.com/uc?id=1EjxE9yDNB9mj2lFiUfnypbdTDUOcZMKC)

                      The imbalence is not as severe as it could be, but this still merits adjusting for. Sci-kit learn has a particular function which calculates the weights needed for each class in order to mimic balanced amounts of data. This is the primary tool I am going to use to address the imbalance, please see the code sections for specifics on the implementation. 
                      With imbalance mitigated we will take a detour into the data itself. Our dataset includes 18436 different records. These records have three text columns. Our target column is the truth_value column which is the basis of our supervised learning task for this paper. In addition, we have a column of quotes which are consumed by our model in order to make predictions on the truth value. The final column is the author column which routinely accounts for the person or organization responsible for the creation of the quote. 
                      What do these columns look like? The next visual represents the 10 most frequent authors and the breakdown of their quotes into the various truth values. 
                      
                      ![authors](https://drive.google.com/uc?id=1gL4xn0HS9aSG4sG4O38i6uVu8MV_Zqlh)

                      With the rise of social media, it makes sense that those at the top of the list are figures in the modern political scene. In general, Politifact reviews quotes with implications for the real world. With the prolification of misinformation as a political tool, much of their work has centered on uncovering the truth behind statements made in the political sphere. Please note that there are lumped categories such as "Facebook posts" and "Viral image". These categories include quotes that achieved viral status in spreading through social media sites. You won't find a good definition of viral in this paper as the exact criteria used to determine what elevates something to viral status isn't exactly clear. For a better sense of the nature of the quotes themselves, please see the hosted application for this paper. 
                      Given the top authors are pulled mostly from the political realm, it follows that words common in modern political times are highly represented. Check out the following two word clouds for a more intuitive idea about the nature of the quotes corpus. This example excludes stop words from consideration (the, and, where, in, is, I, etc...).
                      
                      ![word_cloud](https://drive.google.com/uc?id=1aJmMaPmbhdrM2VB7OGGISv7sRhe61na4)

                      Unsuprizingly, the word cloud that excludes stop words consists of what we might expect: Words about hot topic issues, words about the voting process, locations, individuals in politics, money, governmental institutions and the like. 
                      Summary statistics are also worth considering. While using the quotes to determine the truth value will ultimately be the job of our models, let's see if there's any differences we can spot that might seperate the classes at hand. 
                      
                      ![boxen_plot](https://drive.google.com/uc?id=1pMjZ6eUHw2rFrXjsCD-ip3LTxS9xVG7Q)

                      This confirms the majority of our quotes are short in length. Looking at the medain of the data - denoted by the black line in the middle of the boxes - there are some slight differences among the various truth value categories. The two falsehood categories pants-fire and false have the fewest words on average, approximately 14 and 15 words respectively. Mostly-true, barely-true and true all have nearly identical medians, that being a word or two more than the false categories. Half-true statements had the largest median at around 18 words. 
                      The overall spread of each category’s distributions is roughly similar or at least, aren't unique enough to give immediate insight into the relationship between quote word count and the underlying truth value. 
                      To get more advanced understanding of the text we reach to the standard tools of the NLP field. A first step into any investigation is tokenizing our quotes. Tokenization is the process of breaking up each quote into their individual words from which they can be assigned numerical IDs. I wove lemmatization into this process. Lemmatization is a term used to describe reverting words to their root form. For example, without doing this the words, run, runs and running would be considered different. Why would we want to combine these cases? All forms of run are inherantly denoting the idea of running. I personally think it makes more sense to consider each of these as a single case. In general, this is an accepted practice in the NLP world. 
                      With tokens in hand, we now have the ability to use a vast array of text processing models to gain insight into out quotes. The preprocessing models I used are the following: Bi-grams, Word2Vec embeddings, lda (latent dirichlet allocation) models, lime models and scatter text models. 
                      ## Bi-Grams
                      With our corpus broken up into individual words it's time to start disecting our truth value categories a little more carefully. A natural place to start is at the question of whether or not our categories have any characteristic language that sets them apart from the others. To do this I'm going to leverage another popular NLP technique: bi-grams. The idea here is simple, isolated words a little hard to get insight in without any context. A solution to this issue was to group words by those next to them and display those instead. In general, you can capture any number of adjacent word pairs, but for this paper bi-grams make the most sense to focus on considering the limited length of our quotes. Here's the top 25 pairs for our corpus.
                      ### Top 25 Word Pairs
                      1.  health care
                      1.  united states
                      2.  Donald trump
                      3.  Barack Obama
                      4.  Hillary Clinton
                      5.  President Barack
                      6.  President Obama
                      7.  social security
                      8.  says president.
                      9.  new york  
                      10. Scott walker  
                      11. joe Biden 
                      12. says Donald 
                      13. last year 
                      14. health insurance  
                      15. tax cut 
                      16. mitt Romney 
                      17. photos show 
                      18. Obama administration  
                      19. illegal immigrant
                      20. supreme court
                      21. says u.s
                      22. income tax
                      23. says Hillary
                      24. new jersey
                      Unsurprisingly, many of the most said word pairs are direct references to presidents and presidential hopefulls over the past 3 election cycles. Again, it's stark that the quotes have a large here say pattern, as the word "says" shows up in 4 of the top 25 pairs overall. This relationship gets more interesting if we break the paires into their respective originating truth values. Please note that not all Bi-gram options can be seen in the legend, it was shortened for the visual. 
                      
                      ![bigram](https://drive.google.com/uc?id=1wdpVxcJ3ZgdD1nLXGTYfae087IWZbQ02)

                      Here say mentions don't appear at all in the top 25 words pairs of true quotes and only once in the mostly-true quotes. This is contrasted with the false categories in which they appear 6 times in the outlandish lie category and 4 times in the false category.  **This lends itself to the conclusion that false statements are more likely to include here say mentions**. 
                      If we expand this idea to also include mentions of specific persons, we see that this trend holds.  
                      
                      ![bigram](https://drive.google.com/uc?id=1dwT-aoi7Vxe1K6YZaFNlOMV32jkcefgz)

                      Here say and direct mentions account for 18 of 25 of the most commen word pairs for the outlandish category and 13 of the false category. Contrasting this to the high-truth categories, we see 6 of 25 for true quotes and 7 of 25. In fact, this thrend holds for all categories, steadily increasing in representation as we decrease the relative truth in the quotes. Again, extrapolating, ** direct mentions of a specific author are more likely to be found in false statement when compared to true statements **. 
                      The other common thread between word pairs relates to US political issues: The economy, taxes, health care and the like. 
                      
                      ![bigram](https://drive.google.com/uc?id=16hUwplkuw82L8aFhhsgkUjJyEelwQLQ1)

                      We won't go through them all individually, but here are the ones with the clearest relationships. Health care is on everyone's mind, ranking in the top 3 for every truth category. This makes intuitive sense considering the constant debate surrounding the Affordable Care Act and more recent calls for universal health care. What's interesting about health care is the level of representation. Going back to the most common word pairs from all quotes, health care ranked number 1. The next social issue to show up is social security at number 8. ** This suggests American political discourse is highly focused on health care when comparied to other "hot topic" social issues**.
                      A good comparison for this are phrases that have the word "tax". The word pair "tax cut" and "income tax" are the only mentions of taxes to make the top 25 list overall. Drilling into the categories, "tax cut" is also the only mention of tax to break into the top 10 in any truth category, coming in at number 9 under mostly-true. Some pairs such as "tax break" and "raise taxes" are more skewed to the half-truth and less truth categories. "Income Tax" breaks the top 25 in the true categories whereas "property tax" only shows up in the barely-true category. 
                      It's also somewhat remarkable that word pairs relating to abortion, birth control and womens reproductive rights don't make the top 25 overall or in any distinct category. I expanded my search, and the first mention of this type is "planned parenthood" which is rank 44 in the barely-true category. 
                      Surveying the word pairs, I think it's fair to say **that people and not subjects - with the notable exception of "health care" - are the most common thread of all our quotes**. 
                      ## Lime Models
                      Lime is a wonderful open-source tool that seeks to shed light on the black box nature of the machine learning. How does it do this? Linear approximation of course! The complexity of machine leanring decision boundries is vast and typically highly non-linear. Any attempt to have a global linear approach is destined to fail. That's why Lime focuses on individual cases (small neighborhoods if like like the math terms). When shrinking the area of the decision boundary down to a single case, the ability of linear functions to approximate thest boundaries becomes not only possible, but a reasonable estimate.
                      As a linear model, each word in the text is given a coefficient, thus we have feature importance! Here are some examples. Two in which the model guesses correctly and two in which the model guesses incorrectly. A preface to these images. As the lime model is linear at the core, each word has a coefficient that impacts classifications in a positive or negative way. 
                      ### Correct Classification
                      ![lime] (https://drive.google.com/uc?id=1Xz_jbM12KHjVfdZHD8dERDsyP0MRhd6r)
                      ### Correct Classification
                      ![lime] (https://drive.google.com/uc?id=1O__S7uRgdkO1IgV0Db2HGinGi5-Rv0yc)
                      ### Incorrect Classification
                      ![lime](https://drive.google.com/uc?id=1dzAHN4o91TKM4fpIpX1BvW2sG12ZxiA4)
                      ### Incorrect Classification
                      ![lime](https://drive.google.com/uc?id=1sQGt-rCpe_mpxulhhh1BOAEeJJ5fR8iq)
                      ## Word2vec Models
                      Word2Vec is a modeling technique that creates an embedding space for a corpus through the training of a special kind of neural network. Lda models use word frequncies to create seperate topics based by inspecting what I call characteristic words. By this I mean the words that are, for the most part, unique to a topic. The interesting thing about LDA models is the number of topics is treated as a hyperparameter. If we're lucky, we can create as many topics as there are differnt truth values and then inspect would the LDA topic clusters relate to our truth categories. The last, and somewhat less useful model is a scattertext model. Scattertext, is most useful for binary classifications. For our case, this will force me to collapse all the truth values into two categories. Scatter text then treats these two topics as principle axses and plots the individual words from our corpus in relation to these two topics. 
                      The cool part about creating embeddings is each word in our corpus is transformed into a vector in the embedding space. With vectors comes the ability to preform certain mathematical operations such as addtion and subtraction (or any linear combination of vectors if you are coming from the math side). But what does it mean to preform addition between two words? Because of how word2vec derives the embedding space, the different dimensions in the space are thought of as the underlying semantic patterns within the text. The classic example that demonstrates this concept are the relationships between king, man, queen and woman. In most general text corpa, if we were to have an embedding and subtracted the word "man" from the word "king" then the word "queen" tends to be one of the closes word vectors to the result of the subtraction. Conversely, queen - woman will likely turn up king. Because the quality of the embedding will affect our model's ability to learn, let's checkout some word math examples to determine if the word to vec model is capturing ideas correctly.
                      * Example: health + care

                        1. ('act', 0.9708)
                        2. ('preexisting', 0.9689)
                        3. ('affordable', 0.9682)
                        4. ('pre-existing', 0.9636)
                        5. ("'deny", 0.9624)
                        6. ('coverage', 0.9613)
                        7. ("'lowers", 0.9613)
                        8. ('denying', 0.9585)
                        9. ('adequate', 0.9584)
                        10. ('ration', 0.9471)

                      The previous list shows the word most associated to the added vector of 'health' + 'care'. The numbers beside the words are represent their rankings (this is cosine similarity) with a maximum score of 1. For this one example, we can see the model has learned the concept that health care is a policy related term in our corpus. There are lots of fun word math examples. If you would like to play with the language relationships more, please head over to the associated application for this paper found [here](http//:). 
                      word2vec is an interesting tool to use for garnering a basic understanding of the corpus, but it is much more difficult to examine how the embedding vectors relate to truth in a statement. For more insights we move on to the other two models. 
                      
                      ## LDA Models
                      The first thing you need to know about LDA models is they create clusters. Unsupervised-Learning. While their usage can vary, we will be attempting to inspect our clusters for insight into our text by attemptint to label our clusters. Head over to the <a src = 'https://capstone-dash-app.herokuapp.com/'> app </a> and click on the LDA Model tab to checkout the models for yourself. 
                      When inspecting clustering algorithms there's always some subjective portion to the analysis. There's some debate when attempting to assign labels and this is made more fluid by the nature that I choose how many clusters to create. I thought it appropriate to create 6 clusters for several reasons. One, is the hail mary that the clusters will represent the truth values in some way (surprise, this is not what happens). The second, was to have enough clusters to get reasonable seperation without having too many to handle. In each section there will be a word cloud comprised of only the words for that category. But remember, when I'm talking about specific words, they may not be visible in the word clouds themselves. The LDA models use a more sophistocated method for finding "important" words conpared to word clouds which look only for the most common words. I don't want any emails saying, "You said word x was crucial to understanding the true category but it's not in the word cloud!" (Like anyone in the world is ever going to read this paper anyway ;).
                      For brevity, I won't be going through every cluster for every truth value and will opt to share some of the highlights instead. 
                      
                      ### All Quotes
                      ![word_cloud](https://drive.google.com/uc?id=1IfV8sljPmENLv0g61TXGx-igi7n04pLk)
                      
                      Considering all the quotes together was by far the hardest group to assign labels to. Topics were diverse in nature causing me to consider broader labels. Surprisingly, the labels were, in my estimation, heavily skewed to more politically and socially conservative ideas. If you follow along in the app here are my labels for the clusters as they are labeled. 
                      
                      * Cluster 1: Conservative fiscal issues and political members
                      * Cluster 2: The Political Campaign Trail and Topics
                      * Cluster 3: Bible Belt / Conservative Social Issues
                      * Cluster 4: Obamacare and Controversies
                      * Cluster 5: The Federal Deficit and Spending Trends
                      * Cluster 6: Abortion and Women’s Health

                      I don't have an overall distribution for political party contributions to the quotes at large which makes explaining the conservative tilt to the clusters impossible at this time. Given the strong representation of health care in the bi-grams examination it's not surprising that Obamacare ended up as its own category. The other categories are also central American debate topics in the modern political scene with the exception of cluster 2. In the case of the second cluster, there were so many buzz words from various topics that I got the impression of a chaotic political rally.
                      The all category is hard to draw insightful conclusions from. Its primary use here is as a comparison for the individual truth vlues. 
                      
                      ### Pants-Fire

                      Let's start with the lies! 
                      
                      ![word_cloud](https://drive.google.com/uc?id=1EN5V56le_JgTqM7FKirIS5NvxiIMCoQd)
                      
                      As confirmed by the Bi-Gram analysis, mentions of specific people are much more prevalent in the false categories. This outlandish lie category has very intriguing categories that don't line up with the nature of the other truth value clusters. Here's what I found:
                    
                      * Cluster 1: Political advertisements / Campaign attack language
                      * Cluster 2: Politicians & political jockeying
                      * Cluster 3: 2008 Republican/Sarah Palin platform topics
                      * Cluster 4: Hillary Clinton & Health Care
                      * Cluster 5: Fear & Violence & Death Language
                      * Cluster 6: The Election Cycle

                      In particular, Clusters 5 and 6 I found to be the most interesting. Cluster 5 was the only cluster in this category that didn't appear to have a sensable narrative. For example, when reading cluster 4, we have words such as "hillary", "clinton", "health", "care", "takeover", "illegal", "mandate". When reading these words without context it takes me back to Hillary Clinton's presidential campaign and the attack ads against her and her health care initiatives. The words told a skelenton like story that evoked my memories of a specific time period. Cluster 5 was more pandemonium. Here are some of the prominent words: "rape", "victim", "crime", "threatened", "corperate", "prevention", "cancer", "hate", "protest," "administration", "fake", "removed", "violent", "pedophile", "muslim". It sends quite a message.
                      I should be clear that "obama", "trump", "congressmen", "mccain," do show up in as part of these quotes. What makes their appearance different in my mind is the weak representation when compared to other clusters. Almost all polarizing political figures show in all clusters in all truth categories. And in this case, none of the aforementioned politicians had an abnormally large proportion of mentions in this cluster specifically. Thus, I don't see people being the direct subject of the language in this cluster. ** I suggest that cluster 5 may be one piece of the deliberate mis-information campaign at large**. 
                      At the other end of the story telling spectrum is cluster 6. When reading through the language of cluster 6, I will filled an entire story of a US election cycle. The langeuage is direct and speaks to both the process and our facination with the process. "amnesty", "bible", "campaign", "income", "immigrant", "asked", "broke", "drug", "stimulus", "november", "jobs", "changed", "trade", "religion", "reporter", "endorsed". It's like the news reals from September through November 2nd vomitted on the page. ** Because of the strong tie between blatently false statements and the US election cycle, I find it worth recommending that extra reasources be invested as these times approach. ** 
                      
                      ### False
                      
                      * Cluster 1: Biden vs Trump Campaigns
                      * Cluster 2: Trump & Texas border crisis
                      * Cluster 3: Hillary Clinton Presidential Campaign
                      * Cluster 4: Health Care & Energy Policy
                      * Cluster 5: Obama vs McCain
                      * Cluster 6: Obama's first term Issues

                      The first striking thing about the false labels is the correlation with the election cycle idea in that most categories are related to specific elections. We also see the second appearance of health care - spoilers, it's going to be a motif. Another point worth mentioning is the lack of generally virulent language compared to the pants-fire cluster 5. 
                      
                      ![word_cloud](https://drive.google.com/uc?id=1qMnw_91BpeZbAbSOybpR5eqF630_5zZ9)
                      
                      ### Barely-True
                      * Cluster 1: Health Care & Energy Policy
                      * Cluster 2: Fiscal conservatives & Obama spending policies
                      * Cluster 3: Obama economic policy & Hillary Foreign Policy
                      * Cluster 4: 2016 republican primary candidates & issues
                      * Cluster 5: Obamacare as relates to immigration/fraud/employment
                      * Cluster 6: Mitt Romney's presidential platform

                      Again, we have health care and campaigns/elections. Other than the continued pattern, there isn't much new seen in these clusters. 
                      
                      ![word_cloud](https://drive.google.com/uc?id=1T1gZUtkI2FRBA17FadFVInd6t9TeM7-l)
                      
                      ### Half-True
                      
                      * Cluster 1: Taxes & financial policy
                      * Cluster 2: Obama administration financial policy
                      * Cluster 3: 2016 election questions and participants
                      * Cluster 4: 2012 Romney presidential platform 
                      * Cluster 5: 2016 Hillary presidential platform
                      * Cluster 6: Abortion/Guns/Wall street regulation debate

                      Similar veins of the previous truth values. 
                      
                      ![word_cloud](https://drive.google.com/uc?id=1e2HcXzMMjNG6Mml-WmRAE3N9oq_Y35zM)
                      
                      ### Mostly-True
                      
                      * Cluster 1: Health Care & Education Funding
                      * Cluster 2: School Violence & Crime at large
                      * Cluster 3: American wealth distribution
                      * Cluster 4: Republican talking points and policies
                      * Cluster 5: Obama Presidency policies and debate
                      * Cluster 6: John McCain presidential campaign
                      The issues remain somewhat the same but there is a slow slide back in time. Many of the words in this truth value are related more to the 2008 and 2012 presidential elections and major policies. I believe this is indicative of the increased pervassiveness of misinformation as technology increases in power. This underpins the need to increase resources devoted to misinformation as time continues.
                      
                      ![word_cloud](https://drive.google.com/uc?id=12KtJfbuT_Aj2SSANAzbmV2HLaGoYJ9qQ)
                      
                      ### True
                      
                      * Cluster 1: Government stimulus & spending
                      * Cluster 2: Obamacare court battle
                      * Cluster 3: American crime and military debate
                      * Cluster 4: General society issues & debate
                      * Cluster 5: Financial policies and political funding
                      * Cluster 6: Conservative financial/energy/military platform
                      
                      In the truth category, the volumn of mentions of individuals has fallen off when compared to the false categories. Other than this, the language isn't too dissimilar from the other non-false categories. 
                      
                      ![word_cloud](https://drive.google.com/uc?id=1UMrKY-C9jPIJHUdXDS9ZOkp3yzL874OU)
                      
                      ### Summary
                      
                      With the exception of pants-fire category, all categories have a similar feel to the clusters in that they are focused on elections and politically hot topics. Health care, finance, energy, millitary and women's health are recurring motifs. There is no obvious way to associate a particular topic with a truth value. 
                      The most insightful discovery is cluster 5 of the pants-fire category. I was struck by the chaotic and violent language used in these quotes. I am highly convinced that this cluster is a representaion of misinformation being injected into american politics. Searching for these terms in our body of quotes reveals that facebook posts, bloggers and viral images are responsible for a substantial percentage. Here are some basic stats:
                      
                      1. 8 of 10 pants-fire quotes with "fake"
                      2. 8 of 19 pants-fire quotes with "crime" 
                      3. 7 of 10 pants-fire quotes with "hate"
                      4. 29 of 43 pants-fire quotes with "Muslim"
                      5. 8 of 19 pants-fire quotes with "victim"
                      6. 12 of 23 pants-fire quotes with "death"
                      
                      These are some of the words that stood out to me. It should also be noted that I'm only including the annonymous internet sources, but that other un-reputable websites also make up these categories. I'm focusing on these sources because of the proven connection between social media accounts and deliberate misinformation. There's no realistic way to assign direct fault in any individual case, but it shouldn't be ignored that sources of malicious misinformation are the primary contributors to the most damaging of clusters found in the pants-fire categories. 
                      
                      ## Scattertext
                      
                      Scattertext necessatates that we reduce our classification down to the binary case. I think it best to inspect two different label collapses. One: False vs non-false. Two: True vs non-true. With two breakouts, we can inspect the black and white categories as compared to everything else. Ultimately, there is use in inspecting all categories indiviually, but we will leave this to future research. 
                      
                      Click [this](https://drive.google.com/uc?id=1F3UGG8kYhhgaUDMa0uYfb5g7azZcCws0) link to interact with the scattertext visual.

                      This concludes our exploration into the dataset. Because the purpose of this paper isn't to analyze the impact of the author on our model's ability to determine truth values, we will omit further investigation into that column. Ultimately, a holistic approach to determining the underlying truth of posts on media can and should include all available information available to make classifications. I will consider adding more information in an ending section if time allows. 
                    '''
                  ],
                ), 
              ],
              className='narrative_container',
            ),
          ],
        ),
        dcc.Tab(
          id = 'tab_2',
          label = 'Authors & Quotes',
          value = 'tab_2',
          className='tab',
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
          id = 'tab_5',
          label = 'Modeling',
          value = 'tab_5',
          className = 'tab',
          children = [
            dcc.Markdown(
              id = 'modeling_info',
              className = 'markdown',
              children = [
                '''
                  # Baseline Model
                  The baseilne model is used to establish a baseline for future model iterations. In some cases we might introduce a random guessing models as the baseline, however, in this case, given the complex nature of NLP models, I believe it makes more sense begin with a naive bayes classifer. Naive bayes was traditionally used for high dimensional data, NLP being the foremost example. I would like to note that much of the text preprocessing already done is primarily to support this baseline. For neural network models, much of the preprocessing is left out in order to allow the network to ascertain which features are most important rather than feature engineering using TF-IDF and n-grams. Here is the test set confustion matricies for the four different text preprocessing methods we used.

                  ## Vanilla Tokens Model

                  ![confusion_matrix](https://drive.google.com/uc?id=17qKDX2UMRffb_fPMPCzGH6wiTox2czPt)

                  If you're not familiar with confusion matricies let give you a brief explanation. The vertical axis is the true category label for our testing data. The horizontal axis are the labels that our model predicted for the same data. Rather than showing the raw numbers of where our predictions landed, we give the percentages of the true labels. If you add up each decimal in a row you will find that they add to one. You might already see that a perfect model with 100% percent accuracy would have ones down the main diagonol. Clearly, this model is nowhere near perfect. The text fed into this first model only had the punctuation stripped, without additional alterations. The model has a very strong bias towards the pants-fire column. Well at least it can reasonable identify the crazy statements! Having a stong bias isn't generally good, and it isn't in this case. The model has also mis-classified a large percentage of all the other labels as being the pant-fire label. More specifically, I see almost total confusion for all other labels, as the model gives a roughly random guess for any label it doesn't believe to be a member of the pants-fire column. If anyone cares, the model have an overall accuracy of 16.5%. With much to be desired, let's move on.  

                  ## TF-IDF Model
                  ![confusion_matrix](https://drive.google.com/uc?id=1PraywdIC1sHhuaE4Mb_5EHI8Hmx7xBLP)

                  TF-IDF preprocessing shows a vast improvement over the first case. We begin to see some logical connections that give hope for the ultimate answer to the question of telling truth from lies. The overall accuracy is 25%. This is a 50% increase in performance from a random guess! You might still be holding your breath at this point so let me continue. Looking at our confustion matrix, observe that the most prevalent incorrect predictions are the logical neighbors of the true label class. For example. The pants-fire class was predicted correctly 40% of the time. The model then went on to incorrectly classify false labels as pants-fire 23% of the time. While false and pants-fire are seperate categories in this case, they are both lies in the grand sense of things and thus it makes perfect sense that they could be mistaken for each other. Conversely, the categoryies that mis-classified the least were the true and mostly-true categories. I find this training to be conclusive proof that in the small subsect of quotes we are analyzing, machine learning models are capable of understanding the difference between true and false on some level. This is a great starting point going into the rest of the paper.  


                  ## Lemmatized Model
                  ![confusion_matrix](https://drive.google.com/uc?id=1r8b1N-TL3kr48WObyin9U8pP7yby9dtl)


                  ## Lemmatized TF-IDF Model
                  ![confusion_matrix](https://drive.google.com/uc?id=1TRzAYUS0yRRPXyjme1MLG7ig6P9-ida9)

                  I'll be brief for the last two preprocessing models. In both cases, the only difference was applying NLTK lemmetazation. For the naive bayes model, lemmatezation appeared to degrade the overall performaces. This may suggest that word tense has a role to play in the naive bayes case. The TF-IDF case is particularly odd considering the process of lemmatization has also appeared to cause the model to make almost opposite associations between true and false. 

                  All in all the best of the baseline models has shown some motivation for chasing a truth detector. However, there is still a long way to go. If you would like to inspect any of the code used to create the models go ahead and expand the following cells. Also, some executions will create the models in question and save them to folders if you wish to use the models themselves for another purpose. 
                
                  # LSTM Neural Network
                  With Naive Bayes showing some promise as a potential modeling solution, we turn to neural networks and deep NLP in hopes that we can super charge the trend. This section of neural networks will focus on Long/Short-Term Memory models. 

                  LSTM, until recently, were on the cutting edge of deep NLP breakthroughs. The underlying premise is to create internal states for the nodes of model, such that, they can keep track of words and their associated embeddings. In this specific model we will use a bi-directional layer of LSTM cells. The idea to state is really another way of saying time dependence. What does time have to do with text data? The order or reading! Text doesn't make sense if we choose a bunch of words in random succession. There are complex ideas embedded within the context surrounding each word, and an LSTM attempts for keep track of this information by adjusting its weights at the first word in a sequence and feeding the new configuration back into itself for every new word. This way, the model remembers (somewhat) the imporatnce of the previous words in the phrase. 

                  LSTMs do have a couple issues. One is known as saturation. There are only so many weights an LSTM can adjust as each successive step through the text data. Take the example of having 100,000 weights that the layer can adjust. This seems moere than adequite for being able to learn for a couple words. What about 1000 words? What about 100,000? One weight per word doesn't seem sufficient. Saturation is actually a well understood problem that arose in predecessors to the LSTM. The LSTM has internal functions that attempt to conpensate for this issue by allowing the model to forget words that don't seem very impactful. While a great idea, the implementation is far from perfect and from the AI the industry's progress in the interim, this is known to be an incomplete solution.

                  The other issue with LSTMs is their limitation on computation. With an internal struture that is meant to mimic time dependance, LSTMs have their order of opperations fixed, thus making parallel computaiton impossible. While not the most relevant issue in this paper, it does have significant drawbacks at scale. 

                  So, how did the LSTM models do?!

                  The results were underwhelming. 

                  ## LSTM with vanilla tokens

                  The vanilla tokens LSTM is currently leading the pack with 26.1% accuracy. Looking at the confusion matrix, there are some good and bad trends. On the positive side, the model appears to understand the pants-fire at a high level, whereby it correctly classified 58% of quotes with that tag. In addition, 30% of the false category was mis-classified as pants-fire but this is still encouraging considering they are logical neighbors. Something not quite obvious were those entries predicted as true. While the overall percentages are not as encouraging as the pants-fire category, the model is making reasonable connections betweent the true category and its neighbor classes, while shunning the false quptes. 

                  The main eye sore for this iteration is the barely-true category. The model was clearly confused by this label and treated it as the catch all for guessing. A plurality of the incorrectly classified quptes in the barely-true category came from the false class. The false class - the largest numbers wise - was almost never guessed.

                  In conclusion, the LSTM hasn't met expecations for a neural network. There could be several reasons for this ranging from the architecture to the data preprocessing methods. It is my hope that these scores can be improved upon in a second version of this paper. 

                  For the training process itself, the network appeared to continue learning for 5 epochs, after which it began to degrade rapidly. 

                  ![confusion_matrix](https://drive.google.com/uc?id=1O99Gzk2ACc1_HaQCbeaWiW47ZnPYExZ8)

                  ![raining_curves](https://drive.google.com/uc?id=1ESM7-V6UupxV1pXfw2gFsowL6joAqmtV)

                  ## LSTM with TF-IDF

                  Due to the incredible amount of time spent training the TF-IDF model - which has 20000 dimnensions - only the non-lemmatized tokens were used. This desicion was made based on the poor performace of this model. Interestingly enough, the TF-IDF model used the true and pants-fire categories as the catch all columns compared to the vanilla tokens model which showed a propensity for guessing barely-true. Ultimately, I don't believe TF-IDF is an appropriate vectorization method for neural networks given the incredibly large feature dimensions. Moreover, I will attribute the different prediction distribution to a degredation in the model. This iteration has lost the relationship between the false and pants-fire categories, which is pretty fundamental. Let's move on. 

                  ![confusion_matrix](https://drive.google.com/uc?id=1eNo2-Bxdi_Ro7lB2B7NvYOVmnix60xuj)

                  ![training_curves](https://drive.google.com/uc?id=1Yr17WtU8N4nSQ2SGwSyNt--d9St81amV)

                  ## LSTM with lemmatized tokens

                  This was the worst of the LSTM models. The confusion was high for most columns and accuracy was poor > 20%. The model does have some of the logical associations we've talked about in the baseline model but overall, the quality of these association has degraded across the board. I justify this statement with the lack of distinction between the correct predictions and every other category. In the baseline models, we saw a clear distinction between the correct predictions and predictions on the other side of the categorical spectrum. The lemmatized model has closed this gap to an unacceptable margin. 

                  ![confusion_matrix](https://drive.google.com/uc?id=1WJDk3kCW5WU_JEzHEAQ6CIe96njgYuac)
                  ![training_curves](https://drive.google.com/uc?id=1_-JV8kmfLE7tP0HXdmJBJr2cunb2vQo9)

                  ## LSTM with word2vec embedding and vanilla tokens

                  ![confusion_matrix](https://drive.google.com/uc?id=1gA7dfYwZWigFq-wZrqYUdkF7ces-CqGP)
                  ![training_curves](https://drive.google.com/uc?id=1wtY89UAx5eIRVVlOHiWZmvDfeyLiHzTv)

                  ## LSTM with word2vec embedding and lemmatized tokens

                  My final hope for the LSTM models was quickly dashed against the rocks of failure. This LSTM used the word2vec model mentioned in the Data Exploration section as a seed for word embedding. The idea here was the model could use the leared semantic relationship information from word2vec to process and classify quotes more effectively. While not the worst result in the LSTM section, the final result fell short of expectations in a large way. Something of particular note are the training curves which demonstrate a poor learning process in general. This LSTM only shows 2 to 3 epochs of learning before a plateau becomes visible. This is much shorter than should be expected and is worth further examination in the future. 

                  ![confusion_matrix](https://drive.google.com/uc?id=1uXM-fXgsnWVCztZryZy04bHMJPAsaItx)
                  ![training_curves](https://drive.google.com/uc?id=1Awka80moM65iwKlguRa6ii7bSTJ-cDIi)

                  In conclusion, the LSTM models preformed poorly on the whole when compared to the baseline case and are not production worthy. Moreover, getting data to increase the model's preformace is going to be somewhat difficult. There are other sources of truth vetting that were not considered in this project. A future iteration of this project may attempt to bring this data in, but it is worth acknowledging the hurdel exists, especially since labels from other data likely won't match up to Politifact. 

                  # Multi-head Attention Transformer
                  Attention is a hot word in the deep learning world at the moment. With the publication of "Attention is all you need" by (Vaswani et al 2017), attention mechanisms have been heavily studied in a wide variety of deep learning fields. More recently, the notion of self-attention arose and offered an interesting alternative to LSTM and other memory state models. Self-attention achieved state of the art results on NLP staple challenges with the BERT implementaion of self-attention by (Devlin et al 2019). 

                  So, what is it? Without getting into the underlying structure (which is slightly cumbersome) attention takes the core idea of a model having memory and abstracts it away from time dependence. So, what does this mean? The whole idea behind the adding a time dependence to NLP was an attempt to capture context in speech. Because we read in a single direction, time was thought to be the crucial component. And it is, in a sort of way. But the issue with a directional approach to modeling is that context isn't always built by adjacent words. Take the sentance, "I am going to my friend's house with my cat." If I (as an NLP noob!) were to boil this sentence down to a skeleton frame I may do something like this: 

                  I/cat go friend house.

                  As you can see, there are several filler words that make the sentance flow in a grammatically correct manner that aren't exactly necessary to understanding what's going on. Now if you remember, LSTM has an internal mechanism to deal with this, but even with losing the context from some of the filler words, the remaining sequence of words may still be of considerable length. Attention improves upon this problem by abandoning the directional approach and instead creates a large relational matrix of dimension N x N where N is the number of unique words in the vocabulary. This way, we can store each word's association with every other word all at the same time without having to worry about the order. If you're reading this paper then you probably know that operations with large matricies is computationally expensive, and this is the case with attention mechanisms as well. There's good news though! Because we've removed the time requirement for computing words associations, creating this matrix can be done with parallel processing! If you're waiting for the downside you'll need to wait, as almost all the top benchmark models in the NLP (and most other AI fields) run on the attention mechanism.

                  Can the average data scientist use this cutting-edge model without too many hoops? Yes, yes, we can. Fortunately, Keras has adopted a self-attention transformer class that is available [here](https://keras.io/examples/nlp/text_classification_with_transformer). Below is the keras implementation with slight adjustments and connected layers added on top. 

                  ## Multi-head Attention with Build in Keras Preprocessing

                  Our first candidate is the the purest form of the open source Keras adaptaion. In this case we are ditching our preprocessing methods for what is known as a token and positioning embedding. This form of embedding preforms a simple label encodded tokenizer on each quote. The code then tracks both the words and their positions within every quote and uses this information to create the embedding. The main difference between this process and something like word2vec is the lack of semantic information. This model will be used as an additional baseline layer of sorts to see how our changes affect the initial keras implementation. 

                  # Combining Labels
                  Our first run through the modeling process has yielded interesting results as the neural networks are creating appropriate associations between the various truth values. While the overall accuracy isn't what we would like to see, it also isn't entirely necessary that we break the truth into so many categories. We will go through a couple iterations of collapsing out existing categories in logical ways to see if a coarser assignment might create a more useful classification model. 

                  # Conclusion
                  Need to wrap everything up at this point. 

                '''
              ],

            )
          ]
        ),
        dcc.Tab(
          id = 'tab_3',
          label = 'Word2Vec Embedding',
          value = 'tab_3',
          className='tab',
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
          id = 'tab_4',
          label = 'LDA Cluster Models',
          value = 'tab_4',
          className='tab',
          children=[
            html.Div(
              children = [
                html.H1(
                  children = ["All Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_frame',
                  src = 'assets/lda_model_all.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window'
                )
              ],
              className='lda_container'
            ),
            html.Div(
              children = [
                html.H1(
                  children=["True Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_true_frame',
                  src = 'assets/lda_model_true.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                ),
              ],
              className = 'lda_container'
            ),
            html.Div(
              children = [
                html.H1(
                  children = ["Mostly-True Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_mostly_true_frame',
                  src = 'assets/lda_model_mostly-true.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                ),
              ],
              className = 'lda_container'
            ),
            html.Div(
              children = [
                html.H1(
                  children=["Half-True Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_half_true_frame',
                  src = 'assets/lda_model_half-true.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                ),
              ],
              className = 'lda_model'
            ),
            html.Div(
              children = [
                html.H1(
                  children=["Barely-True Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_barely_true_frame',
                  src = 'assets/lda_model_barely-true.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                ),
              ],
              className = 'lda_container'
            ),
            html.Div(
              children = [
                html.H1(
                  children=["False Quotes"],
                  className='tab_4_title'
                ),
                html.Iframe(
                  id = 'lda_false_frame',
                  src = 'assets/lda_model_false.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                ),
              ],
              className = 'lda_container'
            ),
            html.Div(
              children = [
                html.H1(
                  children=["Pants-Fire Quotes"],
                  className='tab_4_title',

                ),
                html.Iframe(
                  id = 'lda_pants_fire_frame',
                  src = 'assets/lda_model_pants-fire.html',
                  height = '1000px',
                  width = '80%',
                  className='lda_window',
                )
              ],
              className='lda_container'
            ),
          ]
        ),
      ]
    ),  
  ],
  className='app')

##############################################################################################################################################
# CALLBACK SECTION
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