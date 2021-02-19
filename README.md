# Modeling Truth: Can Neural Networks Sniff Out Lies

## Author: Michael Mahoney

### Email: michaeljosephmahoney@gmail.com

#### Git-Hub Repo: https://github.com/minthammock/cap-stone

#### Project Application: https://capstone-dash-app.herokuapp.com/

# Abstract

This paper is a first attempt at using neural network technology to determine the truth value of short, headline like, statements. The statements used in this paper are political in nature. While an examination of lies in politics was not my direct motivation, ultimately, many of the damaging mis-information campaigns are centered around the political scene. The sole data source for this project reflects this political leaning, and as such, I will refrain from drawing conclusions about more general statements. 

Since the proliferation of social media as a primary means of spreading misinformation, society, worldwide, has been reeling from the effects. Here in the United States, misinformation has been used as a tool for undermining our most fundamental democratic institutions and is quickly eroding the moral cohesion of local communities. Since the beginning of the mis-information plague, there have yet to be any widely used tools which can combat the disease with reasonable surety. Perhaps, this is due to the fact that a lie takes only a moment to create, but hours to fully debunk. To be clear, this paper is an exploration of neural networks and their ability to discern truth with varying degrees of information. At this point, the models are not performing well enough to be used as a digital vaccine to the larger problem. It is my hope that the work in this paper will be improved upon by others to create a lasting light in the dark sea of lies we have all been swimming in.

It's worth defining the use of misinformation as it will be used in this paper. Misinformation is the presentation of factually incorrect information that is portrayed as being true. Whether or not the source/speaker of the information believes what was said was correct, is irrelevant. Specifically, the data from PolitiFact with labels of "pants-fire", "false," and to some extent "barely-true" are what I consider to be the primary examples of misinformation.
At the end of this technical paper, you will understand much about the nature of the quotes we examined and of the modeling approaches we investigated. As it stands the final/best model in this paper is not ready for production. More research, data and approaches will be needed to solve - to any realistic degree anyways - the plague of misinformation. This isn't to say no progress has been made. To be clear, most models in this paper have shown the ability to classify the truth of statements significantly better than a random guess. The metrics of the models show the training process has picked up on the nature of truth to varying degrees.

This paper is for all interested in misinformation and how modeling might be used as a tool to combat its use. More specifically, I am reaching out to social media companies, journalists and people of the political world to examine the trends found in this paper. The points I wish to leave with this audience are the following:

1. Combatting misinformation is in everyone's best interests.
2. In our dataset, completely false information is easily identified when compared to true information. 
2. Special resource allocation and attention should be given to information that is being generated and distributed during the high interest events of the American election cycle.
3. People and groups of people are likely to be the subjects of misinformation.
4. Portions of the outrageously false category - pants fire - use language that is particularly violent. Most of these quotes were generated on social media and it is my belief that they represent malicious content.

CRISP-DM was my research methodology of choice. Now to the matter at hand.

## Recommendations

Despite the lack of the production worthy model there are several action items for various stake holders. 
1. For those wishing to devote resources to fight misinformation; virtually all clusters of the false quotes center around presidential elections and candidates. Reasorces should be ramped up and deployed at times of heightened contention during presidential elections and primaries. 
2. For politicians, most misinformation is based around people and not issues (with the notable exception of health care). If you wish to advance legislation, decoupling the issue from individuals will the best way to limit misinformation's impact.
3. For Journalists and people at large: Email chains, social media posts and viral content should never be used as a source for content.

# Business Understanding

Given the intent of my work is to act as a defense against misinformation I believe everyone is a stakeholder in this paper. With the focus of our dataset being on Politics, the primary interested parties are likely social media companies, journalists and those who work in politics.

What does a defense against misinformation look like? The answer is multifaceted. A general model is a great start, but not the end all be all. No model will be perfect, and thus it is equally important to understand what lies look like. Any understanding of the structure of lies will be essential in both, curtailing the pervasive spread and efficiently debunking them in near real time.

## Why Care About Misinformation?

It's worth asking, why anyone should care about verifying the truth value of content floating around the internet? Any normal person would understand that the answer involves scary things like ethics and morals but for those of you not convinced I'll continue on. 

How about money? Everyone likes money, right? While the total cost of misinformation is difficult to ascertain and hard to quantify, CHEQ AI, a cybersecurity company that uses AI to protect entities from misinformation, commissioned a study run by professor Roberto Cavazos from the University of Baltimore, to quantify the cost of mis-information on businesses in the US and abroad. You can find the whole article here, but the general conclusions are the following.

* 37 billion in losses on the stock market
* 17 billion in losses annually from false/misleading financial information
* 9.54 billion spent on reputation management for false information about companies.
* 3 billion has been already spent to combat fake news.
* 400 million spent by US public institutions defending against misinformation.
* 250 million spent on brand safety measure.

All in all, the report concluded that 78 billion dollars is the total spent because of and combatting against misinformation in the recent past. These are only the direct costs that could be found. The price tag for indirect consequences is unknown and assumed to be many times greater than what can be directly attributed.

Not all of this applies equally to our stakeholders. For social media companies, as the host for much of the fake news, their share of the burden is significant. In 2018, Facebook's CEO claimed that the company's annual budget for defending against misinformation was more than 3 billion dollars. This includes over 30000 positions world-wide which are dedicated content monitors. While I'm not a legal scholar, the relationship social media has to fake news isn't completely detached. Several governments world-wide have entered into agreements with social media companies to fight versions of misinformation that may be deemed bad for the public or governmental institutions. This relationship is costly for the companies and often required in order to conduct business around the world.

What about journalists and politicians? Well, the 9.54 billion in reputation management hits individuals as well as companies. Both journalists and politicians are aware of the various laywers and fixers involved in damage control for misinformaion that goes viral. The effects can be damaging to one's life, relationships, career and finances. 

Given this is paper is being written at the end of 2020, I've reserved a specific shout out for COVID-19 to underscore the idea that everyone is in this fight. Misinformation has taken a toll of human life in the US in the year following the outbreak of COVID-19. 300,000+ as of writing this. I'll let that one sink in.

In addition, the ensuing social and commercial restrictions have negatively impacted hundreds of thousands of individuals and businesses world-wide. The handling of the virus has been negatively affected by the dissemination of fake news, whereby prolonging and exacerbating the human and economic damage. As the pandemic is still raging and vaccines have only just begin being distributed, the total cost is still unknown. But there can be little doubt that the price tag of the pandemic will be a very large number. I'm interested to see how much of the total cost will be attributed to misinformation when the dust settles.

## What is the Business Objective?

Having made the case why we should all should - and most of us currently do - care about misinformation, let's move on to the more important question. What is the business solution to this complex question? 

There are several different versions of what monitoring for misinformation can look like. The scale ranges from simple identification and defense preperation to more advanced methods of targeted censorship, moderation and my personal favorite, providing truth and context. Any case begins with the ability to determine the truth value of statements posted to social media. As such, the primary business objective is to create a way to effectively categorize user generated text as to the level of truth it contains in a cost-effective manner. Ideally, this will be done with some level of specificity.

No model is perfect, and I take personal care when thinking about integrating AI into a system that makes decisions about people. In the interest of staying away from the messy world of censorship, this paper will focus on finding the truth value of statements and shedding light on the connection between truth and language such that social media companies will be able to find specific text with strong correlations to specific truth categories.

# Data Understanding

The data used for this paper was pulled from Politifact.com. Politifact is a lovely organization that takes the time to investigate viral claims that deal with hot or sensative topics. Unsurprisingly, many of these claims are political in nature, but not strictly so. Politifact investigates these claims and writes a small article summarizing their findings as to the validity of the content in question. They do the hard work! If you find this paper at all interesting it might be worth sending a little money their way. 5 bucks people, I'm not asking for the moon. Anyways, Politifact vets the claims and boils it down to six distinct categories: Pants on fire, false, barly-true, half-true and true. Pants fire is a special category reserved for falsehoods that not only lack any semblence of truth, but are also particularly outlandish or detrimental. A good example would be, "The democrates say they are going to blow up the moon." The classifications are somewhat subjective but because professionals took the time to look into the truth and cite their sources, I hold the labels as the gold standard when it comes to verifying the level of truth in a statement. There are approximately 19000 quotes in total.

An important consideration ever present in modeling is class imbalance. Here's our given composition.

![pie_chart](https://drive.google.com/uc?id=1EjxE9yDNB9mj2lFiUfnypbdTDUOcZMKC)

The imbalence is not as severe as it could be, but this still merits adjusting for. Sci-kit learn has a particular function which calculates the weights needed for each class in order to mimic balanced amounts of data.

What do these columns look like? The next visual represents the 10 most frequent authors and the breakdown of their quotes into the various truth values.

![top_authors](https://drive.google.com/uc?id=1gL4xn0HS9aSG4sG4O38i6uVu8MV_Zqlh)

With the rise of social media, it makes sense that those at the top of the list are figures in the modern political scene. In general, Politifact reviews quotes with implications for the real world. With the prolification of mis-information as a political tool, much of their work has centered on uncovering the truth behind statements made in the political sphere. Please note that there are lumped categories such as "Facebook posts" and "Viral image". These categories include quotes that achieved viral status in spreading through social media sites. You won't find a good definition of viral in this paper as the exact criteria used to determine what elevates something to viral status isn't exactly clear. For a better sense of the nature of the quotes themselves, please see the hosted application for this paper.

Given the top authors are pulled mostly from the political realm, it follows that words common in modern political times are highly represented. Check out the following two word clouds for a more intuitive idea about the nature of the quotes corpus. This example excludes stop words from consideration (the, and, where, in, is, I, etc...).

<img 
  src="https://drive.google.com/uc?id=1aJmMaPmbhdrM2VB7OGGISv7sRhe61na4" 
/>

Unsuprizingly, the word cloud that excludes stop words consists of what we might expect: Words about hot topic issues, words about the voting process, locations, individuals in politics, money, governmental institutions and the like. 

# Data Preparation for Modeling
Most of our data preprocessing has already been completed in our search for knowledge about the corpus. To re-hash, the methods which will be employed for modeling will be the following. 

1. Stop Word removal 
2. lemmatization using NLTK
3. ID Vectorization
4. TF-IDF Vectorization
5. Token Embeddings
6. Word2vec Embeddings

Various combinations of these tools will be employed to determine what level of preprocessing gives the best result. For those of you who are new to the NLP word here are some short explanations of these terms.

## Stop Words

Stop words are the high frequency words used in speech that don't give any information about the content they are used in. "The", "For", "When" are some examples. Pronouns are also considered stop words because they don't give information when taken out of context.

## Lemmatization

This is a technical term from the etemology - the study of words - world. Lemmatizing is the process of reducing words to their root form. 

Running -> run, 

says -> say, 

denies -> deny. 

The rules for how this works are technical and based on latin. You've probably seen this in action before in 7th grade grammer. If the examples don't enlighten you, there are many great resources that will explain further. 

Why would you lemmatize words? The fundamental idea is to concentrate meanings so the model can more easily understand words. When the computer "reads" words it really is processing a bunch of numbers. For a computer, "run," "runs," and "running" are all totally different words. As humans, we know that these words are all conveying the idea of "run" in various tenses. Thus, when lemmatizing words, we assign all these instances to a single word and give the model much more information from which to make decisions for any given root word. 

## ID Vectorization

This is the one of the most basic forms of encodding information. The idea is simple; Give every word in the corpus a unique integer ID. That's it.

## TF-IDF

TF-IDF is a token frequency based vectorization method. TF-IDF stands for term frequency - inverse document frequency and is most used in conjunction with NLP feature engineering processes such as stemming and lematization. We will follow this trend. To know more about the math behind TF-IDF you can check out the [wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) page and go from there. 

## Word Embeddings
 
Word embeddings were a field that bloomed with the emergance of deep NLP. There are several great detailed explanations of how embeddings were derived from neural newworks. For simplicity sake, word embeddings are created by defining a space that designate the number of dimensions you want to consider for each word - typically integer multiples of 50 are fan favorites. This space is then used as part of a neural network architecture and trained for several epochs. What results is a matrix with the number of dimensions specified and weights for each dimension as adjusted by the neural network while training on a text corpus. More interestingly, because the neural network attemped to optimize the matrix for the text corpus supplied, the dimensions of the matrix end up being surprisingly acute semantic relationships between the words in the text. Boiling this down, embeddings are matricies that carry semantic relationships between words. This is very effective within deep NLP as the decisions made by the overarching regressor/classifier are use the underlying semantic meanings of the words within the text to make decisions rather than more general tools (like TF-IDF or other term frequency vectorizing methods.)


# LSTM Neural Network

LSTM, until recently, were on the cutting edge of deep NLP breakthroughs. The underlying premise is to create internal states for the nodes of model, such that, they can keep track of words and their associated embeddings. In this specific model we will use a bi-directional layer of LSTM cells. The idea to state is really another way of saying time dependence. What does time have to do with text data? The order or reading! Text doesn't make sense if we choose a bunch of words in random succession. There are complex ideas embedded within the context surrounding each word, and an LSTM attempts for keep track of this information by adjusting its weights at the first word in a sequence and feeding the new configuration back into itself for every new word. This way, the model remembers (somewhat) the imporatnce of the previous words in the phrase. 

LSTMs do have a couple issues. One is known as saturation. There are only so many weights an LSTM can adjust as each successive step through the text data. Take the example of having 100,000 weights that the layer can adjust. This seems moere than adequite for being able to learn for a couple words. What about 1000 words? What about 100,000? One weight per word doesn't seem sufficient. Saturation is actually a well understood problem that arose in predecessors to the LSTM. The LSTM has internal functions that attempt to conpensate for this issue by allowing the model to forget words that don't seem very impactful. While a great idea, the implementation is far from perfect and from the AI the industry's progress in the interim, this is known to be an incomplete solution.

The other issue with LSTMs is their limitation on computation. With an internal struture that is meant to mimic time dependance, LSTMs have their order of opperations fixed, thus making parallel computaiton impossible. While not the most relevant issue in this paper, it does have significant drawbacks at scale. 

So, how did the LSTM models do?!

The results were underwhelming. 

## LSTM with vanilla tokens

The vanilla tokens LSTM is currently leading the pack with 26.1% accuracy. Looking at the confusion matrix, there are some good and bad trends. On the positive side, the model appears to understand the pants-fire at a high level, whereby it correctly classified 58% of quotes with that tag. In addition  30% of the false category was mis-classified as pants-fire but this is still encouraging considering they are logical neighbors. Something not quite obvious were the those entries predicted as true. While the overall percentages are not as encouraging as the pants-fire category, the model is making reasonable connections betweent the true category and it's neighbor classes, while shunning the false quptes. 

The main eye sore for this iteration is the barely-true category. The model was clearly confused by this label and treated it as the catch all for guessing. A plurality of the incorrectly classified quptes in the barely-true category came from the false class. The false class - the largest numbers wise - was almost never guessed.

In conclusion, the LSTM hasn't met expecations for a neural network. There could be several reasons for this ranging from the architecture to the data preprocessing methods. It is my hope that these scores can be improved upon in a second version of this paper. 

For the training process itself, the network appeared to continue learning for 5 epochs, after which it began to degrade rapidly. 


<img src = "https://drive.google.com/uc?id=1O99Gzk2ACc1_HaQCbeaWiW47ZnPYExZ8" />

<img src = "https://drive.google.com/uc?id=1ESM7-V6UupxV1pXfw2gFsowL6joAqmtV" />

## LSTM with TF-IDF


Due to the incredible amount of time spent training the TF-IDF model - which has 20000 dimnensions - only the non-lemmatized tokens were used. This desicion was made based on the poor performace of this model. Interestingly enough, the TF-IDF model used the true and pants-fire categories as the catch all columns compared to the vanilla tokens model which showed a propensity for guessing barely-true. Ultimately, I don't believe TF-IDF is an appropriate vectorization method for neural networks given the incredibly large feature dimensions. Moreover, I will attribute the different prediction distribution to a degredation in the model. This iteration has lost the relationship between the false and pants-fire categories, which is pretty fundamental. 

# Combining Labels
Our first run through the modeling process has yielded interesting results as the neural networks are creating appropriate associations between the various truth values. While the overall accuracy isn't what we would like to see, it also isn't entirely necessary that we break the truth into so many categories. We will go through a couple iterations of collapsing out existing categories in logical ways to see if a more coarse assignment might create a more useful classification model. In an ideal world we would love to have a successful classification on a granular level for insight reasons. With the poor performace of our models, combining labels is more appealing in hopes there is a different configuration of our target column that will be more useful in practice. Here are the collapses we will consider:
1. Mapping 1
 * pants-fire -> false
 * false -> false
 * barely-true -> some-truth
 * half-true -> some-truth
 * mostly-true -> substantially-true
 * true -> substantially-true
2. Mapping 2
 * pants-fire -> false
 * false -> false
 * barely-true -> some-truth
 * half-true -> some-truth
 * mostly-true -> some-truth
 * true -> true
3. Mapping 3
 * pants-fire -> false
 * false -> false
 * barely-true -> not-false
 * half-true -> not-false
 * mostly-true -> not-false
 * true -> not-false
4. Mapping 4
 * pants-fire -> not-true
 * false -> not-true
 * barely-true -> not-true
 * half-true -> not-true
 * mostly-true -> not-true
 * true -> true

With re-defining our labels comes the need to reevaluate class imbalance. This is done in the same fashion as the original. Please see the code for more information. 

## Mapping 1

Mapping 1 is an experiment. The only label with a definitive truth value is the "false" category. During the training and evaluating of the previous models it's become clear that the center truth values - barely-true, half-true and mostly-true have substantial issues during classification. Of all the models I've run throughout this project, these categories have habitually defied high precision classification. Mapping 1 is more of an experiment to see if we can create some seperation. Well spoilers, again, are aren't easily seperable. Here was the testing confusion matrix. 

![confusion_matrix](https://drive.google.com/uc?id=1D79aCjdM6MBvSmdWTiR1WkbxTdmcx6-5)

![training_curves](https://drive.google.com/uc?id=13wSoSA3IQ4V0TOyvYdBHsFLfZCqisc4z)


### Overall Metrics:
* Accuracy: 46%

Our performace on the false labels was reputable. The substantially-true category did make proper associations as evidenced by the correct label being assigned at almost twice the rate as the "some-truth" category and five times the rate of the false quotes. This is known has high precision. The bad news is poor accuracy at only 17%. Again, we see that the some-truth label is difficult to classify. The model is reasonaly sure "some-truth" statements are not "false" given the 25% mis-classification rate. As for the difference between "some-truth" and "substantially-true" it looks like a coin toss. In summary, much better than a random guess but still lacking something stellar. 


## Mapping 2

Mapping 2 is more of what I would consider the logical partion of our truth values. All complete lies are false, are complete truths are true and everything else is something in the middle. 

![confusion_matrix](https://drive.google.com/uc?id=1M26w8VLYHXoIOHzKwJU7eexjGVJsnmXc)

![training_curves](https://drive.google.com/uc?id=1hGvl9wXA1IgpwDp_cHpPeSFEndWRD4kE)

### Overall Metrics:
* Accuracy: 47%

Performace conforms substantially to mapping 1. We gained some accuracy for the true label but lost some precision. False didn't change in any substantial way and the "some-truth" category preformed with an equal level of confusion - in addition to less accuracy. 

## Mapping 3

Mapping 3 and 4 was born out of desperation! Collapsing down to 3 labels yielded only limited results, which leaves the binary case. A binary consideration of our problem necessatates two seperate mappings (3 & 4) with one label being true or false and the other label being not those. I'm not going to throw out the majority of our data to make everything neatly fit into two categories as this violates the purpose of what we are trying to achieve. Spliting the data down the middle is also foolish because regardless of model performance, the categories wouldn't really tell us anything. 

![confusion_matrix](https://drive.google.com/uc?id=1UAm_5xo0Jgm__VKYfLNQvOUS0184Ewfs)

![training_curves](https://drive.google.com/uc?id=1bImtUZMqwInYwIImH1LVGsboQiLavfM1)

### Overall Metrics:
* Accuracy: 60%

Of all the models I find this one the best. The two largest percentages are on the main diagonol of the matrix which means we have successfully created a model that, given a label, will guess correctly the majority of the time! We've also managed to capture 75% of the false quotes. Precision for the false labels is low but not overwhelmingly so. The not true labels, are again, a coin toss.

## Mapping 4

The mirror image of the last model.

![confusion_matrix](https://drive.google.com/uc?id=1Bjh-wi4iUYFbSVlqhzojOzJP4-Nok8n1)

![training_curves](https://drive.google.com/uc?id=19KT4l637tSQ84nbDSVK9qRgnOqZv_ODk)

### Overall Metrics:
* Accuracy: 80%

At this point, we are confirming what we already know. The model is somewhat presice when guessing true but that overall accuracy of the true quotes is low. As for the non-true quotes, the model is looking like a slightly biased coin toss. I would certainly attribute this to the fact that the false labels are in the "not-true" column. Don't be fooled by the 80% overall accuracy. This data set is substantially biased to the "not-true" labels in terms of raw numbers. In highly imbalanced datasets, recall and precision are much more useful metrics that accuracy. 

## Section Conclusion

At this point I can confidently make the statement that for our dataset, **false statements are easier to detect than statements with non-zero truth values.** It's also worth stating for the record that **the model is reasonable able to identity the false labels overall.** Detecting false statements with reasonable surety is a huge step in the right direction and is valuble in industry. Identifying false quotes would significantly reduce the amount of manual research and moderation that needs to be done to debunk. For both social media and journalists, it might also help prioritize sending resources towards other statements that aren't so easily determined. 

# Conclusion

It's been a long road on this one. I'm going to break this conclusion up into insights about the data and the modeling process. Both paths have interesting findings but don't integrate into a single narrative in a nice way.

## Data Insights
I had a lot of fun with this dataset. Until this point I have yet to have occasion to interact with a dataset that is as colorful and important to American society. I take a certain amount of pride in my analysis having good evidence that the modern political scene is a mudslingging festival. The density of falsehoods is highly concentrated around statements that are here say or making claims about politicians. "Obama said this," "Romney said that," "Trump did this.," "McCain's tax proposal is this." Those types of statements are much more likely to be lacking in truth. It was also exciting to identify what I believe to be malicious misinformation; the quinticential pants-fire quote that leads to a grown man assaulting a pizza joint beceause he is single -handedly going to upset the Hillary Clinton child sex ring. These quotes are formed largely on social media and includes language that is meant to incite violent emotions. 

## Models
The best model in this paper was the Vanilla tokens LSTM by a razor thin margin. The one common thread among all models was the ability to classify the "pants-fire" category with high precision. Of all the diappointments in the modeling process for this project, this is the diamond in the rough. The pants-fire category is full of the more vile and dangerous retorhic that we've become accustomed to. It is encouraging and enlighting to know that models and neural networks can find this information at comparitively high accuracy. At some points, this is +50% in the full 6 truth value models and is around 75% in the collapsed labels versions. Such models can be a useful foundation for future work. The ability to weed out the completely false statements is huge in terms of quelling the spread of misinformaion and in terms of resource allocaion when combating misinformation. At the end of the day, I conclude that no model in this paper is production worthy. 

## Recommendations
Despite the lack of the production worthy model there are several action items for various stake holders. 
1. For those wishing to devote resources to fight misinformation; virtually all clusters of the false quotes center around presidential elections and candidates. Reasorces should be ramped up and deployed at times of heightened contention during presidential elections and primaries. 
2. For politicians, most misinformation is based around people and not issues (with the notable exception of health care). If you wish to advance legislation, decoupling the issue from individuals will the best way to limit misinformation's impact.
3. For Journalists and people at large: Email chains, social media posts and viral content should never be used as a source for content.

# Future Work and Additional Features

As the author of this paper it's depressing how much is still left to do. I mean that in the most optimistic way possible. 

1. The modeling portion of my work was largely unsuccessful. I don't think there are true limits to what neural networks can accomplish which means there's always something I can do to make them better. In terms of realistic goals, I would like to bring in some more quotes; perhaps double the size of the current dataset. This would be invaluble to understanding whether my models will improve or if I need to go back to the drawing board in terms of the modeling approach. The second modeling goal is to implement a version of a BERT transformer. BERT is the state of the art in most NLP benchmarks and is based on multi-head self attention (which I use in this paper). 

2. As for the data exploration; I would like to make more use of the various NLTK apis for text engineering. These tool can create wonderful visuals that will make my job of communication the nuances of the data more simple and effective.

3. The app was the one thing I thought didn't turn out half bad. I would like to make the data more interactive on the app such that users can select specific quotes and interact with them through the various text models: lda, word2vec and lime in real time. This is a major upgrade to the feel and likely the first major commit I'll make after the initial launch of this paper. 

These are the reasonable goals on the horizon! I am ambitious that I will see them through in the near future as my intersest in this subject hasn't dissolved - wanned slightly from burnout, but a short break should do wonders. Misinformation remains a massive obsticle in social cohesion and well being and the work isn't going to be done by watching on the sidelines. 

There are some sections of code below this text that include elements I didn't want to integrate into the general narrative or aren't ready for presentation. Feel free to check them out but don't blame me for the poorly formatted and commented code!

Cheers everyone, thanks for reading.
