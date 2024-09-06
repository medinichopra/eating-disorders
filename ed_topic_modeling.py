# import statements

!pip install autocorrect
!pip install unidecode
!pip install contractions

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
import re
import ast
from autocorrect import Speller
import unidecode
import contractions
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
import gensim
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# read file
df = pd.read_csv('/content/EatingDisorders_posts.csv')

# extract specific relevant fields including the IDs, text, when it was posted, number of comments
df = df[["id", "selftext", "title", "created_utc", "num_comments", "score"]]

# changing date-time format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')

# combining title and selftext columns to combine all text of the post
df['title'] = df['title'].str.replace('Request: ', '', regex=False)
df['selftext'] = df['selftext'].astype(str)
df['text'] = df['selftext'].str.cat(df['title'], sep=' ')

# dropping extra columns to avoid clutter
df = df.drop(['selftext', 'title', 'created_utc'], axis=1)
df = df.dropna()

def data_cleaning(text):

  # lower case and html tags
  text = text.lower()
  html_pattern = r'<.*?>'
  text = re.sub(pattern=html_pattern, repl=' ', string=text)

  # urls, numbers, de-anonymising the data
  url_pattern = r'https?://\S+|www\.\S+'
  text = re.sub(pattern=url_pattern, repl=' ', string=text)
  number_pattern = r'\d+'
  text = re.sub(pattern=number_pattern, repl=' ', string=text)
  email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
  text = re.sub(pattern=email_pattern, repl=' ', string=text)

  # unidecode, expanding contractions, remove punctutation
  text = unidecode.unidecode(text)
  text = contractions.fix(text)
  text = text.translate(str.maketrans('', '', punctuation))

  # removing single characters, extra spaces
  single_char_pattern = r'\s+[a-zA-Z]\s+'
  text = re.sub(pattern=single_char_pattern, repl=" ", string=text)
  space_pattern = r'\s+'
  text = re.sub(pattern=space_pattern, repl=" ", string=text)

  return text

def data_processing(text):

  tokens = word_tokenize(text.lower())

  # remove custom stopwords and tokenize
  custom_stopwords = {'eating', 'disorder', 'removed'}
  stop_words = set(stopwords.words('english')) | custom_stopwords
  tokens = [token for token in tokens if token not in stop_words]

 # lemmatize
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]

  return tokens
  # return " ".join(tokens) # for word cloud generation

# running the cleaning and processing
df['cleaned_text'] = df['text'].apply(data_cleaning)
df['processed_text'] = df['cleaned_text'].apply(data_processing)
df.head(5)

# custom cleaning: thread post titles may contain this text
df = df[~df['text'].str.startswith("This is a weekly thread")]

# extracting year so as to divide in 2022 and 2023
df['year'] = df['created_date'].dt.year
df_2022 = df.loc[df['year'] == 2022].reset_index(drop=True)
df_2023 = df.loc[df['year'] == 2023].reset_index(drop=True)

# word cloud for 2022
long_string = ' '.join([' '.join(row) for row in df_2022['processed_text']])
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()

# word cloud for 2023
long_string = ' '.join([' '.join(row) for row in df_2023['processed_text']])
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()

# ran pipeline for 2022, and then 2023
id2word = Dictionary(df_2023['processed_text'])
corpus = [id2word.doc2bow(d) for d in df_2023['processed_text']] # same cell run for df_2022['processed_text']

# topics for 2022

# loading model with parameters
base_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=16, #tried with 5, 6, 10, 12, 15
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=0.91,
                                            eta=0.01,
                                            iterations=100,
                                            eval_every=1)

# loads list of topic probabilities for each post, creates a new column for each topic (1 to 16), and creates corresponding column names, appending it to previous 2022 dataframe
topic_distributions = base_model[corpus]
topics_df = pd.DataFrame([[dict(doc).get(i, 0) for i in range(base_model.num_topics)] for doc in topic_distributions])
topics_df.columns = [f'Topic_{i+1}' for i in range(base_model.num_topics)]
df_2022 = pd.concat([df_2022, topics_df], axis=1)

# creates separate column for the dominant topic by calculating max value, and separate column for the corresponding dominant topic strength
df_2022['dominant_topic'] = topics_df.idxmax(axis=1)
df_2022['topic_strength'] = topics_df.max(axis=1)

# prints only the top 10 topic words for each (not the strength)
words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]
topics = [' '.join(t[0:10]) for t in words]

for id, t in enumerate(topics):
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")

# topics for 2023

# same process as listed above
base_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=16, #5, 6, 10, 12, 15, 16
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=0.91,
                                            eta=0.01,
                                            iterations=100,
                                            eval_every=1)

topic_distributions = base_model[corpus]

topics_df = pd.DataFrame([[dict(doc).get(i, 0) for i in range(base_model.num_topics)] for doc in topic_distributions])
topics_df.columns = [f'Topic_{i+1}' for i in range(base_model.num_topics)]

df_2023 = pd.concat([df_2023, topics_df], axis=1)

df_2023['dominant_topic'] = topics_df.idxmax(axis=1)
df_2023['topic_strength'] = topics_df.max(axis=1)

words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]
topics = [' '.join(t[0:10]) for t in words]

for id, t in enumerate(topics):
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")

# trial for perplexity and coherence scores, used for 2022 and 2023
base_perplexity = base_model.log_perplexity(corpus)
print('\nPerplexity: ', base_perplexity)

coherence_model = CoherenceModel(model=base_model, texts=df_2022['processed_text'],dictionary=id2word, coherence='c_v')
coherence_lda_model_base = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_base)

# 2022 topic visualization
!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(base_model, corpus, id2word)
vis

# 2023 topic visualization
!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(base_model, corpus, id2word)
vis

# write out dataframe with strength of each topic for each post, and column for dominant topic
df_2022.to_csv('topics_for_2022.csv')
df_2023.to_csv('topics_for_2023.csv')

# number of posts per topic 2022
posts_per_topic = df_2022['dominant_topic'].value_counts()
print(posts_per_topic)

# number of posts per topic 2023
posts_per_topic = df_2023['dominant_topic'].value_counts()
print(posts_per_topic)