#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Importing libraries')
import sys
import pandas as pd
import spacy
import numpy as np
import pandas as pd
import gensim
from gensim.models.phrases import Phrases, Phraser

import wordcloud as wc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md')


# In[2]:


print_preprocessed_collection = False


# In[3]:


print("Load 5000 rows of the dataset...")
df = pd.read_csv('./data/dataset.csv', nrows=5000)
df


# In[4]:


print("Create the collection using the 'webTitle' and 'bodyContent' columns")

collection = df.webTitle + df.bodyContent
collection


# # Preprocessing

# In[5]:


print('NaN values in collection:',collection.isnull().sum())
print('Remove NaN values...')
collection.dropna(inplace=True)


# In[6]:


print('Lemmatization...')
lematized_collection = []

for i,d in enumerate(collection):
  print(f'{i} of {len(collection)}',end='')
  tdoc=nlp(d)
  lm = ' '.join([token.lemma_ for token in tdoc  if not(
      token.is_stop == True or 
      token.is_digit == True or 
      token.is_punct == True or 
      token.like_url == True or 
      token.like_email == True or
      token.like_num == True or
      token.is_currency == True or
      token.pos_ == 'VERB' or
      token.lemma_.startswith('@') or
      len(token.lemma_) < 3
  )])
  lematized_collection.append(lm)
  print('\r\r\r\r\r\r\r\r',end='')

print('\r\r\r\r\r\r\r\r',end='')
print('Collection lematized')


# In[7]:


print('Original first document:\n')
print(collection[0])
print('\n\nLemmatized first document:\n')
print(lematized_collection[0])


# In[8]:


print('Tokenizing...', end='')

tokenized_collection = [gensim.utils.simple_preprocess(doc, deacc= True, min_len=3) for doc in lematized_collection] 

print('\r\r\r\r\r\r\r\r',end='')
print('Collection tokenized')


# In[9]:


print('Building bigrams...')

phrases  = Phrases(tokenized_collection, min_count = 2,threshold=9)
bigram = Phraser(phrases)
preprocessed_collection = [bigram[d] for d in tokenized_collection]

print('Bigrams built')


# # Statistics

# In[10]:


def calculateStatistics(collection):
  ndocs = len(collection)

  collection_size = 0

  shortest_doc_index = 0
  largest_doc_index = 0

  for i, d in enumerate(collection):
    collection_size += len(d)

    # shortest_doc
    if (len(d) < len(collection[shortest_doc_index])):
      shortest_doc_index = i

    #largest_doc
    if (len(d) > len(collection[largest_doc_index])):
      largest_doc_index = i

  print("Number of documents: ", ndocs)
  print(f"Shortest doc index: { shortest_doc_index }, size: { len(collection[shortest_doc_index]) }")
  print(f"Largest doc index: { largest_doc_index }, size: { len(collection[largest_doc_index]) }")
  print("Average size: ", collection_size/len(collection))

calculateStatistics(preprocessed_collection)


# In[11]:


threshold = 50

print(f'Removing documents with size smaller than {threshold}...')

ncol = []
for doc in preprocessed_collection:
  if len(doc) >= threshold:
    ncol.append(doc)

print('Recalculating statistics...\n')

calculateStatistics(ncol)

preprocessed_collection = ncol


# In[12]:


if print_preprocessed_collection:
    for doc in preprocessed_collection:
        print(doc,'\n')


# In[13]:


print('Original first document:\n')
print(collection[0])
print('\n\nPreprocessed first document:\n')
print(preprocessed_collection[0])


# In[14]:


figure(figsize=(15, 20))

allw = []
for d in preprocessed_collection:
  allw += d
allw = ' '.join(w for w in allw)

mycloud = wc.WordCloud().generate(allw)
plt.imshow(mycloud)

