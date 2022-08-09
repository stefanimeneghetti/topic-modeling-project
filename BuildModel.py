#!/usr/bin/env python
# coding: utf-8

from gensim.models.coherencemodel import CoherenceModel
import tomotopy as tp
import csv

ldamodel = tp.LDAModel


# First we will import the preprocessed tokens.

# In[24]:


preprocessed_collection = []

with open('data/tokens_20k.csv', 'r', newline='') as file:
  myreader = csv.reader(file, delimiter=',')
  for row in myreader:
    preprocessed_collection.append(row)


# In[ ]:


find_hyperparameters =  False


# In[ ]:


if find_hyperparameters:
    rm_top =[10, 20, 30] #the number of top words to be removed (default 0)
    min_df = [0, (int) (len(preprocessed_collection) * 0.01), (int) (len(preprocessed_collection) * 0.02)] #minimum document frequency of words (default 0)
    min_cf = [0, (int) (len(preprocessed_collection) * 0.01)] #minimum collection frequency of words. (default 0)
    alphas = [0.2, 0.15, 0.1, 0.05] #hyperparameter of Dirichlet distribution for document-topic (default 0.1)
    etas = [0.2, 0.15, 0.1, 0.05]  #hyperparameter of Dirichlet distribution for topic-word (default 0.01)
    K=[10, 20, 30, 40, 50]
    iterations = len(rm_top) * len(min_df) * len(min_cf) * len(alphas) * len(etas) * len(K)
    cv=[]
    iter = 0
    configs = []
    print('# of docs:', len(preprocessed_collection)) 
    print('# of iterations:', iterations)
    for k in K:
      for mdf in min_df:
        for rm in rm_top:
          for a in alphas: 
            for e in etas:
              for mcf in min_cf:
                iter += 1
                print(iter, end=' ')
                #create an object
                #tw term weight IDF (Inverse Document Frequency term weighting), ONE (equal - default), PMI (Use Pointwise Mutual Information term weighting)
                LDA = ldamodel(tw = tp.TermWeight.IDF, k=k, alpha = a , eta = e, seed = 9999,  min_df = mdf, min_cf = mcf, rm_top = rm)
                #add documents to it
                for doc in preprocessed_collection:
                    LDA.add_doc(doc)
                #train
                LDA.train(iter = 500) # iter (# of iterations - default 10) # workers (# of cores to be used - 0 all)
                #get the coherence (c_v)
                coh = tp.coherence.Coherence(LDA, coherence='c_v')
                average_coherence = coh.get_score()
                print('K: %2d mcf: %2d mdf: %2d rm: %2d alfa: %.3f beta: %.3f coherence: %.3f'%(k,mcf,mdf,rm,a,e,average_coherence),end=' Collection ')
                print(' Vocab size:', len(LDA.used_vocabs), ' # of words:', LDA.num_words)
                configs.append([k,mcf,mdf,rm,a,e,average_coherence])
                cv.append(average_coherence)


# ## Create model

# In[21]:


rm_top = 10
min_df = 194
min_cf = 194
alpha = 0.2
eta = 0.1
K = 20

LDA = ldamodel(
    tw = tp.TermWeight.IDF,
    k=K, alpha = alpha, 
    eta = eta,
    seed = 9999,
    min_df = min_df,
    min_cf = min_cf,
    rm_top = rm_top
)
for doc in preprocessed_collection:
    LDA.add_doc(doc)

LDA.train(iter = 500) 


# In[22]:


coh = tp.coherence.Coherence(LDA, coherence='c_v')
av_co = coh.get_score()
print("\nCoherence:", av_co)


# In[17]:


LDA.summary()


# In[26]:


print("\n** Topics **\n")
for i in range(K):
  print("Topic", i, end=' => ')
  for w in LDA.get_topic_words(i):
    print(w[0], end=' ')
  print()

