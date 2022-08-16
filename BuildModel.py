#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gensim.models.coherencemodel import CoherenceModel
import tomotopy as tp
import csv
import matplotlib.pyplot as plt

ldamodel = tp.LDAModel


# First we will import the preprocessed tokens.

# In[3]:


preprocessed_collection = []

with open('data/tokens_20k.csv', 'r', newline='') as file:
  myreader = csv.reader(file, delimiter=',')
  for row in myreader:
    preprocessed_collection.append(row)


# In[4]:


find_hyperparameters =  False


# In[5]:


if find_hyperparameters:
    rm_top =[10, 20, 30, 40] #the number of top words to be removed (default 0)
    min_df = [0, (int) (len(preprocessed_collection) * 0.005),(int) (len(preprocessed_collection) * 0.01), (int) (len(preprocessed_collection) * 0.02), (int) (len(preprocessed_collection) * 0.02)] #minimum document frequency of words (default 0)
    min_cf = [0, 10, 20, 30, 50, 100, 200, 300] #minimum collection frequency of words. (default 0)
    alphas = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05] #hyperparameter of Dirichlet distribution for document-topic (default 0.1)
    etas = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]  #hyperparameter of Dirichlet distribution for topic-word (default 0.01)
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

# In[6]:


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


# In[7]:


coh = tp.coherence.Coherence(LDA, coherence='c_v')
av_co = coh.get_score()
print("\nCoherence:", av_co)


# In[8]:


LDA.summary()


# In[9]:


print("\n** Topics **\n")
for i in range(K):
  print("Topic", i, end=' => ')
  for w in LDA.get_topic_words(i):
    print(w[0], end=' ')
  print()


# In[10]:


topics_dist = [(x, 0) for x in range(K)]
topics_docs = [[] for x in range(K)]
c = 0
for index, doc in enumerate(LDA.docs):
  threshold = 0.3

  d_topics = doc.get_topics(top_n=4)
  found_topic = False;

  for topic in d_topics:
    if topic[1] >= threshold:
      ntd = topics_dist[topic[0]][1] + 1
      topics_dist[topic[0]] = (topic[0], ntd)
      topics_docs[topic[0]].append(index)
      found_topic = True    
    
  if not found_topic:
    ntd = topics_dist[d_topics[0][0]][1] + 1
    topics_dist[d_topics[0][0]] = (d_topics[0][0], ntd) 


# In[11]:


fig = plt.figure(figsize=(10, 4))
ax = fig.add_axes([0,0,1,1])
topics = [str(x) for x in range(K)]
documents = [y for (x, y) in topics_dist]
ax.bar(topics, documents)
plt.show()


# In[12]:


topics_dist = sorted(topics_dist, reverse=True, key=lambda x: x[1])

print("\n** Sorted Topics **\n")

for topic in topics_dist:
  print("Topic", topic[0], end=' => ')
  for w in LDA.get_topic_words(topic[0]):
    print(w[0], end=' ')
  print()


# Topic 17 => child thing life friend mother family man parent father woman (Family)
# 
# Topic 18 => book woman novel story writer man word life black author (Books)
# 
# Topic 5 => police court case prison law investigation officer australia report inquiry (Police Cases)
# 
# Topic 12 => brexit britain europe referendum country prime_minister european british cameron vote (United Kingdom News)
# 
# Topic 4 => school student child university community council education housing work young_people (Young People)
# 
# Topic 7 => game player team england sport match race season ball coach (Sports)
# 
# Topic 0 => film bbc movie actor comedy character episode series star hollywood (Entertainment)
# 
# Topic 10 => art artist theatre museum work play exhibition painting london gallery (Art)
# 
# Topic 19 => club player game team football season goal manager premier_league liverpool (Soccer)
# 
# Topic 11 => trump clinton election donald_trump candidate labor obama president sanders hillary_clinton (Politics)
# 

# In[13]:


LDA.docs[topics_docs[5][20]] # PoliceCases Topic Example


# In[14]:


LDA.docs[topics_docs[7][100]] # Sports Topic Example


# In[15]:


LDA.docs[topics_docs[11][11]] # Politics Topic Example


# In[18]:


LDA.docs[topics_docs[19][12]]

