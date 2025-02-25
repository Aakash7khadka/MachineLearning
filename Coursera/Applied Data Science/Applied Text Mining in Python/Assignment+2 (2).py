
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[2]:


import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# In[8]:


len(text1)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[7]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[9]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[12]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[28]:


def answer_one():
    text=set(text1)
    
    return len(text)/len(text1)

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[106]:


def answer_two():
    dist=nltk.FreqDist(text1)
    freq1=dist['whale']
    freq2=dist['Whale']
    freqt=len(text1)
    return (freq1+freq2)/freqt*100

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[57]:


def answer_three():
    dist=dict(nltk.FreqDist(text1))
    sort_dict=sorted(dist.items(),key=lambda x:x[1],reverse=True)
    
    return sort_dict[:20]
answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[63]:


def answer_four():
    dist=nltk.FreqDist(text1)
    token=[w for w in dist if len(w)>5 and dist[w]>150]
    
    
    return sorted(token)

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[125]:


def answer_five():
#     text=[(w,len(w)) for w in text1 if w.isalpha()]
   
    maxx=[(w,len(w)) for w in text1 if len(w)>19]
    
    return max(maxx,key=lambda x:x[1])

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[120]:


def answer_six():
    text=[w for w in text1 if w.isalpha()]
    dist=nltk.FreqDist(text)
    sorted_freq=sorted(dist.items(),key=lambda x:x[1],reverse=True)
    have_2000=[w for w in sorted_freq if w[1]>2000]
    have_2000=[(w[1],w[0]) for w in have_2000]
    return have_2000

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[84]:


def answer_seven():
    words=nltk.word_tokenize(moby_raw)
    sents=nltk.sent_tokenize(moby_raw)
    
    
    return len(words)/len(sents)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[119]:


def answer_eight():
    word=nltk.word_tokenize(moby_raw)
    
    tag=nltk.pos_tag(word)
    pos=[p[1] for p in tag]
    pos=[ p for p in pos ]#if p.isalpha()]
    dist=nltk.FreqDist(pos)
    sort=sorted(dist.items(),key=lambda x: x[1],reverse=True)
    return sort[:5]

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[3]:


from nltk.corpus import words

correct_spellings = words.words()
correct_spellings


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[49]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    final=[]
    for word in entries:
        spellings=[w for w in correct_spellings if w[0]==word[0]]
        result=[]
        for spell in spellings:
            res=(nltk.jaccard_distance(set(nltk.ngrams(word,3)),set(nltk.ngrams(spell,3))),spell)
            result.append(res)
            
#         print(result)    
        print(min(result))
        final.append(min(result)[1])
    return final
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[53]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    final=[]
    for word in entries:
        spellings=[w for w in correct_spellings if w[0]==word[0]]
        result=[]
        for spell in spellings:
            res=(nltk.jaccard_distance(set(nltk.ngrams(word,4)),set(nltk.ngrams(spell,4))),spell)
            result.append(res)

    #         print(result)    
        print(min(result))
        final.append(min(result)[1])

    return final
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    
    return # Your answer here 
    
answer_eleven()

