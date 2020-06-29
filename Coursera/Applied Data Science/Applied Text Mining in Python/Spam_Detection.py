
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[26]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[27]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[28]:


def answer_one():
    target=spam_data['target']
    
    return target.sum()/len(target)*100


# In[29]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[30]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vector=CountVectorizer()
    vector.fit(X_train)
    feature_names=np.array(vector.get_feature_names())
    maxx=max(feature_names,key=len)
    return maxx


# In[31]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[32]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vector=CountVectorizer()
    vector.fit(X_train)
    xtrain_vectorized=vector.transform(X_train)
    nb=MultinomialNB(alpha=0.1).fit(xtrain_vectorized,y_train)
    pred=nb.predict(vector.transform(X_test))
    
    return roc_auc_score(y_test,pred)


# In[33]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vector=TfidfVectorizer()
    vector.fit(X_train)
    feature_names=np.array(vector.get_feature_names())
    xtrain_vectorized=vector.transform(X_train)
    tfidf_index=xtrain_vectorized.max(0).toarray()[0].argsort()
    smallest=pd.Series(tfidf_index[:20],index=feature_names[tfidf_index[:20]]).astype('float64')
    largest=pd.Series(tfidf_index[:-21:-1],index=feature_names[tfidf_index[:-21:-1]]).astype('float64')
    return smallest.sort_values(),largest.sort_values(ascending=False)


# In[35]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[36]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
def answer_five():
    vector=TfidfVectorizer(min_df=3)
    vector.fit(X_train)
    xtrain_vectorized=vector.transform(X_train)
    model=MultinomialNB(alpha=0.1)
    model.fit(xtrain_vectorized,y_train)
    auc=roc_auc_score(y_test,model.predict(vector.transform(X_test)))
    
    
    return auc


# In[37]:


answer_five()


# In[48]:


def add_len(row):
    row['text_length']=len(row['text'])
    return row


# In[69]:


def ser_len(ser):
    count=0
    for sen in ser:
        count+=len(sen)
    return count


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[39]:


def answer_six():
    spam=spam_data[spam_data['target']==1].reset_index(drop=True)
    not_spam=spam_data[spam_data['target']==0].reset_index(drop=True)
    spam=spam.apply(add_len,axis=1)
    not_spam=not_spam.apply(add_len,axis=1)
    spam_sum=spam['text_length'].sum()
    nspam_sum=not_spam['text_length'].sum()
    return nspam_sum/len(not_spam),spam_sum/len(spam)
#     return spam['text_length'].sum()


# In[40]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[41]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[89]:


from sklearn.svm import SVC

def answer_seven():
    vector=TfidfVectorizer(min_df=5)
    vector.fit(X_train)
    xtrain_vectorized=vector.transform(X_train)
    xtest_vectorized=vector.transform(X_test)
    xtrain_features=add_feature(xtrain_vectorized,X_train.str.len())
    xtest_features=add_feature(xtest_vectorized,X_test.str.len())
    svc=SVC(C=10000)
    svc.fit(xtrain_features,y_train)
    pred=svc.predict(xtest_features)
    return roc_auc_score(y_test,pred)


# In[91]:


answer_seven()


# In[100]:


def num_count(row):
    count=0
    for x in row['text']:
        if x.isdigit():
            count+=1
    row['num_count']=count
    return row


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[103]:


import re
def answer_eight():
    spam=spam_data[spam_data['target']==1].reset_index(drop=True)
    not_spam=spam_data[spam_data['target']==0].reset_index(drop=True)
    spam=spam.apply(num_count,axis=1)
    not_spam=not_spam.apply(num_count,axis=1)
    avg=(spam['num_count'].sum())/(len(spam))
    navg=(not_spam['num_count'].sum())/(len(not_spam))
    return navg,avg


# In[104]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[115]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    vector=TfidfVectorizer(min_df=5,ngram_range=(1,3))
    xtrain_transformed=vector.fit_transform(X_train)
    xtest_transformed=vector.transform(X_test)
    xtrain_final=add_feature(xtrain_transformed,[X_train.apply(lambda x:len(x)),X_train.apply(lambda x:len(re.findall(r'\d',x)))])
    xtest_final=add_feature(xtest_transformed,[X_test.apply(lambda x:len(x)),X_test.apply(lambda x:len(re.findall(r'\d',x)))])
    logreg=LogisticRegression(C=100).fit(xtrain_final,y_train)
    pred=logreg.predict(xtest_final)
    return roc_auc_score(y_test,pred)


# In[116]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[152]:


def answer_ten():
    spam=spam_data[spam_data['target']==1].reset_index(drop=True)
    not_spam=spam_data[spam_data['target']==0].reset_index(drop=True)
    avg=(spam['text'].apply(lambda x: len(re.findall(r'\W',x))).sum())/len(spam)
    navg=(not_spam['text'].apply(lambda x: len(re.findall(r'\W',x))).sum())/len(not_spam)
    
    return navg,avg


# In[153]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[179]:


def answer_eleven():
    vector=CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb')
    vector.fit(X_train)
    feature_names=np.array(vector.get_feature_names())
    xtrain_vectorized=vector.transform(X_train)
    xtest_vectorized=vector.transform(X_test)
    f2=X_train.apply(lambda x:len(re.findall(r'\d',x)))
    f22=X_test.apply(lambda x:len(re.findall(r'\d',x)))
    f1=X_train.apply(lambda x:len(x))
    f11=X_test.apply(lambda x:len(x))
    f3=X_train.apply(lambda x : len(re.findall(r'\W',x)))
    f33=X_test.apply(lambda x : len(re.findall(r'\W',x)))
    xtrain_final=add_feature(xtrain_vectorized,[f1,f2,f3])
    xtest_final=add_feature(xtest_vectorized,[f11,f22,f33])
    logreg=LogisticRegression(C=100).fit(xtrain_final,y_train)
    pred=logreg.predict(xtest_final)
    auc=roc_auc_score(y_test,pred)
    coeff=logreg.coef_[0].argsort()
    smallest=coeff[:10]
    largest=coeff[:-11:-1]
#     print('Smallest Features:',feature_names[smallest])
#     print('Largest Features:',feature_names[largest])
    
    return auc,smallest,largest


# In[180]:


answer_eleven()

