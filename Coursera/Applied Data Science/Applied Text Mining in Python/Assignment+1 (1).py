
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[135]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)

date_sorter()


# In[96]:


def date_sorter():



    df1=df.str.extractall(r'((\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2}))')
    def insert19(row):
        if len(row)==2:
            row='19'+row
        return row
    df1[3]=df1[3].apply(insert19)


    df1[4]=df1[3]+'/'+df1[1]+'/'+df1[2]

    df1=df1.loc[df1.index.get_level_values(1)!=1]
    df1[4]=pd.to_datetime(df1[4])

    df2=df.str.extractall(r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?[\s-](\d{2}),?[\s-](\d{2,4}))')
    df2[0]=pd.to_datetime(df2[0])


    df3=df.str.extractall(r'((\d{2})\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?,?\s(\d{4}))')
    df3[0]=pd.to_datetime(df3[0])

    df4=df.str.extractall(r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{2})[a-zA-Z]{2},\s(\d{4}))')
    df4

    df5=df.str.extractall(r'(?<!\d\d\s)((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*,?\s(\d{4}))')

    df5[0]=df5[0].str.replace('Janaury','January')
    df5[0]=df5[0].str.replace('Decemeber','December')
    df5[0]=pd.to_datetime(df5[0])
    df6=df.str.extractall(r'(\s|^|~|e|\(|-|n)((\d{1,2})/(\d{4}))')
    df6



    df7=df.loc[df.index.get_level_values(0)>454]
    df7=df7.str.extractall(r'((19|20)\d{2})')

    dff=pd.concat([df1[4],df3[0],df2[0],df5[0],df6[1],df7[0]])


    dff=dff.reset_index()
    dff=dff.set_index(0).sort_index().reset_index()['level_0']
    return dff

