#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


dataset = pd.read_csv("twitter.csv")


# In[4]:


dataset


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset.info()


# In[9]:


dataset.describe()


# In[10]:


dataset["labels"] = dataset["class"].map({0:"Hate Speech",
                                          1:"Offensive Language",
                                          2:"No hate or offensive language"})


# In[11]:


dataset


# In[12]:


data = dataset[["tweet","labels"]]


# In[13]:


data


# In[22]:


import re
import nltk
import string
nltk.download('stopwords')


# In[19]:


#importing stop words 
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))


# In[25]:


pip install --upgrade nltk


# In[27]:


pip install snowballstemmer


# In[28]:


from nltk.stem import SnowballStemmer


# In[34]:


#import stemming
stemmer = nltk.SnowballStemmer("english")


# In[40]:


#data cleaning 
def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.S+','',text)
    text = re.sub('\[.*?\]','',text)
    text = re.sub('<,*?>+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    #stop words removal
    text = [word for word in text.split(' ')if word not in stopwords]
    text = " ".join(text)
    #stemming the text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


# In[41]:


data["tweet"] = data["tweet"].apply(clean_data)


# In[42]:


data


# In[43]:


x = np.array(data["tweet"])
y = np.array(data["labels"])


# In[44]:


x


# In[ ]:





# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[48]:


CV = CountVectorizer()
x = CV.fit_transform(x)


# In[49]:


x


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=42)


# In[51]:


x_train


# In[52]:


#building out ml model
from sklearn.tree import DecisionTreeClassifier


# In[53]:


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[54]:


y_pred = dt.predict(x_test)


# In[55]:


#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[56]:


import seaborn as sns
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlGnBu")



# In[62]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[63]:


sample = "Let's unite and kill all people who are protesting against the government"
sample = clean_data(sample)


# In[64]:


sample


# In[67]:


data1 = CV.transform([sample]).toarray()


# In[68]:


data1


# In[69]:


dt.predict(data1)


# In[ ]:




