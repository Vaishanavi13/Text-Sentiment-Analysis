#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Tokenizing of text
import nltk
#tokenize into sentences
from nltk.tokenize import sent_tokenize
text="""In today’s scenario, one way of people’s success is identified by how they are communicating and sharing information with others. That’s where the concepts of language come into the picture. However, there are many languages in the world. Each has many standards and alphabets, and the combination of these words arranged meaningfully resulted in the formation of a sentence. Each language has its own rules while developing these sentences, and these sets of rules are also known as grammar."""
tokenized_text = sent_tokenize(text)
print(tokenized_text)

#tokenize into words
from nltk.tokenize import word_tokenize
tokenized_words = word_tokenize(text)
print(tokenize_words)


#Frequency of occurence of words
from nltk.probability import FreqDist
fdist = FreqDist()
for word in tokenized_words:
    fdist[word.lower()] += 1
print("Frequencies: \n", fdist)

fdist_top10 = fdist.most_common(10)
print("Top 10 most common words: \n", fdist_top10)


# In[ ]:


#Noise removal
#First, removing stop-words
from nltk.corpus import stopwords
stop_word = set(stopwords.words("english"))
print("Stopwords: ", stop_words)
filtered_sent = []
for w in tokenized_word:
    if w not in stop_word:
        filtered_sent.append(w)
print("Filtered Sentences: \n", filtered_sent)

#Lexical Normalisation or Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_words = []
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("Stemmed Sentence: \n", stemmed_words)


# In[ ]:


#Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
lem = WordNetLemmatizer()
print("Lemmatized words: \n", lem.lemmatize(filtered_sent,"v"))

#Tagging
print(nltk.pos_tag(text))


# In[3]:


#Sentiment Analysis using BOW method
#text classification into sentiments
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\HP\Downloads\train\train.tsv', sep = "\t")
print(data.head())
print(data.info())

print(data.Sentiment.value_counts())


# In[5]:


#Visualizing data
Sentiment_counts = data.groupby('Sentiment').count()
plt.bar(Sentiment_counts.index.values, Sentiment_count['Phrase'])
plt.xlabel("Review Sentiment")
plt.ylabel("Number of Reviews")
plt.show()


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-z0-9]+')
cv = CountVectorizer(lowercase = True, stop_words = "english", ngram_range = (1,1), tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Phrase'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size = 0.3, random_state = 1)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB().fit(x_train,y_train)
predicted = clf.predict(x_test)
print("MultinomialNB Accuracy: ", metrics.accuracy_score(y_test,predicted))


# In[9]:


#Sentiment Analysis using tfidf method
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
text_tf = tf.fit_transform(data['Phrase'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size = 0.3, random_state = 1)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB().fit(x_train,y_train)
predicted = clf.predict(x_test)
print("MultinomialNB Accuracy: ", metrics.accuracy_score(y_test,predicted))

