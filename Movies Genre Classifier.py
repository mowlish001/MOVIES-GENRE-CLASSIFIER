#!/usr/bin/env python
# coding: utf-8

# # MOVIE GENRES CLASSIFICATION

# In[1]:


# Importing essential libraries
import numpy as np
import pandas as pd


# In[2]:


# Loading the dataset
df = pd.read_csv('kaggle_movie_train.csv')


# # **Exploring the dataset**

# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.head(10)


# In[6]:


# Importing essential libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# Visualizing the count of 'genre' column from the dataset
plt.figure(figsize=(12,12))
sns.countplot(x='genre', data=df)
plt.xlabel('Movie Genres')
plt.ylabel('Count')
plt.title('Genre Plot')
plt.show()


# # **Data Cleaning and Preprocessing**

# In[8]:


# Finding unique genres
movie_genre = list(df['genre'].unique())
movie_genre.sort()
movie_genre


# In[9]:


# Mapping the genres to values
genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
df['genre'] = df['genre'].map(genre_mapper)
df.head(10)


# In[10]:


# Finding any NaN values
df.isna().any()


# In[11]:


# Removing the 'id' column
df.drop('id', axis=1, inplace=True)
df.columns


# In[12]:


# Importing essential libraries for performing Natural Language Processing on given dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[13]:


df.shape


# In[15]:


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Assuming df is already defined
corpus = []
ps = PorterStemmer()

# Pre-compiling the regex pattern for better performance
pattern = re.compile(r'[^a-zA-Z]')

# Getting the stopwords as a set for faster lookup
stop_words = set(stopwords.words('english'))

# Cleaning the text column
def clean_text(dialog):
    # Remove special characters
    dialog = pattern.sub(' ', dialog)
    # Convert to lowercase
    dialog = dialog.lower()
    # Tokenize and remove stopwords
    words = [word for word in dialog.split() if word not in stop_words]
    # Stem the words
    words = [ps.stem(word) for word in words]
    # Join the cleaned words
    return ' '.join(words)

# Applying the cleaning function to the text column
corpus = df['text'].apply(clean_text).tolist()


# In[16]:


corpus[0:10]


# In[17]:


len(corpus)


# In[18]:


drama_words = []
for i in list(df[df['genre']==4].index):
  drama_words.append(corpus[i])

action_words = []
for i in list(df[df['genre']==1].index):
  action_words.append(corpus[i])

comedy_words = []
for i in list(df[df['genre']==3].index):
  comedy_words.append(corpus[i])

drama = ''
action = ''
comedy = ''
for i in range(0, 3):
  drama += drama_words[i]
  action += action_words[i]
  comedy += comedy_words[i]


# In[19]:


# Creating wordcloud for drama genre
from wordcloud import WordCloud
wordcloud1 = WordCloud(background_color='white', width=3000, height=2500).generate(drama)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title("Words which indicate 'DRAMA' genre ")
plt.show()


# In[20]:


# Creating wordcloud for action genre
wordcloud2 = WordCloud(background_color='white', width=3000, height=2500).generate(action)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title("Words which indicate 'ACTION' genre ")
plt.show()


# In[21]:


# Creating wordcloud for comedy genre
wordcloud3 = WordCloud(background_color='white', width=3000, height=2500).generate(comedy)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud3)
plt.axis('off')
plt.title("Words which indicate 'COMEDY' genre ")
plt.show()


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer

# Ensure corpus is a list of strings
if not isinstance(corpus, list) or not all(isinstance(doc, str) for doc in corpus):
    raise ValueError("The 'corpus' must be a list of strings.")

# Initialize the CountVectorizer with specified parameters
cv = CountVectorizer(max_features=10000, ngram_range=(1, 2), dtype=np.float32)

# Fit and transform the corpus into a sparse matrix, then convert it to a dense array
X = cv.fit_transform(corpus).toarray()


# In[25]:


y = df['genre'].values


# # **Model Building**

# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))


# ## *Multinomial Naive Bayes*

# In[27]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)


# In[28]:


# Predicting the Test set results
nb_y_pred = nb_classifier.predict(X_test)


# In[29]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
score1 = accuracy_score(y_test, nb_y_pred)
print("---- Score ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))


# In[30]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
nb_cm = confusion_matrix(y_test, nb_y_pred)


# In[31]:


nb_cm


# In[32]:


# Plotting the confusion matrix
plt.figure(figsize=(15,12))
axis_labels = ['other', 'action', 'adventure', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller']
sns.heatmap(data=nb_cm, annot=True, cmap="Blues", xticklabels=axis_labels, yticklabels=axis_labels)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for Multinomial Naive Bayes Algorithm')
plt.show()


# In[33]:


# Hyperparameter tuning the Naive Bayes Classifier
best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier = MultinomialNB(alpha=i)
  temp_classifier.fit(X_train, y_train)
  temp_y_pred = temp_classifier.predict(X_test)
  score = accuracy_score(y_test, temp_y_pred)
  print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
  if score>best_accuracy:
    best_accuracy = score
    alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[34]:


classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


# # **Predictions**

# In[35]:


def genre_prediction(sample_script):
  sample_script = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_script)
  sample_script = sample_script.lower()
  sample_script_words = sample_script.split()
  sample_script_words = [word for word in sample_script_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_script = [ps.stem(word) for word in sample_script_words]
  final_script = ' '.join(final_script)

  temp = cv.transform([final_script]).toarray()
  return classifier.predict(temp)[0]


# In[36]:


# For generating random integer
from random import randint


# In[38]:


# Loading test dataset
test = pd.read_csv('kaggle_movie_test.csv')
test.columns


# In[39]:


test.shape


# In[40]:


test.drop('id', axis=1, inplace=True)
test.head(10)


# In[41]:


# Predicting values
row = randint(0,test.shape[0]-1)
sample_script = test.text[row]

print('Script: {}'.format(sample_script))
value = genre_prediction(sample_script)
print('Prediction: {}'.format(list(genre_mapper.keys())[value]))


# In[42]:


# Predicting values
row = randint(0,test.shape[0]-1)
sample_script = test.text[row]

print('Script: {}'.format(sample_script))
value = genre_prediction(sample_script)
print('Prediction: {}'.format(list(genre_mapper.keys())[value]))


# In[43]:


# Predicting values
row = randint(0,test.shape[0]-1)
sample_script = test.text[row]

print('Script: {}'.format(sample_script))
value = genre_prediction(sample_script)
print('Prediction: {}'.format(list(genre_mapper.keys())[value]))


# In[44]:


# Predicting values
row = randint(0,test.shape[0]-1)
sample_script = test.text[row]

print('Script: {}'.format(sample_script))
value = genre_prediction(sample_script)
print('Prediction: {}'.format(list(genre_mapper.keys())[value]))


# In[45]:


# Predicting values
row = randint(0,test.shape[0]-1)
sample_script = test.text[row]

print('Script: {}'.format(sample_script))
value = genre_prediction(sample_script)
print('Prediction: {}'.format(list(genre_mapper.keys())[value]))


# In[47]:


# Assuming 'genre_prediction' function and 'genre_mapper' dictionary are defined
# Example genre_mapper: {0: 'Drama', 1: 'Comedy', 2: 'Action'}

# Take user input for text summary
sample_script = input("Enter the text summary: ")

# Display the input script
print("\nScript: {}".format(sample_script))

# Predict the genre
value = genre_prediction(sample_script)

# Map the numeric prediction to a genre
predicted_genre = genre_mapper.get(value, "Unknown")

# Display the prediction
print("Prediction: {}".format(predicted_genre))


# In[ ]:




