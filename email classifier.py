#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string


# In[3]:


# Load the dataset
data = pd.read_csv('spam[1].csv', encoding='latin-1')

# Keep only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Encode labels: 'ham' -> 0, 'spam' -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# In[7]:


nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)


# In[5]:


vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['label']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = MultinomialNB()
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred))


# In[10]:


test_email = ["Congratulations! You've won a $1,000 Walmart gift card. Click here to claim."]
test_vectorized = vectorizer.transform(test_email).toarray()
prediction = model.predict(test_vectorized)
print("Spam" if prediction[0] == 1 else "Not Spam")


# In[11]:


import pickle

# Save the model and vectorizer
pickle.dump(model, open('spam_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot label distribution
sns.countplot(data['label'])
plt.title("Spam vs Ham")
plt.show()


# In[13]:


# Test email example
test_email = ["Congratulations! You've won a $1,000 Walmart gift card. Click here to claim."]

# Preprocess the email
test_vectorized = vectorizer.transform(test_email).toarray()

# Predict using the trained model
prediction = model.predict(test_vectorized)
print("Spam" if prediction[0] == 1 else "Not Spam")


# In[ ]:




