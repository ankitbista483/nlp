from os import sep, spawnl
from nltk import corpus
from nltk.corpus.reader import reviews
import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep = '\t', names=["label", "message"])
#print(messages)
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub("[^a-zA-Z]", " ", messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#print(corpus)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 2500)
x = cv.fit_transform(corpus).toarray()
#print(x)

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state= 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)
#print(y_pred)
#print(y_test)
#from sklearn.metrics import confusion_matrix
#confusion_m = confusion_matrix(y_test, y_pred)
#print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
print(accuracy*100)