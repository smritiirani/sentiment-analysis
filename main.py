import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

data = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\Assignment Data.csv')
df = data.copy()


print(df.dtypes)

df=df.astype(str,errors = 'raise')
print(df.dtypes)

missing_values = ["NaN"]
df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\Assignment Data.csv', na_values = missing_values)
print(df.isnull().sum())

median = df['price'].median()
df['price'].fillna(median, inplace=True)
print(median)
print(df[['price']].isnull().sum())

df = df.fillna(df.mode().iloc[0])
print(df.isnull().sum())

print(df)
print(df.shape)



dfs = pd.DataFrame(data = df , columns=['description','variety'])
df = dfs.copy()
print(df)



print(df['variety'].unique())
print(df['description'].unique())



df['description	'] = df.drop(['variety'], axis=1)
X = df['description	']
df['variety'] = df['variety']
y = df['variety']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0,shuffle=True )
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)



# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
print(X_train_tfidf)

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)
# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, y_train)
print(text_clf)

import numpy as np
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))

from sklearn.metrics import accuracy_score
accuracy_dt =accuracy_score(y_test,predicted)
print("Accuracy: {}".format(accuracy_dt))

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])
text_clf_svm = text_clf_svm.fit(X_train,y_train)
print(text_clf_svm)

predicted_svm = text_clf_svm.predict(X_test)
accuracy_svc = np.mean(predicted_svm == y_test)
print(accuracy_svc)



import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
# Stemming Code
# import nltk
# nltk.download()

# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(X_train, y_train)
predicted_mnb_stemmed = text_mnb_stemmed.predict(X_test)
accury =np.mean(predicted_mnb_stemmed == y_test)
print(accury)

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])


text_clf_svm = text_clf_svm.fit(X_train,y_train)
print(text_clf_svm)

predicted_svm = text_clf_svm.predict(X_test)
accuracy_1 =np.mean(predicted_svm == y_test)
print(accuracy_1)

comparison_dict={"Algorithm":["Naive Bayes Algorithm","support vector machine"],
                 "Accuracy":[accuracy_dt,accuracy_svc]}


comparison = pd.DataFrame(comparison_dict)
print(comparison.sort_values([ 'Accuracy'], ascending=False))

#After removing stop words comperison
comparison_dict={"Algorithm":["Naive Bayes Algorithm","support vector machine"],
                 "Accuracy":[accury,accuracy_1]}


comparison = pd.DataFrame(comparison_dict)
print(comparison.sort_values([ 'Accuracy'], ascending=False))



















