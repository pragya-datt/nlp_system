from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import operator

#
df = pd.read_csv(r"Reviews.csv")
print(df.head())
# cleaning the data
# checking if any Nan values are present in the dataset
# clearing them
df = df.dropna()
print(df.isnull().sum())
print("Since all values are 0, you are good to go :)!!!")


df['new_ratings'] = np.where(df['Score'] > 3, 1, 0)

X = df['Text']
y = df['new_ratings']
# print(y)


# perform the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# print(X_train.head())
# print(X_train.shape)

# extracting the feature vectors and their term frequency-inverse document frequency
tf_idf = TfidfVectorizer()
X_train_tfidf = tf_idf.fit_transform(X_train)
print(X_train_tfidf)
print(X_train_tfidf.shape)


# defining the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# Pipeline
text_clf = Pipeline([
    ('tf_idf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)
print(predictions)
acc = metrics.accuracy_score(y_test, predictions)
# print(acc)
print(classification_report(y_test, predictions))


# f = open('sentiment_analysis_opinion_mining_model.dat', 'rb')
f = open('sentiment_analysis_model.dat', 'rb')
text_clf = pickle.load(f)
# print(X_train)
predictions = text_clf.predict(X_test)
acc = metrics.accuracy_score(y_test, predictions)
print("Accuracy of the model is: ", acc*100, "%")
print("Predictions for the training data set : ")
print(predictions)
print("Classification report : ")
print(classification_report(y_test, predictions))


test_data = (["The product purchased was an absolute waste of money. Its superbly pathetic.", "Its a really good product. Really enjoyed using it."])
print(text_clf.predict(test_data))
f.close()

# -------------------------------Opinion Mining--------------------


print("------------------------------------------------------------------------------------------")

rds = pd.read_csv(r"Amazon_Unlocked_Mobile.csv")
rds = rds.drop(['Brand Name', 'Price', 'Rating', 'Review Votes'], axis=1)

rds = rds[pd.notnull(rds['Reviews'])]
rds = rds[0:100000]
d = {}

for i in rds['Product Name']:
    if i not in d.keys():
        d[i] = (rds.loc[rds['Product Name'] == i]['Reviews']).to_list()


tot_positive = list(text_clf.predict(rds['Reviews'])).count(1)


def myKey(e):
    return e[1]

def recommend(d):
    dictionary = {}
    for i in d:
        lreviews = d[i]
        predictions = list(text_clf.predict(lreviews))
        score = (predictions.count(1)/tot_positive)*100
        # print(i, " : ", predictions, score)
        dictionary[i] = score
#
    l = list(dictionary.items())
    l.sort(reverse=True, key=myKey)
    return l


print("Top 5 Recommended phones for you : ")
print(recommend(d)[0:5])
recommend(d)