from sklearn.datasets import fetch_20newsgroups


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

twenty_train = fetch_20newsgroups(subset='train', shuffle=True,categories=categories)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True, categories=categories)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)