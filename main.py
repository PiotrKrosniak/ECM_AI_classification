import pandas as pd
import numpy as np
data = pd.read_csv('test_data4.csv',encoding='latin1', dtype={'SourcePath': str}, )

numpy_array = data.as_matrix()
X = numpy_array[:, 0]
Y = numpy_array[:, 1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, Y_train)

predicted = text_clf.predict(X_test)
np.mean(predicted == Y_test)
data.to_csv('file_name.csv', sep='\t', encoding='utf-8')
