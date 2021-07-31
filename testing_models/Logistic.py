from extract_split_data import sets
from extract_split_data import vectorized_set

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

def train_logistic():

    vectorizer, X_grav_train, X_grav_test, X_ses_train, X_ses_test, grav_train, grav_test, ses_train, ses_test = vectorized_set()

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_grav_train, grav_train)

    classifier2 = LogisticRegression(max_iter=500)
    classifier2.fit(X_ses_train, ses_train)

    score = classifier.score(X_grav_test, grav_test)
    score2 = classifier2.score(X_ses_test, ses_test)

    print('Precisión para datos de gravedad: {:.4f}'.format(score))
    print('Precisión para datos de sesgo: {:.4f}'.format(score2))

    plt.hist(classifier.predict(X_grav_test)[0]-grav_test,[-3,-2,-1,0,1,2,3,4],align= 'mid', rwidth=0.8)
    plt.show()

    return vectorizer, classifier, classifier2

def modelo(pers_test):
    vectorizer, classifier, classifier2 = train_logistic()
    test = vectorizer.transform([pers_test])

    gravedad = classifier.predict(test)
    ses = classifier2.predict(test)

    print("La noticia tiene gravedad {} y sesgo {}.".format(*gravedad,*ses))
