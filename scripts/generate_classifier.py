import string
from collections import Counter

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

msg = ("`rank` is deprecated; use the `ndim` attribute or function instead. "
       "To find the rank of a matrix see `numpy.linalg.matrix_rank`.")
import warnings
warnings.filterwarnings("ignore", message=msg)

exclude = set(string.punctuation)

def clean(s):
    colon = s.find(':')
    if colon == -1:
        return ('','')
    speaker = s.split(':')[0]
    s = s[colon:]
    return (''.join(ch for ch in s if ch not in exclude).lower(), speaker)

def get_data(min_lines):
    lines = [clean(line.rstrip('\n')) for line in open('data/quotes.txt')]
    speakers = Counter([p[1] for p in lines])
    common = set([x for x in speakers if speakers[x] > min_lines])
    print 'Considering %d characters' % len(common)
    filtered = [l for l in lines if l[1] in common]
    return train_test_split(filtered, test_size=0.2)

def learn(clf, data):
    train, test = data
    train_features = [p[0] for p in train]
    train_labels = [p[1] for p in train]
    test_features = [p[0] for p in test]
    test_labels = [p[1] for p in test]

    vectorizer = TfidfVectorizer(max_df = 0.003,
                                 ngram_range=(1,3))
    train_vector = vectorizer.fit_transform(train_features)
    test_vector = vectorizer.transform(test_features)

    clf.fit(train_vector, train_labels)
    predictions = clf.predict(test_vector)
    acc = accuracy_score(test_labels, predictions)
    print 'accuracy: %f' % acc

    return clf, vectorizer

if __name__ == "__main__":
    data = get_data(100)
    clf = svm.SVC(kernel='linear')
    clf, v = learn(clf, data)
    joblib.dump(clf, 'pickles/classifier.pkl')
    joblib.dump(v, 'pickles/vectorizer.pkl')
