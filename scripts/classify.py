import string

from sklearn.externals import joblib

msg = ("`rank` is deprecated; use the `ndim` attribute or function instead. "
       "To find the rank of a matrix see `numpy.linalg.matrix_rank`.")
import warnings
warnings.filterwarnings("ignore", message=msg)
# [''.join()]
lines = [(line.rstrip('\n')) for line in open('plan.txt')]

clf = joblib.load('pickles/classifier.pkl')
v = joblib.load('pickles/vectorizer.pkl')

vector = v.transform(lines)
prediction = clf.predict(vector)

print prediction