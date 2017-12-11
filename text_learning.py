
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')


sw = stopwords.words("english")
print sw
sw_set = set(sw)
print sw_set
print len(sw)
print len(sw_set)
