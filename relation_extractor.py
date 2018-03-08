import glob
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


# Create Co-occurrence tuple for wnd[-1] with all other wnd[i]
def create_cooccurrence_pair(wnd):
    return [tuple([t, wnd[-1]]) for t in wnd[int(WND_SIZE / 2):-1] if t != wnd[-1]]


# Build up two sides relations, simple add the reverted pair for each existing.
def double_rel(list):
    for pair in list:
        yield pair
        yield pair[::-1]


# Get the expansion term for a word.
def get_expan_terms(aword):
    return [rw for rw, fw in cfd[aword].most_common(10)]


if len(sys.argv) > 1:
    # Define the data path
    data_path = sys.argv[1]


list_of_file = sorted(glob.glob(data_path))
list_of_rels = []
WND_SIZE = 10
MIN_COOCC = 10

stop = set(stopwords.words('english'))
ps = PorterStemmer()

for fname in list_of_file:
    with open(fname, encoding='utf-8') as file:
        raw = file.read()
        # Extract all the <TEXT> field
        result = re.findall(r'<TEXT>(.*?)</TEXT>', raw, re.DOTALL)
        texts = ''.join(result)
        # Tokenize
        tokens = word_tokenize(texts)
        # Filter Tokens is alphabetical and keep the in lower case
        tokens = [t.lower() for t in tokens if t.isalpha()]
        # Filter by stopwords and stemming
        tokens_norm = [t for t in tokens if t not in stop]
        # Tokes neighbors window
        wnd = []
        for t in tokens_norm:
            wnd.append(t)
            if len(wnd) > WND_SIZE:
                wnd.pop(0)
            new_rels = create_cooccurrence_pair(wnd)
            list_of_rels += new_rels
    # Build up two sides relations, simple add the reverted pair for each existing.
    all_rels = [x for x in double_rel(list_of_rels)]
    cfd = nltk.ConditionalFreqDist(all_rels)
    # Filter the cooccurrence, remove the COOCC <= MIN_COOCC
    for w in cfd:
        fd = cfd[w]
        for ww in list(fd):
            if fd[ww] <= MIN_COOCC:
                fd.pop(ww)

print(get_expan_terms("trade"))
