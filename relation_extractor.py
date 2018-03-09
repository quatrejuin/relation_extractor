import glob
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pdb
import pickle
import time

# CONSTANTS
WND_SIZE = 10
HALF_WND_SIZE = int(WND_SIZE/2)
MIN_COOCC = 10
TOP_N = 10

# Create Co-occurrence tuple for wnd[-1] with all other wnd[i]
# Get the expansion term for a word.
def get_expan_terms(aword):
    global MIN_COOCC
    return [rw for rw, fw in cfd[aword].most_common(TOP_N) if fw > MIN_COOCC]


def add_conditional_frequence_table(wnd):
    global cfd
    for term in wnd[HALF_WND_SIZE:-1]:
        new_term = wnd[-1]
        if term != new_term:
            cfd[term][new_term] += 1
            cfd[new_term][term] += 1


if len(sys.argv) > 1:
    # Define the data path
    data_path = sys.argv[1]

start_time = time.time()

list_of_file = sorted(glob.glob(data_path))
cfd = nltk.ConditionalFreqDist()

stop = set(stopwords.words('english'))
ps = PorterStemmer()

for index, fname in enumerate(list_of_file):
    print("No.{} File: {}".format(index, fname))
    with open(fname, encoding='latin') as file:
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
            wnd = wnd[:WND_SIZE]
            # Add to conditional frequence table
            add_conditional_frequence_table(wnd)

pickle.dump(cfd, open("/Users/jason.wu/Downloads/ap_extract_relation_dump_cfd", "wb"))
print("Time: {}".format(time.time()-start_time))

pdb.set_trace()
