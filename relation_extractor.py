import glob
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pdb
import ujson
import time

# CONSTANTS
TERM_DISTANCE = 5
WND_SIZE = TERM_DISTANCE * 2
MIN_COOCC = 10
TOP_N = 20

# Create Co-occurrence tuple for wnd[-1] with all other wnd[i]
# Get the expansion term for a word.
def get_expan_terms(aword):
    global MIN_COOCC
    return [rw for rw, fw in cfd[aword].most_common(TOP_N) if fw > MIN_COOCC]


def add_conditional_frequence_table(wnd):
    global cfd
    new_term = wnd[-1]
    for term in wnd[-TERM_DISTANCE:-1]:
        if term != new_term:
            cfd[term][new_term] += 1
            cfd[new_term][term] = cfd[term][new_term]


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
        # Filter by stopwords
        tokens_norm = [ps.stem(t.lower()) for t in tokens if t.isalpha() and (t.lower() not in stop)]

        # Tokes neighbors window
        wnd = []
        for t in tokens_norm:
            wnd.append(t)
            wnd = wnd[-WND_SIZE:]
            # Add to conditional frequency table
            add_conditional_frequence_table(wnd)

print("Time1: {}".format(time.time()-start_time))

cfd_mini = nltk.ConditionalFreqDist()
for w in cfd:
    top_list = cfd[w].most_common(TOP_N)
    cfd_mini[w] = dict([(w, f) for w, f in top_list if f > MIN_COOCC])

print("Time2: {}".format(time.time()-start_time))

ujson.dump(cfd_mini, open("/Users/jason.wu/Downloads/ap_cfd_dis{}_min{}_top{}_stm.json".format(
    TERM_DISTANCE, MIN_COOCC, TOP_N), "w"))

print("Time3: {}".format(time.time()-start_time))

pdb.set_trace()
