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
import math


# Create Co-occurrence tuple for wnd[-1] with all other wnd[i]
# Get the expansion term for a word.
def get_expan_terms(aword):
    return [rw for rw, fw in cfd[aword].most_common(TOP_N)]


def get_exapn_for_query(query_text):
    qt = nltk.word_tokenize(query_text)
    expans = {}
    for query_term in qt:
        expan_candidats = get_expan_terms(query_term)
        list_scores = {ex_term: cfd[term_i][term_j] for ex_term in expan_candidats}
        expans.update(list_scores)
    # Get the most TOP_N from expans
    expans= dict(sorted(expans.items(), key=lambda x: x[1], reverse=True)[:TOP_N])
    return expans

# Calculate P( term_j | term_i )
# For the cooccurrence,
# term_i and term_j is the two terms in the corpus
#
#                                  #coocc(term_i, term_j)
# P( term_j | term_i )  =     -----------------------------
#                            sum_k  #coocc(term_i, term_k)
#
#                                  P( term_i, term_j)
#  PMI( term_i, term_j) =  log10 ---------------------------
#                                  P( term_i) P(term_j)
#
#                                  #coocc(term_i, term_j) x number_of_all_the_cooccurences
#                       =  log10 ---------------------------------------------------------------
#                                  sum_k  #coocc(term_i, term_k) x sum_k  #coocc(term_j, term_k)
#
def score_term_in_term(term_j, term_i, cfd_N):
    global cfd
    if PMI_FLAG:
        pmi = math.log10(cfd[term_i][term_j]*cfd_N / (cfd[term_i].N()*cfd[term_j].N()))
        r = pmi
    else:
        p_term_j_in_term_i = cfd[term_i][term_j] / cfd[term_i].N()
        r = p_term_j_in_term_i
    return r


# Indri va faire ça, on ne fait pas le calcul
# P(term_j | Q)
#    =      lambda * P_ml(term_j | Query) +
#             (1-lambda)* sum{ P( term_j | term_i) * P_ml( term_i | Query) }
#    =      l * frequency_in_query(term_j)/length(Query) +
#              (1-l)* sum_{i}{ score_term_term(term_i, term_j) * frequency_in_query(term_i)/length(Query) }
#
# def score_term_in_query(term_j, qt_list, l=0.5):
#     fd = nltk.FreqDist(qt_list)
#     # If term_j is not in the fd, fd[term_j] equals 0
#     r = l * fd[term_j] / len(qt_list) + \
#         (1-l) * sum([cfd[term_i][term_j] * fd[term_i]/len(qt_list) for term_i in qt_list])
#     return r


def add_conditional_frequence_table(wnd):
    global cfd
    new_term = wnd[-1]
    for term in wnd[-TERM_DISTANCE:-1]:
        if term != new_term:
            cfd[term][new_term] += 1
            cfd[new_term][term] = cfd[term][new_term]


# Read the cfd.json file
def reload_cfd_json(fname):
    global cfd
    cfd_list = ujson.load(open(fname))
    cfd = nltk.ConditionalFreqDist()
    for w in cfd_list:
        cfd[w] = nltk.FreqDist(cfd_list[w])
    return cfd


def extract_cooccurence():
    global cfd
    if len(sys.argv) > 1:
        # Define the data path
        data_path = sys.argv[1]
    start_time = time.time()

    list_of_file = sorted(glob.glob(data_path))
    cfd = nltk.ConditionalFreqDist()
    list_freq = nltk.FreqDist()

    stop = set(stopwords.words('english'))
    if not STOP_FLAG:
        stop = []
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
            tokens_norm = [t.lower() for t in tokens if t.isalpha() and (t.lower() not in stop)]

            # Count the Frequency for each word
            for w in tokens_norm:
                list_freq[w] += 1

            # Tokes neighbors window
            wnd = []
            for t in tokens_norm:
                wnd.append(t)
                wnd = wnd[-WND_SIZE:]
                # Add to conditional frequency table
                add_conditional_frequence_table(wnd)

    print("Time1: {}".format(time.time() - start_time))

    cfd_filter = nltk.ConditionalFreqDist()
    # Filter the MIN_COOCC and Calculate the score

    # Calculate cfd.N()
    cfd_N = cfd.N()
    for term_i in cfd:
        cfd_filter[term_i] = nltk.FreqDist({term_j: score_term_in_term(term_j, term_i, cfd_N)
                                            for term_j in cfd[term_i] if cfd[term_i][term_j] > MIN_COOCC})

    cfd_topn = nltk.ConditionalFreqDist()
    # Get the TOP N
    for w in cfd_filter:
        cfd_topn[w] = nltk.FreqDist(dict(cfd_filter[w].most_common(DOUBLE_TOP_N)))

    print("Time2: {}".format(time.time() - start_time))

    file_tag = {
        'dist': '_dist'+str(TERM_DISTANCE),
        'min': '_min'+str(MIN_COOCC),
        'top': '_top'+str(TOP_N),
        'stop': '_stp' if STOP_FLAG else '',
        'pmi': '_pmi' if PMI_FLAG else ''
    }

    ujson.dump(cfd_topn, open("/Users/jason.wu/Downloads/ap_cfd{dist}{min}{top}{stop}{pmi}.json".format(
        **file_tag), "w"), double_precision=3)

    print("Time3: {}".format(time.time() - start_time))
    pdb.set_trace()
    return cfd_topn


# CONSTANTS
TERM_DISTANCE = 5
WND_SIZE = TERM_DISTANCE * 2
MIN_COOCC = 10
TOP_N = 10
DOUBLE_TOP_N = TOP_N * 2
PMI_FLAG = True
STOP_FLAG = True

# GLOBALS
cfd = nltk.ConditionalFreqDist()


if __name__ == "__main__":
    extract_cooccurence()



