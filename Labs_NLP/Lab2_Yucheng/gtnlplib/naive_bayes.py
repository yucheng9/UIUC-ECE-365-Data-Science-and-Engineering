from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation
import numpy as np
import math
from math import log
from collections import Counter
from collections import defaultdict

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for i in range(len(y)):
        if y[i] == label:
            counter = x[i]
            words_in_counter = counter.keys()
            current_corpus_keys = corpus_counts.keys()
            for word in words_in_counter:
                if word not in current_corpus_keys:
                    corpus_counts[word] = counter[word]
                else:
                    corpus_counts[word] += counter[word]
    return corpus_counts

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    V = len(vocab)
    defaultvalue = log(1/V)
    corpus_counts = defaultdict(float)
    corpus_counts_log_prob = defaultdict(lambda:defaultvalue)
    total_words_of_label = 0
    for i in range(len(y)):
        if y[i] == label:
            counter = x[i]
            words_in_counter = counter.keys()
            current_corpus_keys = corpus_counts.keys()
            for word in words_in_counter:
                total_words_of_label += counter[word]
                if word not in current_corpus_keys:
                    corpus_counts[word] = counter[word]
                else:
                    corpus_counts[word] += counter[word]
    words = corpus_counts.keys()
    for word in vocab:
        log_prob = log((corpus_counts[word]+smoothing)/(total_words_of_label+V*smoothing))
        corpus_counts_log_prob[word] = log_prob
    return corpus_counts_log_prob

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    list_of_smoothed_log_probability = Counter()
    weights = Counter()
    vocab = []
    for item in x:
        words = list(item.keys())
        for word in words:
            if word not in vocab:vocab.append(word)
    for label in labels:
        list_of_smoothed_log_probability[label] = estimate_pxy(x,y,label,smoothing,vocab)
    for label in labels:
        for word in vocab:
                weights[(label, word)] = (list_of_smoothed_log_probability[label])[word]
    for label in labels:
        doc_counts[label] = 0
    for i in range(len(y)):
        label = y[i]
        doc_counts[label] += 1
    for label in labels:
        weights[(label, OFFSET)] = log(doc_counts[label]/len(y))
    return weights

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    scores = {}
    labels_predict = list(set(y_dv))
    labels_train = list(set(y_tr))
    labels = labels_predict + labels_train
    best_smoother = smoothers[0]
    best_accuracy = 0
    current_accuracy = 0
    for i in range(len(smoothers)):
        theta_nb = estimate_nb(x_tr, y_tr, smoothers[i])
        y_hat = clf_base.predict_all(x_dv, theta_nb, labels)
        current_accuracy = evaluation.acc(y_hat, y_dv)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_smoother = smoothers[i]
        scores[smoothers[i]] = current_accuracy
    return best_smoother, scores







