from collections import Counter
import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    words = text.split()
    freqs = {}
    for item in words: 
        if (item in freqs): 
            freqs[item] += 1
        else: 
            freqs[item] = 1
    return Counter(freqs)

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''
    counts = Counter()
    # YOUR CODE GOES HERE
    counts = sum(bags_of_words, Counter())
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    bow1words = bow1.keys()
    bow2words = bow2.keys()
    return bow1words - bow2words

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    pruned_data = []
    vocab = []
    keys = training_counts.keys()
    for item in keys:
        if training_counts[item] >= min_counts:
            vocab.append(item)
    for counter in target_data:
        pruned = Counter()
        keys = counter.keys()
        for item in keys:
            if training_counts[item] >= min_counts:
                pruned[item] = counter[item]
        pruned_data.append(pruned)
    return pruned_data, vocab

# deliverable 4.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)
    matrix_bags_of_words = np.zeros(shape=(len(bags_of_words), len(vocab)), dtype=float)
    for i in range(len(bags_of_words)):
        counter = bags_of_words[i]
        words = counter.keys()
        for word in words:
            j = vocab.index(word)
            matrix_bags_of_words[i, j] = counter[word]
    return matrix_bags_of_words




### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
