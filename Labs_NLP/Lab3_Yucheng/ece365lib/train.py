import nltk
import copy
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from ece365lib import evaluate


# deliverable 1.1
def tokenize_corpus(corpus):
    '''
    Returns the tokenized version of the nltk corpus string.
    
    :param corpus: str, corpus
    :returns: tokenized version of corpus
    :rtype: list of list of strings
    
    Hint: use nltk.tokenize.sent_tokenize, and nltk.tokenize.word_tokenize
    '''
    tokenized_sentences = []
    sentences = sent_tokenize(corpus)
    for sentence in sentences:
        if len(sentence.strip()) > 0:
            words = word_tokenize(sentence)
            tokenized_sentences.append(words)
    return tokenized_sentences
    

# deliverable 1.2
def pad_corpus(corpus):
    '''
    Returns a padded version of the tokenized corpus.
    
    :param corpus: list of list of str, tokenized corpus.
    :returns: padded version of the tokenized corpus.
    :rtype: list of list of strings
    '''
    start_symbol = '<s>'
    end_symbol = '</s>'
    pd_corpus = copy.deepcopy(corpus)
    for pd_sentence in pd_corpus:
        pd_sentence.append(end_symbol)
        pd_sentence.insert(0, start_symbol)
    return pd_corpus
    

# deliverable 1.3    
def split_corpus(corpus):
    '''
    Splits the input corpus into a train and test corpus based on a 80-20 split.
    
    :param corpus: list of list of str, padded tokenized corpus.
    :returns: train subset of the corpus.
    :returns: test subset of the corpus.
    :rtype: list of list of strings, list of list of strings
    '''
    split = math.floor(len(corpus) * 0.8)
    train = corpus[0:split]
    test = corpus[split:]
    return train, test
    

# deliverable 1.4    
def count_ngrams(corpus, n=3):
    '''
    Takes in a corpus and counts the frequency of all unique n-grams (1-grams, 2-grams, ..., up to length n), and stores them in a dictionary. It also returns a list of all unique words (vocab).
    
    :param corpus: list of list of str, padded tokenized training corpus.
    :param n: maximum order of n-grams considered.
    :returns: dictionary of count of n-grams. Keys are n-grams (tuples), and values are their frequency in the corpus.
    :returns: list of vocab words
    :rtype: dictionary, list of strings
    '''
    
    ngrams = defaultdict(float)
    vocab = []
    for sentence in corpus:
        for i in range(len(sentence)):
            if sentence[i] not in vocab: vocab.append(sentence[i])
            a = []
            for j in range(n):
                if (i+j < len(sentence)):
                    a.append(sentence[i+j])
                    key = tuple(a)
                    keys = ngrams.keys()
                    if key in keys:
                        ngrams[key] = ngrams[key] + 1
                    else: ngrams[key] = 1      
    return ngrams, vocab
    

# deliverable 1.5    
def estimate(counts, word, context):
    '''
    Estimates the n-gram probability of a word [w_i] following a context of size n-1.
    
    :param counts: a dictionary of n-gram counts.
    :param word: a list of one word, [w_i]
    :param context: a list of preceding n-1 words in order
    :returns: probability of the n-gram.
    :rtype: float.
    '''
    b = tuple(context)
    words = []
    for i in context:
        words.append(i)
    for j in word:
        words.append(j)
    a = tuple(words)
    prob = counts[a]/counts[b]
    return prob
    

# deliverable 3.1    
def vary_ngram(train_corpus, test_corpus, n_gram_orders):
    '''
    Returns a dictionary of perplexity values at different order n-gram LMs
    
    :param train_corpus: list of list of str, corpus to train language model on.
    :param test_corpus: list of list of str, corpus to test language model on.
    :n_gram_orders: list of ints, orders of n-grams desired.
    :returns: a dictionary of perplexities at different orders, key=order, value=perplexity.
    :rtype: dict.
    
    Hint: Follow the same LM training procedure as in the notebook in the end of Exercise 1. 
    '''
    perplexity = defaultdict(float)
    for n in n_gram_orders:
        train, vocab = padded_everygram_pipeline(n, train_corpus[:])
        test = sum([['<s>'] + x + ['</s>'] for x in test_corpus[:]],[])
        lm = Laplace(n)
        lm.fit(train, vocab)
        pp = evaluate.get_perplexity(lm, test)
        perplexity[n] = pp
    return perplexity
            
    
    
    
    
    
