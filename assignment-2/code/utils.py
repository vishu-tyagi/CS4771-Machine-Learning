import re
import random

import numpy as np
import pandas as pd

import scipy
from scipy import sparse

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import normalize

from itertools import chain
from collections import Counter


def load_data(merge=False):
    train = pd.read_csv('data/reviews_tr.csv')
    test = pd.read_csv('data/reviews_te.csv')

    if merge:
        data = pd.concat([train, test])
        return data

    return train, test


class UnigramModel:

    def __init__(self):
        self.unigram_vocab_size = None
        self.unigram_vocab = None
        self.token_pattern = r"(?u)\b\w\w+\b"


    def build_vocab(self, data, unigram_threshold):

        unigrams = data.text.apply(lambda x: re.findall(self.token_pattern, x))

        counts = Counter(chain.from_iterable(unigrams.tolist()))
        unique_unigram_tokens = [token for token in counts if counts[token] > unigram_threshold]
        unique_unigram_tokens = sorted(unique_unigram_tokens)

        unigram_vocab = {}

        for i in range(len(unique_unigram_tokens)):
            unigram_vocab[unique_unigram_tokens[i]] = i

        self.unigram_vocab_size = len(unigram_vocab)
        self.unigram_vocab = unigram_vocab


    def tf(self, data, type=None):

        if type not in ['train', 'test']:
            raise Exception('input either train or test set')

        mat = sparse.lil_matrix((data.shape[0], self.unigram_vocab_size), dtype=np.float32)

        if type == 'train':
            idf = np.zeros((self.unigram_vocab_size, 1))

        reviews = data.text.apply(lambda x: re.findall(self.token_pattern, x))
        
        i = 0
        for sentence in reviews.tolist():
            freq = Counter(sentence)
            idx = [self.unigram_vocab[word] for word in freq if word in self.unigram_vocab]
            vals = [freq[word] for word in freq if word in self.unigram_vocab]
            mat[i, idx] = vals

            if type == 'train':  
                unigram_tokens = list(set(sentence))
                idx = [self.unigram_vocab[token] for token in unigram_tokens if token in self.unigram_vocab]
                idf[idx, 0] += 1

            i += 1
        
        if type == 'train':
            idf = np.log((1+data.shape[0]) / (1 + idf)) + 1
            return mat, idf

        return mat


    def tfidf(self, data, term_frequency_mat, idf_mat):

        mat = term_frequency_mat.multiply(idf_mat.reshape(-1,))
        mat = mat.tocsr()
        mat = normalize(mat, norm='l2', axis=1)

        return mat


class BigramModel:

    def __init__(self):
        self.unigram_bigram_vocab_size = None
        self.unigram_bigram_vocab = None
        self.token_pattern = r"(?u)\b\w\w+\b"


    def build_vocab(self, data, unigram_threshold, bigram_threshold):

        unigrams = data.text.apply(lambda x: re.findall(self.token_pattern, x))

        counts = Counter(chain.from_iterable(unigrams.tolist()))
        unique_unigram_tokens = [token for token in counts if counts[token] > unigram_threshold]

        bigrams = [[review[i]+' '+ review[i+1] for i in range(len(review)-1)] for review in unigrams]

        counts = Counter(chain.from_iterable(bigrams))
        unique_bigram_tokens = [token for token in counts if counts[token] > bigram_threshold]
        unique_tokens = unique_unigram_tokens + unique_bigram_tokens
        unique_tokens = sorted(unique_tokens)

        unigram_bigram_vocab = {}

        for i in range(len(unique_tokens)):
            unigram_bigram_vocab[unique_tokens[i]] = i

        self.unigram_bigram_vocab_size = len(unigram_bigram_vocab)
        self.unigram_bigram_vocab = unigram_bigram_vocab

    
    def tf(self, data, type=None):

        if type not in ['train', 'test']:
            raise Exception('input either train or test set')

        mat = sparse.lil_matrix((data.shape[0], self.unigram_bigram_vocab_size), dtype=np.float32)

        if type == 'train':
            idf = np.zeros((self.unigram_bigram_vocab_size, 1))
        
        unigrams = data.text.apply(lambda x: re.findall(self.token_pattern, x))
        
        i = 0
        for sentence in unigrams.tolist():
            bigram_tokens = [sentence[i]+' '+sentence[i+1] for i in range(len(sentence)-1)]
    
            freq = Counter(sentence)
            idx = [self.unigram_bigram_vocab[word] for word in freq if word in self.unigram_bigram_vocab]
            vals = [freq[word] for word in freq if word in self.unigram_bigram_vocab]
            mat[i, idx] = vals

            freq = Counter(bigram_tokens)
            idx = [self.unigram_bigram_vocab[word] for word in freq if word in self.unigram_bigram_vocab]
            vals = [freq[word] for word in freq if word in self.unigram_bigram_vocab]
            mat[i, idx] = vals

            if type == 'train':
                bigram_tokens = list(set(bigram_tokens))
                idx = [self.unigram_bigram_vocab[token] for token in bigram_tokens if token in self.unigram_bigram_vocab]
                idf[idx, 0] += 1
                
                unigram_tokens = list(set(sentence))
                idx = [self.unigram_bigram_vocab[token] for token in unigram_tokens if token in self.unigram_bigram_vocab]
                idf[idx, 0] += 1

            i += 1
        
        if type == 'train':
            idf = np.log((1+data.shape[0]) / (1 + idf)) + 1
            return mat, idf
                   
        return mat
    

    def tfidf(self, data, term_frequency_mat, idf_mat):

        mat = term_frequency_mat.multiply(idf_mat.reshape(-1,))
        mat = mat.tocsr()
        mat = normalize(mat, norm='l2', axis=1)

        return mat


class Perceptron:

    def __init__(self, vocab_size):
        self.w = None
        self.w_avg = None
        self.vocab_size = vocab_size


    def fit(self, target, data_representation):

        target = np.array(target)
        mat = sparse.hstack([sparse.csr_matrix(np.ones((data_representation.shape[0], 1))), data_representation], format='csr')

        iter = 1
        n_passes = 2
        self.w = np.zeros((self.vocab_size+1, 1))
        tail_weight = np.zeros((self.vocab_size+1, 1))

        for p in range(n_passes):
            index = np.arange(np.shape(mat)[0])
            np.random.shuffle(index)
            target = target[index]
            mat = mat[index, :]

            for i in range(data_representation.shape[0]):
                iter += 1

                if target[i] == 0:
                    y = -1
                else:
                    y = 1

                if y * (self.w.T @ mat[i].reshape(-1,1)) <= 0:
                    self.w += y * mat[i].reshape(-1,1)

                if iter > data_representation.shape[0]:
                    tail_weight += self.w

        avg_weight = tail_weight/(data_representation.shape[0]+1)
        self.w_avg = avg_weight


    def predict(self, data_representation):

        mat = sparse.hstack([sparse.csr_matrix(np.ones((data_representation.shape[0], 1))), data_representation], format='csr')
        pred = mat @ self.w_avg
        pred = pred.reshape(-1,)
        pred = (pred >= 0).astype(int)

        return pred


    def evaluate(self, target, data_representation):
        
        target = np.array(target)
        target = np.where(target.astype(int) == 1, 1, -1).astype(int)
        
        pred = self.predict(data_representation)
        pred = np.where(pred.astype(int) == 1, 1, -1).astype(int)

        accuracy_ = accuracy_score(y_true=target, y_pred=pred)
        precision = precision_score(y_true=target, y_pred=pred)
        recall = recall_score(y_true=target, y_pred=pred)

        return [accuracy_, precision, recall]

