from __future__ import division
import numpy
from funktional.layer import Layer, Dense, OneHot, Dropout, WithDropout
from funktional.util import softmax, tanh, grouper, autoassign, CrossEntropy, Adam
import funktional.util as util
import theano
import theano.tensor as T
import random
from models import Imaginet, predictor_r
import json
from tokens import tokenize
import cPickle as pickle
import gzip
import os
import sys
import funktional.context as context
from collections import Counter
import itertools
import sklearn.metrics as metrics

class Classify(Layer):

    """3-layer classifier with dropout."""
    def __init__(self, size_repr, size_hidden=200, size_classify=3, activation=tanh, dropout=0.0):
        autoassign(locals())
        self.Dropout = Dropout(prob=self.dropout)
        self.L1 = WithDropout(Dense(self.size_repr * 2, self.size_hidden), prob=dropout)
        self.L2 = WithDropout(Dense(self.size_hidden, self.size_hidden), prob=dropout)
        self.L3 = WithDropout(Dense(self.size_hidden, self.size_hidden), prob=dropout)
        self.classify = Dense(self.size_hidden, self.size_classify)
        self.params = util.params(self.Dropout, self.L1, self.L2, self.L3, self.classify)
        #self.names  = ###

    def __call__(self, premise, hypo):
        inp = T.concatenate([premise, hypo], axis=1) # Which axis?
        return softmax(self.classify(
            self.activation(self.L3(
            self.activation(self.L2(
            self.activation(self.L1(self.Dropout(inp)))))))))

class LinearClassify(Layer):

    """Linear classifier with dropout."""
    def __init__(self, size_repr, size_classify=3, dropout=0.0):
        autoassign(locals())
        self.Dropout = Dropout(prob=self.dropout)
        self.classify = Dense(self.size_repr * 2, self.size_classify)
        self.params = util.params(self.Dropout, self.classify)
        #self.names  = ###

    def __call__(self, premise, hypo):
        inp = T.concatenate([premise * hypo, abs(premise - hypo)], axis=1) # features
        return softmax(self.classify(self.Dropout(inp)))
    
class RTE(object):
    """Trainable RTE classifier."""
    def __init__(self, size_repr=1024, size_hidden=200, dropout=0.0, lr=0.0002):
        autoassign(locals())
        self.size_classify = 3
        if self.size_hidden is None:
            self.network = LinearClassify(size_repr=self.size_repr, size_classify=self.size_classify,
                                          dropout=self.dropout)
        else:
            self.network = Classify(size_repr=self.size_repr, size_hidden=self.size_hidden,
                                    size_classify=self.size_classify, activation=tanh,
                                    dropout=self.dropout)
        premise = T.fmatrix()
        hypo    = T.fmatrix()
        target  = T.fmatrix() # should be one hot
        with context.context(training=True):
            predicted = self.network(premise, hypo)
            cost = CrossEntropy(target, predicted)
        with context.context(training=False):
            predicted_test = self.network(premise, hypo)
            cost_test = CrossEntropy(target, predicted_test)
        self.updater = Adam(lr=self.lr)
        updates = self.updater.get_updates(self.network.params, cost, disconnected_inputs='error')
        self.train = theano.function([premise, hypo, target], cost, updates=updates)
        self.loss_test = theano.function([premise, hypo, target], cost_test)
        self.predict = theano.function([premise, hypo] , predicted_test)

        

def cmd_predict_r(model_path='.', 
                  batch_size=128,
                  split='train',
                  output_premise='predict_premise_r.npy',
                  output_hypo='predict_hypo_r.npy',
                  output_labels='entailment_labels.npy'):
    def load(f):
        return pickle.load(gzip.open(os.path.join(model_path, f)))
    model_name = 'model.pkl.gz'
    batcher, scaler, model = map(load, ['batcher.pkl.gz','scaler.pkl.gz', model_name])
    mapper = batcher.mapper
    predict_r = predictor_r(model)
    sents_premise, sents_hypo, labels  = zip(*parse_snli(split=split))
    inputs_premise = list(mapper.transform(sents_premise))
    inputs_hypo    = list(mapper.transform(sents_hypo))
    preds_premise_r = numpy.vstack([ predict_r(batcher.batch_inp(batch))
                                     for batch in grouper(inputs_premise, batch_size) ])
    numpy.save(os.path.join(model_path, split + '_' + output_premise), preds_premise_r)
    preds_hypo_r = numpy.vstack([ predict_r(batcher.batch_inp(batch))
                                     for batch in grouper(inputs_hypo, batch_size) ])
    numpy.save(os.path.join(model_path, split + '_' + output_hypo), preds_hypo_r)
    numpy.save(os.path.join(model_path, split + '_' + output_labels), labels)
    
def parse_snli(split='train', path='/home/gchrupala/repos/reimaginet/data/snli_1.0', omit_hyphen=True):
    """Return pair of premise, hypothesis from the specified split of the SNLI dataset."""
    def labelid(s):
        return ["contradiction","neutral","entailment"].index(s)
    with open(path+'/'+'snli_1.0_' + split + '.jsonl') as f:
        for line in f:
            record = json.loads(line)
            
            if omit_hyphen and record['gold_label'] == '-':
                pass
            else:
                yield (tokenize(record['sentence1']), tokenize(record['sentence2']), labelid(record['gold_label']))
            
def cmd_train_rte(data_path='.',
                  size=200,
                  dropout=0.0,
                  lr=0.0002,
                  epochs=1,
                  batch_size=64,
                  model_path='.',
                  seed=None):
    sys.setrecursionlimit(50000)
    if seed is not None:
                random.seed(seed)
    classify_size = 3
    premise_r = numpy.load(os.path.join(data_path, "train_predict_premise_r.npy"))
    hypo_r    = numpy.load(os.path.join(data_path, "train_predict_hypo_r.npy"))
    labels    = onehot(numpy.load(os.path.join(data_path, "train_entailment_labels.npy")), classify_size)
    val_premise_r = numpy.load(os.path.join(data_path, "dev_predict_premise_r.npy"))
    val_hypo_r  = numpy.load(os.path.join(data_path, "dev_predict_hypo_r.npy"))
    val_labels = onehot(numpy.load(os.path.join(data_path, "dev_entailment_labels.npy")), classify_size)
    size_repr = premise_r.shape[1]
    model = RTE(size_repr=size_repr, size_hidden=size, dropout=dropout, lr=lr)
    start_epoch=1
    for epoch in range(start_epoch, epochs+1):
        costs = Counter()
        for _j,item in enumerate(grouper(itertools.izip(premise_r, hypo_r, labels), batch_size)):
            j = _j + 1
            premise, hypo, label = zip(*item)
            cost = model.train(premise, hypo, label)
            costs += Counter({'cost':cost, 'N':1})
        costs_valid = valid_loss(model, val_premise_r, val_hypo_r, val_labels)
        
        print epoch, j, j*batch_size, "train", "ce", costs['cost']/costs['N']
        print epoch, j, j*batch_size, "valid", "ce", costs_valid['cost']/costs_valid['N']
        print epoch, j, j*batch_size, "valid", "ac", \
            metrics.accuracy_score(numpy.argmax(val_labels, axis=1),
                                   numpy.argmax(model.predict(val_premise_r, val_hypo_r), axis=1))
#        pickle.dump(model, gzip.open(os.path.join(model_path, "entailment_model.{}.pkl.gz".format(epoch)),'w'),
#                    protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(model, gzip.open(os.path.join(model_path, "entailment_model.pkl.gz"),'w'),
                protocol=pickle.HIGHEST_PROTOCOL)
    
def valid_loss(model, premise, hypo, label):
    cost = model.loss_test(premise, hypo, label)
    return Counter({'cost': cost, 'N':1})

def onehot(a, size):
    a = numpy.array(a)
    z = numpy.zeros((a.shape[0], size), dtype='float32')
    z[numpy.arange(a.shape[0]), a] = 1
    return z
