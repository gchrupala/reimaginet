#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Imaginet (http://arxiv.org/abs/1506.03694) models implemented with funktional
from __future__ import division 
import theano
import numpy
import random
import cPickle as pickle
import argparse
import gzip
import sys
import os
import copy
import funktional.util as util
from funktional.util import linear, clipped_rectify, grouper, pad, autoassign, CosineDistance
from collections import Counter
import data_provider as dp
from models import Imaginet, MultitaskLM, predictor_v
import evaluate
from tokens import tokenize
import json
from  sklearn.preprocessing import StandardScaler

def batch_inp(sents, BEG, END):
    mb = padder(sents, BEG, END)
    return mb[:,1:]

def padder(sents, BEG, END):
    return numpy.array(pad([[BEG]+sent+[END] for sent in sents], END), dtype='int32')

def batch(item, BEG, END):
    """Prepare minibatch."""
    mb_inp = padder([s for s,_,_ in item], BEG, END)
    mb_out_t = padder([r for _,r,_ in item], BEG, END)
    inp = mb_inp[:,1:]
    out_t = mb_out_t[:,1:]
    out_prev_t = mb_out_t[:,0:-1]
    out_v = numpy.array([ t for _,_,t in item ], dtype='float32')
    return (inp, out_v, out_prev_t, out_t)
    
def valid_loss(model, data):
    """Apply model to validation data and return loss info."""
    c = Counter()
    for item in data.iter_valid_batches():
        inp, out_v, out_prev_t, out_t = item
        cost, cost_t, cost_v = model.loss_test(inp, out_v, out_prev_t, out_t)
        c += Counter({'cost_t': cost_t, 'cost_v': cost_v, 'cost': cost, 'N': 1})
    return c
    
class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

def stats(c):
    return " ".join(map(str, [c['cost_t']/c['N'], c['cost_v']/c['N'], c['cost']/c['N']]))

class Data(object):
    """Training / validation data prepared to feed to the model."""
    def __init__(self, provider, mapper, scaler, batch_size=64, with_para='auto', shuffle=False):
        autoassign(locals())
        self.data = {}
        # TRAINING
        if self.with_para == 'para_rand':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_para_rand(provider.iterImages(split='train'))))
        elif self.with_para == 'auto':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_auto(provider.iterImages(split='train'))))
        elif self.with_para == 'para_all':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_para(provider.iterImages(split='train'))))
             
        sents_in = self.mapper.fit_transform(sents_in)
        sents_out = self.mapper.transform(sents_out)
        imgs = self.scaler.fit_transform(imgs)
        self.data['train'] = zip(sents_in, sents_out, imgs)
        # VALIDATION
        if self.with_para == 'para_rand':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_para_rand(provider.iterImages(split='val'))))
        elif self.with_para == 'auto':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_auto(provider.iterImages(split='val'))))
        elif self.with_para == 'para_all':
            sents_in, sents_out, imgs = zip(*self.shuffled(arrange_para(provider.iterImages(split='val'))))

        sents_in = self.mapper.transform(sents_in)
        sents_out = self.mapper.transform(sents_out)
        imgs = self.scaler.transform(imgs)
        self.data['valid'] = zip(sents_in, sents_out, imgs)

        
    def shuffled(self, xs):
        if not self.shuffle:
            return xs
        else:
            zs = copy.copy(list(xs))
            random.shuffle(zs)
            return zs
        
    def iter_train_batches(self):
        for item in grouper(self.data['train'], self.batch_size):
            yield batch(item, self.mapper.BEG_ID, self.mapper.END_ID)
        
    def iter_valid_batches(self):

        for item in grouper(self.data['valid'], self.batch_size):
            yield batch(item, self.mapper.BEG_ID, self.mapper.END_ID)

    def dump(self, model_path):
        """Write mapper and scaler to disc."""
        pickle.dump(self.scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.mapper, gzip.open(os.path.join(model_path, 'mapper.pkl.gz'),'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

def tokens(sent):
    return tokenize(sent['raw'])
    # return sent['tokens']
                    
        
def arrange_para_rand(data):
    for image in data:
        for sent_in in image['sentences']:
            sent_out = random.choice(image['sentences'])
            yield (tokens(sent_in), tokens(sent_out), image['feat'])
            
def arrange_para(data):
    for image in data:
        for sent_in in image['sentences']:
            for sent_out in image['sentences']:
                yield (tokens(sent_in), tokens(sent_out), image['feat'])
                
def arrange_auto(data):
    for image in data:
        for sent in image['sentences']:
            yield (tokens(sent), tokens(sent), image['feat'])

            
    
def cmd_train( dataset='coco',
               datapath='.',
               model_path='.',
               hidden_size=1024,
               gru_activation=clipped_rectify,
               visual_activation=linear,
               max_norm=None,
               embedding_size=None,
               depth=1,
               scaler=None,
               cost_visual=CosineDistance,
               seed=None,
               shuffle=False,
               with_para='auto',
               architecture=MultitaskLM,
               dropout_prob=0.0,
               alpha=0.1,
               epochs=1,
               batch_size=64,
               validate_period=64*100,
               logfile='log.txt'):
    sys.setrecursionlimit(50000) # needed for pickling models
    if seed is not None:
        random.seed(seed)
    prov = dp.getDataProvider(dataset, root=datapath)
    mapper = util.IdMapper(min_df=10)
    embedding_size = embedding_size if embedding_size is not None else hidden_size
    scaler = StandardScaler() if scaler == 'standard' else NoScaler()
    data = Data(prov, mapper, scaler, batch_size=batch_size, with_para=with_para,
                shuffle=shuffle)
    data.dump(model_path)
    model = Imaginet(size_vocab=mapper.size(),
                     size_embed=embedding_size,
                     size=hidden_size,
                     size_out=4096,
                     depth=depth,
                     network=architecture,
                     cost_visual=cost_visual,
                     alpha=alpha,
                     gru_activation=gru_activation,
                     visual_activation=visual_activation,
                     max_norm=max_norm)
    with open(logfile, 'w') as log:
        for epoch in range(1, epochs + 1):
            costs = Counter()
            N = 0
            for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                inp, out_v, out_prev_t, out_t = item
                cost, cost_t, cost_v = model.train(inp, out_v, out_prev_t, out_t)
                costs += Counter({'cost_t':cost_t, 'cost_v': cost_v, 'cost': cost, 'N': 1})
                print epoch, j, j*batch_size, "train", stats(costs)
                if j*batch_size % validate_period == 0:
                    costs_valid = valid_loss(model, data)
                    print epoch, j, j, "valid", stats(costs_valid)
                sys.stdout.flush()
            pickle.dump(model, gzip.open(os.path.join(model_path, 'model.{0}.pkl.gz'.format(epoch)),'w'),
                        protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(model, gzip.open(os.path.join(model_path, 'model.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

def cmd_predict_v(dataset='coco',
                  datapath='.',
                  model_path='.',
                  model_name='model.pkl.gz',
                  batch_size=128,
                  output='predict_v.npy'):
    def load(f):
        return pickle.load(gzip.open(os.path.join(model_path, f)))
    mapper, scaler, model = map(load, ['mapper.pkl.gz','scaler.pkl.gz', model_name])
    predict_v = predictor_v(model)
    prov   = dp.getDataProvider(dataset, root=datapath)
    sents  = list(prov.iterSentences(split='val'))
    inputs = list(mapper.transform([sent['tokens'] for sent in sents ]))
    preds  = numpy.vstack([ predict_v(batch_inp(batch, mapper.BEG_ID, mapper.END_ID))
                            for batch in grouper(inputs, batch_size) ])
    print preds.shape
    numpy.save(os.path.join(model_path, output), preds)
    
def cmd_eval(dataset='coco',
             datapath='.',
             scaler_path='scaler.pkl.gz',
             input='predict_v.npy',
             output='eval.json'):
    scaler = pickle.load(gzip.open(scaler_path))
    preds  = numpy.load(input)
    prov   = dp.getDataProvider(dataset, root=datapath)
    sents  = list(prov.iterSentences(split='val'))
    images = list(prov.iterImages(split='val'))
    img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                              for j in range(len(images)) ]
                            for i in range(len(sents)) ])
    r = evaluate.ranking(img_fs, preds, correct, ns=(1,5,10), exclude_self=False)
    json.dump(r, open(output, 'w'))
    print 'median_rank', numpy.median(r['ranks'])
    for n in (1,5,10):
        print 'recall@{}'.format(n), numpy.mean(r['recall'][n])
        sys.stdout.flush()
