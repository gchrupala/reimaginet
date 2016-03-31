from funktional.util import grouper, autoassign
from imaginet.task import *
import imaginet.defn.visual as visual
import numpy
import imaginet.data_provider as dp
from  sklearn.preprocessing import StandardScaler
from imaginet.simple_data import SimpleData, characters, phonemes
import sys
import os
import os.path
import cPickle as pickle
import gzip
from evaluate import ranking
import random
from collections import Counter

def train(dataset='coco',
          datapath='.',
          model_path='.',
          tokenize=phonemes,
          max_norm=None,
          min_df=10,
          scale=True,
          epochs=1,
          batch_size=64,
          shuffle=True,
          size_embed=128,
          size_hidden=512,
          depth=2,
          validate_period=64*1000,
          limit=None,
          seed=None):
    # sys.setrecursionlimit(50000) # needed for pickling models
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
    prov = dp.getDataProvider(dataset, root=datapath)
    data = SimpleData(prov, tokenize=tokenize, min_df=min_df, scale=scale, 
                      batch_size=batch_size, shuffle=shuffle, limit=limit)
    config = dict(size_embed=size_embed, size=size_hidden, depth=depth,
                  size_target=4096, max_norm=max_norm)
    model = visual.VisualModel(dict(scaler=data.scaler, batcher=data.batcher), config)
    do_training(model, data, epochs, validate_period, model_path)

def do_training(model, data, epochs, validate_period, model_path):
    task = model.Visual
    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            inp, target_v, _, _ = item
            result.append(task.loss_test(inp, target_v))
        return result
    for epoch in range(1, epochs + 1):
            print len(task.params())
            costs = Counter()
            for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                inp, target_v, _, _ = item
                cost = task.train(inp, target_v)
                costs += Counter({'cost':cost, 'N':1})
                print epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])])
                if j*data.batch_size % validate_period == 0:
                        print epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))])
                sys.stdout.flush()
            model.save(path='model.{0}.zip'.format(epoch))
    model.save(path='model.zip')
    
def evaluate(dataset='coco',
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             tokenize=phonemes
            ):
    model = visual.load(path=model_path)
    task = model.Visual
    scaler = model.scaler
    batcher = model.batcher
    mapper = batcher.mapper
    prov   = dp.getDataProvider(dataset, root=datapath)
    sents_tok =  [ tokenize(sent) for sent in prov.iterSentences(split='val') ]
    predictions = visual.predict_img(model, sents_tok, batch_size=batch_size)
    sents  = list(prov.iterSentences(split='val'))
    images = list(prov.iterImages(split='val'))
    img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sents)) ] )
    return ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)
