from funktional.util import grouper
from imaginet.task import *
import numpy
import imaginet.data_provider as dp
from  sklearn.preprocessing import StandardScaler
from imaginet.simple_data import SimpleData, characters
import sys
import os
import os.path
import cPickle as pickle
import gzip
from evaluate import ranking
import random
from collections import Counter
import cStringIO as StringIO
import zipfile
import json

def train(dataset='coco',
          datapath='.',
          model_path='.',
          tokenize=characters,
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
    model_settings = {}
    sys.setrecursionlimit(50000) # needed for pickling models
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
    prov = dp.getDataProvider(dataset, root=datapath)
    data = SimpleData(prov, tokenize=tokenize, min_df=min_df, scale=scale, 
                      batch_size=batch_size, shuffle=shuffle, limit=limit)
    config = dict(size_vocab=data.mapper.size(), size_embed=size_embed, size_hidden=size_hidden, depth=depth,
                  size=4096, max_norm=max_norm)
    trainer = make_trainer(config)
    do_training(trainer, 'imagine', config, data, epochs, validate_period, model_path)

def make_trainer(config, weights=None):
    encoder = Encoder(size_vocab=config['size_vocab'],
                      size_embed=config['size_embed'],
                      size=config['size_hidden'],
                      depth=config['depth'])
    imagine = Imagine(encoder, size=config['size'])
    trainer = TaskTrainer({'imagine': imagine}, max_norm=config['max_norm'])
    if weights is not None:
        assert len(trainer.params()) == len(weights)
        for param, weight in zip(trainer.params(), weights):
            param.set_value(weight)
    return trainer

def package(trainer, config, data):
    return {'config': config,
            'scaler': data.scaler,
            'batcher': data.batcher,
            'weights': [ param.get_value() for param in trainer.params() ] }
            
def do_training(trainer, taskid, config, data, epochs, validate_period, model_path):
    task = trainer.tasks[taskid]
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
            save(package(trainer, config, data), path='model.{0}.zip'.format(epoch))
    save(package(trainer, config, data), path='model.zip')
    
def evaluate(dataset='coco',
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             tokenize=characters
            ):
    pack = load(path=model_path)
    trainer = make_trainer(pack['config'], pack['weights'])
    scaler = pack['scaler']
    task = trainer.tasks['imagine']
    batcher = pack['batcher']
    mapper = pack['batcher'].mapper
    prov   = dp.getDataProvider(dataset, root=datapath)
    inputs = list(mapper.transform([tokenize(sent) for sent in prov.iterSentences(split='val') ]))
    predictions = numpy.vstack([ task.predict(batcher.batch_inp(batch))
                          for batch in grouper(inputs, batch_size) ])

    sents  = list(prov.iterSentences(split='val'))
    images = list(prov.iterImages(split='val'))
    img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sents)) ] )
    return ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)

def save(pack, path='model.zip'):
    """Save the pack of data needed to reconstruct model pipeline.
    """
    zf = zipfile.ZipFile(path, 'w')
    buf = StringIO.StringIO()
    numpy.save(buf, pack['weights'])
    zf.writestr('weights.npy', buf.getvalue(),                compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr('config.json', json.dumps(pack['config']),    compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr('scaler.pkl',  pickle.dumps(pack['scaler']),  compress_type=zipfile.ZIP_DEFLATED)
    zf.writestr('batcher.pkl', pickle.dumps(pack['batcher']), compress_type=zipfile.ZIP_DEFLATED)
    
def load(path='model.zip'):
    """Load pack needed to reconstruct model pipeline.
    """
    pack = {}
    with zipfile.ZipFile(path, 'r') as zf:
        buf = StringIO.StringIO(zf.read('weights.npy'))
        pack['weights'] = numpy.load(buf)
        pack['config']  = json.loads(zf.read('config.json'))
        pack['scaler']  = pickle.loads(zf.read('scaler.pkl'))
        pack['batcher'] = pickle.loads(zf.read('batcher.pkl'))
    return pack
