from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  clipped_rectify, CrossEntropy, \
                             last, softmax3d, params
import funktional.context as context
from funktional.layer import params
import imaginet.task
from funktional.util import autoassign
import funktional.util as util
import theano.tensor as T
import theano
import zipfile
import numpy
import StringIO
import json
import cPickle as pickle
import random
from  collections import Counter
import sys

class Decoder(Layer):
    def __init__(self, size_vocab, size_embed, size, depth):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.GRU = StackedGRUH0(self.size_embed, self.size, self.depth, activation=clipped_rectify)

    def params(self):
        return params(self.Embed, self.GRU)

    def __call__(self, out_prev):
        return self.GRU(self.Embed(out_prev))

class LM(imaginet.task.Task):
    def __init__(self, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Decode = Decoder(config['size_vocab'],
                              config['size_embed'], config['size'], config['depth'])
        self.ToTxt  = Dense(config['size'], config['size_vocab'])
        self.inputs = [T.imatrix()]
        self.target = T.imatrix()

    def params(self):
        return params(self.Decode, self.ToTxt)

    def __call__(self, out_prev):
        return softmax3d(self.ToTxt(self.Decode(out_prev)))

    def cost(self, target, prediction):
        oh = OneHot(size_in=self.config['size_vocab'])
        return CrossEntropy(oh(target), prediction)

    def args(self, item):
        """Choose elements of item to be passed to .loss_test and .train functions."""
        return (item['target_prev_t'], item['target_t'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Decode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Decode.GRU.intermediate(self.Decode.Embed(*self.inputs))
        return theano.function(self.inputs, rep)

def embeddings(model):
    return model.task.Decode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder

def trainer(data, prov, model_config, run_config, eval_config, runid=''):
    seed  = run_config.get('seed')
    last_epoch = 0
    if  seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
    model = imaginet.task.GenericBundle(dict(scaler=data.scaler,
                                         batcher=data.batcher),
                                         model_config,
                                         run_config['task'])
    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            result.append(model.task.loss_test(*model.task.args(item)))
        return result

    costs = Counter()
    for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        random.shuffle(data.data['train'])
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                cost = model.task.train(*model.task.args(item))
                costs += Counter({'cost':cost, 'N':1})
                print epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])])
                if j % run_config['validate_period'] == 0:
                        print epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))])
                sys.stdout.flush()

        model.save(path='model.r{}.e{}.zip'.format(runid,epoch))
    model.save(path='model.r{}.zip'.format(runid))
