from funktional.layer import Layer, Dense, GRUH0, StackedGRUH0, \
                             Embedding, OneHot,  clipped_rectify,\
                             last, softmax3d, params
import funktional.context as context
from funktional.layer import params
import imaginet.task as task
from funktional.util import autoassign
import funktional.util as util
import theano.tensor as T
import theano
import zipfile
import numpy
import StringIO
import json
import cPickle as pickle
from theano.tensor.shared_randomstreams import RandomStreams
import imaginet.defn.visual as visual
from collections import Counter
import random
import sys

class Network:

    def __init__(self, size_vocab, size_embed, size, size_target):
        autoassign(locals())
        self.Shared = Embedding(self.size_vocab, self.size_embed)
        self.EncodeV = StackedGRUH0(size_embed, size, depth=1, activation=clipped_rectify)
        self.EncodeLM  = StackedGRUH0(size_embed, size, depth=1, activation=clipped_rectify)
        self.ToTxt = Dense(size, size_vocab)
        self.ToImg = Dense(size, size_target)


    def params(self):
        return params(self.Shared, self.EncodeV, self.EncodeLM, self.ToTxt, self.ToImg)

class Visual(task.Task):

    def __init__(self, network, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()

    def params(self):
        return params(self.network.Shared, self.network.EncodeV, self.network.ToImg)

    def __call__(self, input):
        return self.network.ToImg(last(self.network.EncodeV(self.network.Shared(input))))

    def cost(self, target, prediction):
        return util.CosineDistance(target, prediction)

    def args(self, item):
        return (item['input'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.network.EncodeV(self.network.Shared(*self.inputs))
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.network.EncodeV.intermediate(self.network.Shared(*self.inputs))
        return theano.function(self.inputs, rep)

class LM(task.Task):

    def __init__(self, network, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.inputs = [T.imatrix()]
        self.target = T.imatrix()

    def params(self):
        return params(self.network.Shared, self.network.EncodeLM, self.network.ToTxt)

    def __call__(self, prev):
        return softmax3d(self.network.ToTxt(self.network.EncodeLM(self.network.Shared(prev))))

    def cost(self, target, prediction):
        oh = OneHot(size_in=self.network.size_vocab)
        return util.CrossEntropy(oh(target), prediction)

    def args(self, item):
        """Choose elements of item to be passed to .loss_test and .train functions."""
        return (item['target_prev_t'], item['target_t'])

    def _make_pile(self):
        with context.context(training=False):
            rep = self.network.EncodeLM.intermediate(self.network.Shared(*self.inputs))
        return theano.function(self.inputs, rep)


class LMVisual(task.Bundle):

    def __init__(self, data, config, weights=None):
        self.config = config
        self.data = data
        self.batcher = data['batcher']
        self.scaler = data['scaler']
        self.config['size_vocab'] = self.data['batcher'].mapper.size()
        self.network = Network(config['size_vocab'], config['size_embed'], config['size'],
                          config['size_target'])
        self.visual = Visual(self.network, config)
        self.lm = LM(self.network, config)
        if weights is not None:
            assert len(self.network.params())==len(weights)
            for param, weight in zip(self.params(), weights):
                param.set_value(weight)
        self.visual.compile()
        self.visual.representation = self.visual._make_representation()
        self.visual.pile = self.visual._make_pile()
        self.lm.compile()
        self.lm.pile = self.lm._make_pile()
        
    def params(self):
        return self.network.params()

    def get_config(self):
        return self.config

    def get_data(self):
        return self.data

def load(path):
    """Load data and reconstruct model."""
    with zipfile.ZipFile(path,'r') as zf:
        buf = StringIO.StringIO(zf.read('weights.npy'))
        weights = numpy.load(buf)
        config  = json.loads(zf.read('config.json'))
        data  = pickle.loads(zf.read('data.pkl'))
    return LMVisual(data, config, weights=weights)


def trainer(model, data, epochs, validate_period, model_path, prob_lm=0.1, runid=''):
    def valid_loss():
        result = dict(lm=[], visual=[])
        for item in data.iter_valid_batches():
            result['lm'].append(model.lm.loss_test(*model.lm.args(item)))
            result['visual'].append(model.visual.loss_test(*model.visual.args(item)))
        return result
    costs = Counter(dict(cost_v=0.0, N_v=0.0, cost_t=0.0, N_t=0.0))
    print "LM: {} parameters".format(count_params(model.lm.params()))
    print "Vi: {} parameters".format(count_params(model.visual.params()))
    for epoch in range(1,epochs+1):
        for _j, item in enumerate(data.iter_train_batches()):
            j = _j +1
            if random.random() <= prob_lm:
                cost_t = model.lm.train(*model.lm.args(item))
                costs += Counter(dict(cost_t=cost_t, N_t=1))
            else:
                cost_v = model.visual.train(*model.visual.args(item))
                costs += Counter(dict(cost_v=cost_v, N_v=1))
            print epoch, j, j*data.batch_size, "train", \
                    numpy.divide(costs['cost_v'], costs['N_v']),\
                    numpy.divide(costs['cost_t'], costs['N_t'])
            if j % validate_period == 0:
                result = valid_loss()
                print epoch, j, 0, "valid", \
                    numpy.mean(result['visual']),\
                    numpy.mean(result['lm'])
                sys.stdout.flush()
        model.save(path='model.r{}.e{}.zip'.format(runid, epoch))
    model.save(path='model.zip')

def count_params(params):
    def product(xs):
        return reduce(lambda z, x: z*x, xs, 1)
    return sum((product(param.get_value().shape) for param in params))

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.

    For each sentence returns the predicted vector of visual features.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.visual.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])

encode_sentences = predict_img
