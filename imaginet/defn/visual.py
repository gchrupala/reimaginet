from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  clipped_rectify, CosineDistance,\
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

class Encoder(Layer):

    def __init__(self, size_vocab, size_embed, size, depth):
        autoassign(locals())

        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.GRU = StackedGRUH0(self.size_embed, self.size, self.depth,
                                   activation=clipped_rectify)

    def params(self):
        return params(self.Embed, self.GRU)
    
    def __call__(self, input):
        return self.GRU(self.Embed(input))

class Visual(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'], config['size_embed'], config['size'], config['depth'])
        self.ToImg  = Dense(config['size'], config['size_target'])
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
        
    def params(self):
        return params(self.Encode, self.ToImg)
    
    def __call__(self, input):
        return self.ToImg(last(self.Encode(input)))
    
    def cost(self, target, prediction):
        return CosineDistance(target, prediction)
    
    def args(self, item):
        inp, target_v, out_prev, target_t = item
        return (inp, target_v)

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.GRU.intermediate(self.Encode.Embed(*self.inputs))
        return theano.function(self.inputs, rep)

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.
    
    For each sentence returns the predicted vector of visual features.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])
