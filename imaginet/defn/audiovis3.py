from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, Convolution1D, \
                             Embedding, OneHot,  clipped_rectify, clipped_elu, tanh, CosineDistance,\
                             last, softmax3d, params, Attention
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
from imaginet.simple_data import vector_padder

class Encoder(Layer):

    def __init__(self, size_vocab, _size_embed, size, depth,             # TODODODO remove size_embed from this
                residual=False, activation=clipped_rectify,
                filter_length=6, filter_size=1024, stride=3): # FIXME use a more reasonable default
        autoassign(locals())
        self.Conv = Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        self.GRU = StackedGRUH0(self.filter_size, self.size, self.depth,
                                   activation=self.activation, residual=self.residual)

    def params(self):
        return params(self.Conv, self.GRU)
    
    def __call__(self, input):
        return self.GRU(self.Conv(input))

class Visual(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'],
                              config['size_embed'], config['size'],
                              config['depth'],
                              activation=eval(config.get('activation',
                                                         'clipped_rectify')),
                              filter_length=config.get('filter_length', 6), 
                              filter_size=config.get('filter_size', 1024), 
                              stride=config.get('stride', 3),
                              residual=config.get('residual',False))
        self.Attn   = Attention(config['size'])
        self.ToImg  = Dense(config['size'], config['size_target'])
        self.inputs = [T.ftensor3()]
        self.target = T.fmatrix()
        self.config['margin'] = self.config.get('margin', False)
        if self.config['margin']:
            self.srng = RandomStreams(seed=234)
        
        
    def params(self):
        return params(self.Encode, self.Attn, self.ToImg)
    
    def __call__(self, input):
        return self.ToImg(self.Attn(self.Encode(input)))
    
    def cost(self, target, prediction):
        if self.config['margin']:
            return self.Margin(target, prediction, dist=CosineDistance, d=1)
        else:
            return CosineDistance(target, prediction)
    
    def Margin(self, U, V, dist=CosineDistance, d=1.0):
        V_ = (V[self.srng.permutation(n=T.shape(V)[0],
                                      size=(1,)),]).reshape(T.shape(V))
        # A bit silly making it nondet
        return T.maximum(0.0, dist(U, V) - dist(U, V_) + d)
    
    def args(self, item):
        return (item['audio'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.GRU.intermediate(*self.inputs)
        return theano.function(self.inputs, rep)

def predict_img(model, audios, batch_size=32):
    """Project sents to the visual space using model.
    
    For each sentence returns the predicted vector of visual features.
    """
    return numpy.vstack([ model.task.predict(vector_padder(batch))
                            for batch in util.grouper(audios, batch_size) ])
def symbols(model):
    return model.batcher.mapper.ids.decoder
