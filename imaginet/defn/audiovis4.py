from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, Convolution1D, \
                             Embedding, OneHot,  clipped_rectify, sigmoid, steeper_sigmoid, tanh, CosineDistance,\
                             last, softmax3d, params, Attention
import funktional.context as context        
from funktional.layer import params
import imaginet.task as task
from funktional.util import autoassign
import funktional.util as util
from funktional.util import orthogonal, xavier, uniform
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
                 residual=False, fixed=False, activation=clipped_rectify,
                 gate_activation=steeper_sigmoid, init_in=orthogonal, init_recur=orthogonal, 
                filter_length=6, filter_size=1024, stride=3): # FIXME use a more reasonable default
        autoassign(locals())
        self.Conv = Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        self.GRU = StackedGRUH0(self.filter_size, self.size, self.depth,
                                activation=self.activation, residual=self.residual,
                                gate_activation=self.gate_activation,
                                init_in=self.init_in, init_recur=self.init_recur)
    def params(self):
        return params(self.Conv, self.GRU)
    
    def __call__(self, input):
        return self.GRU(self.Conv(input))

class Visual(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.margin_size = config.get('margin_size', 0.2)
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'],
                              config['size_embed'], config['size'],
                              config['depth'],
                              activation=eval(config.get('activation',
                                                         'clipped_rectify')),
                              gate_activation=eval(config.get('gate_activation', 'steeper_sigmoid')),
                              filter_length=config.get('filter_length', 6), 
                              filter_size=config.get('filter_size', 1024), 
                              stride=config.get('stride', 3),
                              residual=config.get('residual',False),
                              init_in=eval(config.get('init_in', 'orthogonal')),
                              init_recur=eval(config.get('init_recur', 'orthogonal')))
        self.Attn   = Attention(config['size'])
        self.ImgEncoder  = Dense(config['size_target'], config['size'])
        self.inputs = [T.ftensor3()]
        self.target = T.fmatrix()

    def compile(self):
        task.Task.compile(self)
        self.encode_images = self._make_encode_images()
        
    def params(self):
        return params(self.Encode, self.Attn, self.ImgEncoder)
    
    def __call__(self, input):
        return util.l2norm(self.Attn(self.Encode(input)))
    
    # FIXME HACK ALERT
    def cost(self, i, s_encoded):
        if self.config['contrastive']:
            i_encoded = util.l2norm(self.ImgEncoder(i))
            return util.contrastive(i_encoded, s_encoded, margin=self.margin_size)
        else:
            raise NotImplementedError

    def args(self, item):
        return (item['audio'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.GRU.intermediate(self.Encode.Conv(*self.inputs))
        return theano.function(self.inputs, rep)

    def _make_encode_images(self):
        images = T.fmatrix()
        with context.context(training=False):
            rep = util.l2norm(self.ImgEncoder(images))
        return theano.function([images], rep)

def encode_sentences(model, audios, batch_size=128):
    """Project audios to the joint space using model.
    
    For each audio returns a vector.
    """
    return numpy.vstack([ model.task.predict(vector_padder(batch))
                            for batch in util.grouper(audios, batch_size) ])

def layer_states(model, audios):
    """Pass audios through the model and for each audio return the state of each timestep and each layer."""
    return [ model.task.pile([audio])[0] for audio in audios ]
                                    

def encode_images(model, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ model.task.encode_images(batch)
                          for batch in util.grouper(imgs, batch_size) ])

def symbols(model):
    return model.batcher.mapper.ids.decoder
