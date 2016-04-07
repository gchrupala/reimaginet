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
        self.Decode = Decoder(config['size_vocab'], config['size_embed'], config['size'], config['depth'])
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
        inp, target_v, out_prev, target_t = item
        return (out_prev, target_t)
    
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
    