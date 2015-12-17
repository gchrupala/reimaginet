# First stab: Like MultitaskLMD, but trained on disjoint data
# - task 1: textual encoder, visual decoder
# - task 2: textual encoder, textual decoder
# Parameters of textual encoder are shared
# Task 1 may involve for example sentence reconstruction
from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot, Dropout, Sum, \
                             last, softmax3d, params
from models import *

import funktional.util as util
from funktional.util import CosineDistance, CrossEntropy, linear, clipped_rectify
from funktional.util import autoassign
import funktional.context as context
import theano.tensor as T
import theano

class Task(Layer):

    def cost(self):
        raise NotImplementedError()
    
    def inputs(self):
        raise NotImplementedError()
    
    def target(self):
        raise NotImplementedError()
    
    def _make_train(self, updater):
        with context.context(training=True):
            prediction = self(*self.inputs())
            cost = self.cost(self.target(), prediction)
        return theano.function(self.inputs(), cost, updates=updater(cost))
    
    def _loss_test(self):
        with context.context(training=False):
            prediction = self(*self.inputs())
            cost = self.cost(self.target(), prediction)
        return theano.function(self.inputs(), cost)
    
class Encoder(Layer):

    def __init__(self, size_vocab, size_embed, size, depth):
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedGRUH0(self.size_embed, self.size, self.depth)

        
    def params(self):
        return params(self.Embed, self.Encode)

    def __call__(self, input):
        return self.Encode(self.Embed(input))
    
class Reconstruct(Task):

    def __init__(self, encoder):
        autoassign(locals())
        self.TxtDecode = StackedGRU(self.size_embed, self.size, self.depth)
        self.ToTxt = Dense(self.size, self.size_embed) # map to embeddings
        self.OH = OneHot(size_in=self.encoder.size_vocab)
        
    def params(self):
        return params(self.shared, self.TxtDecode, self.ToTxt)

    def __call__(self, input, target_prev):
        rep = self.last(self.encoder(input))
        return softmax3d(self.Embed.unembed(self.ToTxt(self.TxtDecode(rep,
                                                                      self.encoder.Embed(target_prev)))))
        
    def cost(self, target, prediction):
        return CrossEntropy(self.OH(target), prediction)

    def inputs(self):
        return (T.imatrix(), T.imatrix())

    def target(self):
        return T.imatrix()
    
             
class Imagine(Task):

    def __init__(self, encoder):
        autoassign(locals())
        self.ToImg  = Dense(self.size, self.size_out)
        
    def params(self):
        return params(self.encoder, self.ToImg)
    
    def __call__(self, input):
        rep = self.last(self.encoder(input))
        return self.ToImg(rep)            
    
    def cost(self):
        return CosineDistance

    def inputs(self):
        return (T.imatrix(),)

    def target(self):
        return T.fmatrix()

class TaskTrainer(object):
    
    def __init__(self, tasks):
        autoassign(locals())
        self.updater = util.Adam(max_norm=self.max_norm, lr=self.lr)
        for task in self.tasks:
            task.train  = task._make_train(updater)
            task.loss_test = _make_loss_test()
            
