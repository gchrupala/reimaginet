# First stab: Like MultitaskLMD, but trained on disjoint data
# - task 1: textual encoder, visual decoder
# - task 2: textual encoder, textual decoder
# Parameters of textual encoder are shared
# Task 2 may involve for example sentence reconstruction

from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  \
                             last, softmax3d, params
from models import *

import funktional.util as util
from funktional.util import CosineDistance, CrossEntropy, MeanSquaredError, linear, clipped_rectify
from funktional.util import autoassign
import funktional.context as context
import theano.tensor as T
import theano

class Task(Layer):
    # Attributes to be implemented
    inputs = None
    target = None
    
    def cost(target, prediction):
        raise NotImplementedError
    
    def _make_train(self, updates):
        with context.context(training=True):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target], thecost, updates=updates(thecost))

    def _make_loss_test(self):
        with context.context(training=False):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target], thecost)
   
    def _make_predict(self):
        with context.context(training=False):
                prediction = self(*self.inputs)
        return theano.function(self.inputs, prediction)
    
class Encoder(Layer):

    def __init__(self, size_vocab, size_embed, size, depth=1):
        autoassign(locals())

        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedGRUH0(self.size_embed, self.size, self.depth,
                                   activation=clipped_rectify)

    def params(self):
        return params(self.Embed, self.Encode)
    
    def __call__(self, input):
        return self.Encode(self.Embed(input))

    
class Reconstruct(Task):

    def __init__(self, encoder):
        autoassign(locals())
        self.TxtDecode = StackedGRU(self.size_embed, self.size, self.depth,
                                    activation=clipped_rectify)
        self.ToTxt = Dense(self.size, self.size_embed) # map to embeddings
        self.OH = OneHot(size_in=self.encoder.size_vocab)
        self.inputs = [T.imatrix(), T.imatrix()]
        self.target = T.imatrix()
        
    def params(self):
        return params(self.shared, self.TxtDecode, self.ToTxt)

    def __call__(self, input, target_prev):
        rep = last(self.encoder(input))
        return softmax3d(self.Embed.unembed(self.ToTxt(self.TxtDecode(rep,
                                                                      self.encoder.Embed(target_prev)))))
        
    def cost(self, target, prediction):
        return CrossEntropy(self.OH(target), prediction)
    
             
class Imagine(Task):

    def __init__(self, encoder, size):
        autoassign(locals())
        self.ToImg  = Dense(self.encoder.size, self.size)
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
        
    def params(self):
        return params(self.encoder, self.ToImg)
    
    def __call__(self, input):
        rep = last(self.encoder(input))
        return self.ToImg(rep)            
    
    def cost(self, target, prediction):
        return CosineDistance(target, prediction)



class TaskTrainer(object):
    
    def __init__(self, tasks, max_norm=None, lr=0.0002):
        autoassign(locals())
        self.updater = util.Adam(max_norm=self.max_norm, lr=self.lr)
        for name, task in self.tasks.items():
            task.train  = task._make_train(self.updates)
            task.loss_test = task._make_loss_test()
            task.predict = task._make_predict()
            
    def updates(self, cost):
        return self.updater.get_updates(self.params(), cost,  disconnected_inputs='warn')
    
    def params(self):
        return params(*self.tasks.values())
    
