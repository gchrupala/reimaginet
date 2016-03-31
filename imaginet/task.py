# First stab: Like MultitaskLMD, but trained on disjoint data
# - task 1: textual encoder, visual decoder
# - task 2: textual encoder, textual decoder
# Parameters of textual encoder are shared
# Task 2 may involve for example sentence reconstruction

from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  \
                             last, softmax3d, params
from models import * # FIXME this needed?
import numpy
import funktional.util as util
from funktional.util import autoassign
import funktional.context as context
import theano.tensor as T
import theano
import zipfile
import cStringIO as StringIO
import json
import cPickle as pickle

class Task(Layer):
    """Task is a trainable Layer.

    You need to set the following attributes:
    - inputs - list of theano symbolic variables
    - target - theano symbolic variable
    - updater - optimizer object (e.g. SGD or Adam)
    """
    inputs = None
    target = None
    updater = None
    
   
    def cost(self, target, prediction):
        raise NotImplementedError
                  
    def _make_train(self):
        """Compile function for training."""
        with context.context(training=True):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target], 
                               thecost, 
                               updates=self.updater.get_updates(self.params(), thecost))

    def _make_loss_test(self):
        """Compile function for computing the loss function at test time."""
        with context.context(training=False):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target], thecost)
   
    def _make_predict(self):
        """Compile function for computing the target."""
        with context.context(training=False):
                prediction = self(*self.inputs)
        return theano.function(self.inputs, prediction)
                               
    def compile(self):
        """Compiles theano functions and adds them to self."""
        self.train     = self._make_train()
        self.loss_test = self._make_loss_test()
        self.predict   = self._make_predict()

class Bundle():
    
    """Interface for combinations of task/data."""
    
    def params(self):
        raise NotImplementedError
        
    def weights(self):
        return [ param.get_value() for param in self.params() ]
    
    def get_config(self):
        raise NotImplementedError
    
    def get_data(self):
        raise NotImplementedError
    
    def save(self, path):
        zf = zipfile.ZipFile(path, 'w')
        buf = StringIO.StringIO()
        numpy.save(buf, self.weights())
        zf.writestr('weights.npy', buf.getvalue(),            compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('config.json', json.dumps(self.get_config()), compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('data.pkl',    pickle.dumps(self.get_data()), compress_type=zipfile.ZIP_DEFLATED)
    