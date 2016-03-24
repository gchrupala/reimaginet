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
from funktional.util import CosineDistance, CrossEntropy, MeanSquaredError, linear, clipped_rectify
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
    """
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

    def _make_representation(self):
        with context.context(training=False):
            rep = self.encoder(*self.inputs)
        return theano.function(self.inputs, rep)

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
    
## Projections 

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.
    
    For each sentence returns the predicted vector of visual features.
    """
    task = model.trainer.tasks['imagine']
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])
    
def representation(model, sents, batch_size=128):
    """Project sents to hidden state space using model.
    
    For each sentence returns a vector corresponding the activation of the hidden layer 
    at the end-of-sentence symbol.
    """
    task = model.trainer.tasks['imagine']
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ task.representation(model.batcher.batch_inp(batch))[:,-1,:]
                            for batch in util.grouper(inputs, batch_size) ])

def states(model, sent):
    """Project each symbol in sent to hidden state space using model.
    
    For each sentence returns a matrix corresponding to the activations of the hidden layer at each 
    position in the sentence.
    """
    task = model.trainer.tasks['imagine']
    inputs = list(model.batcher.mapper.transform([sent]))
    return task.representation(model.batcher.batch_inp(inputs))[0,:,:]

# Accessing model internals

def embeddings(model):
    return model.trainer.tasks['imagine'].encoder.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder

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
    imagine.representation = imagine._make_representation() # compile function to output representation
    return trainer

class Model(object):

    def __init__(self, config, scaler, batcher, weights=None):
            autoassign(locals())
            self.trainer = make_trainer(config, weights=weights)
            
    def save(self, path='model.zip'):
        """Save the data needed to reconstruct model.
        """
        zf = zipfile.ZipFile(path, 'w')
        buf = StringIO.StringIO()
        numpy.save(buf, [ param.get_value() for param in self.trainer.params() ])
        zf.writestr('weights.npy', buf.getvalue(),                compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('config.json', json.dumps(self.config),    compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('scaler.pkl',  pickle.dumps(self.scaler),  compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('batcher.pkl', pickle.dumps(self.batcher), compress_type=zipfile.ZIP_DEFLATED)
    
def load(path='model.zip'):
        """Load data needed and reconstruct model.
        """
        with zipfile.ZipFile(path, 'r') as zf:
            buf = StringIO.StringIO(zf.read('weights.npy'))
            weights = numpy.load(buf)
            config  = json.loads(zf.read('config.json'))
            scaler  = pickle.loads(zf.read('scaler.pkl'))
            batcher = pickle.loads(zf.read('batcher.pkl'))
        return Model(config, scaler, batcher, weights=weights)
