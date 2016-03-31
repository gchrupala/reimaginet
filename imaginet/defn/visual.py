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

    def __init__(self, size_vocab, size_embed, size, depth, size_target, max_norm=None, lr=0.0002):
        autoassign(locals())
        self.updater = util.Adam(max_norm=self.max_norm, lr=self.lr)
        self.Encode = Encoder(self.size_vocab, self.size_embed, self.size, self.depth)
        self.ToImg  = Dense(self.size, self.size_target)
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
        
    def params(self):
        return params(self.Encode, self.ToImg)
    
    def __call__(self, input):
        rep = last(self.Encode(input))
        return self.ToImg(rep)            
    
    def cost(self, target, prediction):
        return CosineDistance(target, prediction)

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.GRU.intermediate(self.Encode.Embed(*self.inputs))
        return theano.function(self.inputs, rep)

class VisualModel(task.Bundle):
    
    def __init__(self, data, config, weights=None):
        self.config = dict(size_embed=1024, size=1024, depth=3, 
                           size_target=4096, max_norm=None, lr=0.0002)
        self.config.update(config)
        self.data = data
        self.batcher = data['batcher']
        self.scaler = data['scaler']
        self.Visual = Visual(self.data['batcher'].mapper.size(), 
                             self.config['size_embed'], 
                             self.config['size'], 
                             self.config['depth'], 
                             self.config['size_target'], 
                             max_norm=self.config['max_norm'], 
                             lr=self.config['lr']
                            )
        if weights is not None:
            assert len(self.Visual.params())==len(weights)
            for param, weight in zip(self.Visual.params(), weights):
                param.set_value(weight)
        self.Visual.compile()
        self.Visual.representation = self.Visual._make_representation()
        self.Visual.pile = self.Visual._make_pile()
                 
    def params(self):
        return self.Visual.params()
    
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
    return VisualModel(data, config, weights=weights)
                 

## Projections 

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.
    
    For each sentence returns the predicted vector of visual features.
    """
    task = model.Visual
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])
    
def representation(model, sents, batch_size=128):
    """Project sents to hidden state space using model.
    
    For each sentence returns a vector corresponding the activation of the hidden layer 
    at the end-of-sentence symbol.
    """
    task = model.Visual
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ task.representation(model.batcher.batch_inp(batch))[:,-1,:]
                            for batch in util.grouper(inputs, batch_size) ])

def states(model, sents, batch_size=128):
    """Project each symbol in each sentence in sents to hidden state space using model.
    
    For each sentence returns a matrix corresponding to the activations of the top hidden layer at each 
    position in the sentence.
    """
    return [ r[:,-1,:] for r in pile(model, sents, batch_size=128) ]

def pile(model, sents, batch_size=128):
    """Project each symbol in each sentence in sents to hidden state spaces corresponding to layers using model.
    
    For each sentence returns a 3D tensor corresponding to the activations of the hidden layers at each 
    position in the sentence.
    """
    task = model.Visual
    lens = map(len, sents)
    inputs = list(model.batcher.mapper.transform(sents))
    rs = [ r for batch in util.grouper(inputs, batch_size)
               for r in task.pile(model.batcher.batch_inp(batch)) ]    
    return [ r[-l-1:,:,:] for (r,l) in zip(rs, lens) ]

# Accessing model internals

def embeddings(model):
    return model.Visual.Encode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder
