from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  clipped_rectify, clipped_elu, CosineDistance,\
                             last, softmax3d, params
import funktional.context as context        
from funktional.layer import params
import imaginet.task as task
from funktional.util import autoassign
import funktional.util as util
from funktional.util import steeper_sigmoid, sigmoid, orthogonal, xavier
import theano.tensor as T
import theano
import zipfile
import numpy
import StringIO
import json
import cPickle as pickle
from theano.tensor.extra_ops import fill_diagonal

class Encoder(Layer):

    def __init__(self, size_vocab, size_embed, size, depth,
                 residual=False, activation=clipped_rectify):
        autoassign(locals())

        self.Embed = OneHot(self.size_vocab)
        self.GRU = StackedGRUH0(self.size_vocab, self.size, self.depth,
                                   activation=self.activation, residual=self.residual)

    def params(self):
        return params(self.Embed, self.GRU)
    
    def __call__(self, input):
        return self.GRU(self.Embed(input))

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
                              residual=config.get('residual',False))
        self.ImgEncoder  = Dense(config['size_target'], config['size'], init=eval(config.get('init_img', 'orthogonal')))
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()      

    def compile(self):
        task.Task.compile(self)
        self.encode_images = self._make_encode_images()
        
        
    def params(self):
        return params(self.Encode, self.ImgEncoder)
    
    def __call__(self, input):
        return util.l2norm(last(self.Encode(input)))

    def cost(self, i, s_encoded):
        if self.config['contrastive']:
            i_encoded = util.l2norm(self.ImgEncoder(i))
            return self.contrastive(i_encoded, s_encoded, margin=self.margin_size)
        else:
            raise NotImplementedError

    def contrastive(self, i, s, margin=0.2): 
        # i: (fixed) image embedding, 
        # s: sentence embedding
        errors = - util.cosine_matrix(i, s)
        diagonal = errors.diagonal()
        # compare every diagonal score to scores in its column (all contrastive images for each sentence)
        cost_s = T.maximum(0, margin - errors + diagonal)  
        # all contrastive sentences for each image
        cost_i = T.maximum(0, margin - errors + diagonal.reshape((-1, 1)))  
        cost_tot = cost_s + cost_i
        # clear diagonals
        cost_tot = fill_diagonal(cost_tot, 0)

        return cost_tot.mean()

    def Margin(self, U, V, dist=CosineDistance, d=1.0):
        V_ = (V[self.srng.permutation(n=T.shape(V)[0],
                                      size=(1,)),]).reshape(T.shape(V))
        # A bit silly making it nondet
        return T.maximum(0.0, dist(U, V) - dist(U, V_) + d)
    
    def args(self, item):
        return (item['input'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.GRU.intermediate(self.Encode.Embed(*self.inputs))
        return theano.function(self.inputs, rep)

    def _make_encode_images(self):
        images = T.fmatrix()
        with context.context(training=False):
            rep = util.l2norm(self.ImgEncoder(images))
        return theano.function([images], rep)

def encode_sentences(model, sents, batch_size=128):
    """Project sents to the joint space using model.
    
    For each sentence returns a vector.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])

def encode_images(model, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ model.task.encode_images(batch)
                          for batch in util.grouper(imgs, batch_size) ])

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.
    
    For each sentence returns the predicted vector of visual features.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])

def embeddings(model):
    return model.task.Encode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder
