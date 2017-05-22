from funktional.layer import Layer, Dense, Sum, \
                             Embedding, OneHot,  CosineDistance,\
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
from theano.tensor.shared_randomstreams import RandomStreams

class Encoder(Layer):

    def __init__(self, size_vocab, size_embed):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Sum = Sum(self.size_embed)

    def params(self):
        return params(self.Embed, self.Sum)

    def __call__(self, input):
        return self.Sum(self.Embed(input))

class VectorSum(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'],
                              config['size_embed'])
        self.ToImg  = Dense(config['size_embed'], config['size_target'])
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()


    def params(self):
        return params(self.Encode, self.ToImg)

    def __call__(self, input):
        # Using last because Sum returns the whole seq of partial sums
        # to be compatible with recurrent layers.
        return self.ToImg(last(self.Encode(input)))


    def cost(self, target, prediction):
        return CosineDistance(target, prediction)


    def args(self, item):
        return (item['input'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):

        with context.context(training=False):
            # no layers, insert dimension for comppatibility with stacked 
            rep = self.Encode(*self.inputs).dimshuffle([0, 1, 'x', 2])
        return theano.function(self.inputs, rep)

def predict_img(model, sents, batch_size=128):
    """Project sents to the visual space using model.

    For each sentence returns the predicted vector of visual features.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])

encode_sentences = predict_img

def embeddings(model):
    return model.task.Encode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder
