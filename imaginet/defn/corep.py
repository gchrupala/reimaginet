import funktional.layer as layer
from funktional.layer import params
import imaginet.task as task
from funktional.util import autoassign
import funktional.util as util
import theano.tensor as T

class Encoder(layer.Layer):

    def __init__(self, size_vocab, size_embed, size, depth=1):
        autoassign(locals())

        self.Embed = layer.Embedding(self.size_vocab, self.size_embed)
        self.Encode = layer.StackedGRUH0(self.size_embed, self.size, self.depth,
                                   activation=util.clipped_rectify)

    def params(self):
        return params(self.Embed, self.Encode)
    
    def __call__(self, input):
        return self.Encode(self.Embed(input))
    
class EncoderTask(task.Task):
    
    def __init__(self, updater, encode, project):
        autoassign(locals())
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
    
    def params(self):
        return params(self.encode, self.project)
    
    def __call__(self, input):
        return self.project(layer.last(self.encode(input)))
    
    def cost(self, target, prediction):
        return util.CosineDistance(target, prediction)
    
    def config(self):
        raise NotImplementedError    

class Corep(task.Bundle):
    
    def __init__(self, data_c, data_w, size_embed_c, size_embed_w, size, depth_c, depth_w, size_target):
        autoassign(locals())
        self.updater  = util.Adam()
        self.Encoder_c = Encoder(data_c.mapper.size(), size_embed_c, size, depth=depth_c)
        self.Encoder_w = Encoder(data_w.mapper.size(), size_embed_w, size, depth=depth_w)
        self.ToImg     = layer.Dense(size, size_target)
        self.Task_c = EncoderTask(self.updater, self.Encoder_c, self.ToImg)
        self.Task_c.compile()
        self.Task_w = EncoderTask(self.updater, self.Encoder_w, self.ToImg)
        self.Task_w.compile()

    def params(self):
        return params(self.Encoder_c, self.Encoder_w, self.ToImg)
    
    def config(self):
        return dict(size=self.size, size_target=self.size_target, 
                      c=dict(size_vocab=self.data_c.mapper.size(), 
                             size_embed=self.size_embed_c, 
                             depth=self.depth_c), 
                      w=dict(size_vocab=self.data_w.mapper.size(), 
                             size_embed=self.size_embed_w, 
                             depth=self.depth_w))
    def weights(self):
        return [ param.get_value() for param in self.params() ]
    
    def data(self):
        return dict(c=dict(batcher=self.data_c.batcher, scaler=self.data_c.scaler),
                    w=dict(batcher=self.data_w.batcher, scaler=self.data_w.scaler))
