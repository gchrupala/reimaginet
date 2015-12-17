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

class Tasks(object):
    
    def __call__(self, task, *args):
        """Dispatch inputs to task."""
        return self.task[task](*args)

class TasksRI(Tasks):

    def __init__(self):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedGRUH0(self.size_embed, self.size, self.depth)
        self.ToImg  = Dense(self.size, self.size_out)
        self.TxtDecode = StackedGRU(self.size_embed, self.size, self.depth)
        self.ToTxt = Dense(self.size, self.size_embed) # map to embeddings
                   
    def params_all(self):
        return params(self.Embed, self.Encode, self.ToImg, self.TxtDecode, self.ToTxt)

    def params(self):
        """Mapping from task names to task params."""
        return dict(reconstruct=params(self.Embed, self.Encode, self.TxtDecode, self.ToTxt),
                    imagine=params(self.Embed, self.Encode, self.ToImg))

    def reconstruct(self, input, target_prev):
            rep = self.last(self.Encode(self.Embed(input)))
            return softmax3d(self.Embed.unembed(self.ToTxt(self.TxtDecode(rep, self.Embed(target_prev)))))

    def imagine(self, input):
            rep = self.last(self.Encode(self.Embed(input)))
            return self.ToImg(rep)            

    def task(self):
        """Mapping from task names to task forward functions."""
        return dict(reconstruct=self.reconstruct, imagine=self.imagine)

    def cost(self):
        """Mapping from task names to task cost functions."""
        return dict(reconstruct=CrossEntropy, imagine=CosineDistance)



