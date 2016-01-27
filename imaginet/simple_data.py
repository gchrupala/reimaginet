import numpy
import cPickle as pickle
import gzip
import os
import copy
import funktional.util as util
from funktional.util import autoassign
from  sklearn.preprocessing import StandardScaler
import string
import random
# Types of tokenization

def words(sentence):
    return sentence['tokens']

def characters(sentence):
    return list(sentence['raw'])

def compressed(sentence):
    return [ c.lower() for c in sentence['raw'] if c in string.letters ]

class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x
    
class Batcher(object):

    def __init__(self, mapper, pad_end=False):
        autoassign(locals())
        self.BEG = self.mapper.BEG_ID
        self.END = self.mapper.END_ID
        
    def pad(self, xss): # PAD AT BEGINNING
        max_len = max((len(xs) for xs in xss))
        def pad_one(xs):
            if self.pad_end:
                return xs + [ self.END for _ in range(0,(max_len-len(xs))) ] 
            return [ self.BEG for _ in range(0,(max_len-len(xs))) ] + xs
        return [ pad_one(xs) for xs in xss ]

    def batch_inp(self, sents):
        mb = self.padder(sents)
        return mb[:,1:]

    def padder(self, sents):
        return numpy.array(self.pad([[self.BEG]+sent+[self.END] for sent in sents]), dtype='int32')

    def batch(self, item):
        """Prepare minibatch. 
        
        Returns:
        - input string
        - visual target vector
        - output string at t-1
        - target string
        """
        mb_inp = self.padder([s for s,_,_ in item])
        mb_target_t = self.padder([r for _,r,_ in item])
        inp = mb_inp[:,1:]
        target_t = mb_target_t[:,1:]
        target_prev_t = mb_target_t[:,0:-1]
        target_v = numpy.array([ t for _,_,t in item ], dtype='float32')
        return (inp, target_v, target_prev_t, target_t)

    
class SimpleData(object):
    """Training / validation data prepared to feed to the model."""
    def __init__(self, provider, tokenize=words, min_df=10, scale=True, batch_size=64, shuffle=False, limit=None):
        autoassign(locals())
        self.data = {}
        self.mapper = util.IdMapper(min_df=self.min_df)
        self.scaler = StandardScaler() if scale else NoScaler()

        # TRAINING
        sents_in, sents_out, imgs = zip(*self.shuffled(arrange(provider.iterImages(split='train'), 
                                                               tokenize=self.tokenize, 
                                                               limit=limit)))
        sents_in = self.mapper.fit_transform(sents_in)
        sents_out = self.mapper.transform(sents_out)
        imgs = self.scaler.fit_transform(imgs)
        self.data['train'] = zip(sents_in, sents_out, imgs)

        # VALIDATION
        sents_in, sents_out, imgs = zip(*self.shuffled(arrange(provider.iterImages(split='val'), tokenize=self.tokenize)))
        sents_in = self.mapper.transform(sents_in)
        sents_out = self.mapper.transform(sents_out)
        imgs = self.scaler.transform(imgs)
        self.data['valid'] = zip(sents_in, sents_out, imgs)
        self.batcher = Batcher(self.mapper, pad_end=False)
        
    def shuffled(self, xs):
        if not self.shuffle:
            return xs
        else:
            zs = copy.copy(list(xs))
            random.shuffle(zs)
            return zs
        
    def iter_train_batches(self):
        for bunch in util.grouper(self.data['train'], self.batch_size*20):
            bunch_sort = [ bunch[i] for i in numpy.argsort([len(x) for x,_,_ in bunch]) ]
            for item in util.grouper(bunch_sort, self.batch_size):
                yield self.batcher.batch(item)
        
    def iter_valid_batches(self):
        for bunch in util.grouper(self.data['valid'], self.batch_size*20):
            bunch_sort = [ bunch[i] for i in numpy.argsort([len(x) for x,_,_ in bunch]) ]
            for item in util.grouper(bunch_sort, self.batch_size):
                yield self.batcher.batch(item)

    def dump(self, model_path):
        """Write scaler and batcher to disc."""
        pickle.dump(self.scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.batcher, gzip.open(os.path.join(model_path, 'batcher.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

def arrange(data, tokenize=words, limit=None):
    for i,image in enumerate(data):
        if limit is not None and i > limit:
            break
        for sent in image['sentences']:
            toks = tokenize(sent)
            yield (toks, toks, image['feat'])
            
            
