import numpy
from sklearn.decomposition import PCA
import pylab
import theano.tensor as T
import theano
import funktional.util

class Tracer(object):
    
    def __init__(self, model, algo='pca'):
        self.algo = algo
        self.model = model
        self.network = self.model['model'].network
        self._predict = self._make_predictor()        
    
    def _make_predictor(self):
        input = T.imatrix()
        return theano.function([input], self.network.Visual.Encode(self.network.Embed(input)))

    def embed(self, sents):       
        return self._predict(self.model['batcher'].batch_inp(list(self.model['batcher'].mapper.transform(sents))))
    
    def embed_last(self, sents, batch_size=128):
        def last(x):
            return x.transpose((1,0,2))[-1]
        return numpy.vstack([last(self.embed(batch)) for batch in funktional.util.grouper(sents, batch_size) ])
    
    def fit(self, data):
        """Fit a dimensionality reduction model on data. 
        
        data: list of lists of strings"""
        print "Embedding"
        X = self.embed_last(data)
        print "Fitting PCA"
        if self.algo == 'pca':
            self.proj = PCA(n_components=2)
            self.proj.fit(X)
        else:
            raise ValueError("Unknown algo {}".format(self.algo))

    def project(self, sents):
        return self.proj.transform(self.embed(sents))
        
    def traces(self, sents, size=(7,5), loc="best"):
        """Plots traces for given sents.
        
        sents: list of lists of strings"""
        pylab.figure(figsize=size)
        for i in range(len(sents)):
            sent_i = self.embed([sents[i]])[0]
            xy = self.proj.transform(sent_i)
            x = xy[0:-1,0] ; y = xy[0:-1,1]
            pylab.plot(x, y, label=' '.join(sents[i]), linewidth=3, alpha=0.3)
            for j in range(0,xy.shape[0]-1):
                pylab.text(xy[j,0], xy[j,1], (sents[i])[j], va='center', ha='center', alpha=0.5)
        pylab.legend(loc=loc)
