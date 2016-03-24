from sklearn.decomposition import PCA
import pylab

class Tracer(object):
    
    def __init__(self, algo='pca'):
        self.algo = algo
    
    def fit(self, data):
        """Fit a dimensionality reduction model on data. 
        """
        print "Embedding"
        print "Fitting PCA"
        if self.algo == 'pca':
            self.proj = PCA(n_components=2)
            self.proj.fit(data)
        else:
            raise ValueError("Unknown algo {}".format(self.algo))

    def project(self, data):
        return self.proj.transform(data)
        
    def traces(self, sents, reps, loc="best", eos=False):
        """Plots traces for given sents.
        """
        last = None if eos else -1
        for i in range(len(sents)):
            xy = self.proj.transform(reps[i])
            x = xy[0:last,0] ; y = xy[0:last,1]
            pylab.plot(x, y, label=''.join(sents[i]), linewidth=3, alpha=0.5)
            for j in range(0,xy.shape[0]-1):
                pylab.text(xy[j,0], xy[j,1], sents[i][j], va='center', ha='center', alpha=0.5)
        pylab.legend(loc=loc)