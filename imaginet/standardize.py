# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

class Std(object):
    
    def __init__(self):
        self.N = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float('inf')
        self.max = -float('inf')
    
    def update(self, x):
        self.N = self.N + 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.N
        self.M2 = self.M2 + delta * (x-self.mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        
    def var(self):
        if self.N < 2:
            return float('nan')
        else:
            return self.M2 / (self.N - 1)
        
    def std(self):
        return self.var() ** 0.5
    
    def __repr__(self):
        return "Std(N={}, mean={}, M2={}, min={}, max={})".\
                  format(self.N, self.mean, self.M2, self.min, self.max)
 
    def standardize(self, x):
	return (x-self.mean)/self.std()


