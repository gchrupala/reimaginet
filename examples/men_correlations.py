# coding: utf-8

import imaginet.task
import imaginet.defn.lm as lm
import imaginet.defn.visual1h as visual1h
from scipy.spatial.distance import cosine
from urllib2 import urlopen
from subprocess import check_output
import numpy
import scipy.stats

# Conversion to phonemes

def espeak(words):
    return phon(check_output(["espeak", "-q", "--ipa=3",
                        '-v', 'en',
                        words]).decode('utf-8'))
def phon(inp):
    # get rid of accents
    return [ ph.replace(u"ˈ","").replace(u"ˌ","") for word in inp.split() for ph in word.split("_") ]

# Cosine distance from precomuted representation table
def cos(R, word1, word2, layer=-1):
    return 1-cosine(R[word1][layer,:], R[word2][layer,:])

# Precompute representations
def reps(model, data):
    R = {}
    vocab = set()
    for x in data:
        if x[0] not in vocab:
            vocab.add(x[0])
        if x[1] not in vocab:
            vocab.add(x[1])
    words = [ w for w in vocab]
    ipas  = [ espeak(w) for w in vocab ]
    reps = imaginet.task.pile(model, ipas)
    for i in range(0,len(words)):
        R[words[i]] = reps[i][-1,:,:]
    return R

# Human judgements
def judge(url):
    result = []
    f = urlopen(url)
    for line in f:
        fields = line.split("\t")
        if len(fields) == 3:
            result.append((fields[0], fields[1], float(fields[2])))
    return result



M_old = imaginet.task.load(path="/home/gchrupala/repos/reimaginet/examples/vis/model.10.zip")
M_new = imaginet.task.load(path="/home/gchrupala/repos/reimaginet/examples/emnlp-2016/phon-gru.8.zip")
menurl = "https://raw.githubusercontent.com/kadarakos/wordsims.io/master/wordsims/word-sim-data/EN-MEN-TR-3k.txt"
MEN = judge(menurl)
human_m  = [ x[2] for x in MEN ]



R_old = reps(M_old, MEN)
R_new = reps(M_new, MEN)



# Print correlations for MEN
print "OLD"
for layer in range(0,3):
    system_m = [ cos(R_old, x[0], x[1], layer=layer) for x in MEN ]
    print layer+1
    print "MEN", scipy.stats.spearmanr(human_m, system_m)

print "NEW"    
for layer in range(0,3):
    system_m = [ cos(R_new, x[0], x[1], layer=layer) for x in MEN ]
    print layer+1
    print "MEN", scipy.stats.spearmanr(human_m, system_m)
    



