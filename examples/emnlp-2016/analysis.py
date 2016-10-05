# coding: utf-8
import imaginet.task
import imaginet.defn.lm as lm
import imaginet.defn.visual1h 
import imaginet.defn.visual1h2
import imaginet.defn.visual2 
from scipy.spatial.distance import cosine
from urllib2 import urlopen
from subprocess import check_output
import numpy
import scipy.stats
import json
import subprocess

# # Conversion to phonemes
# def clean_phonemes(espeakoutput):
#     '''
#     Takes espeak output as input
#     returns list of phonemes and boundaries
#     boundaries are indicated by an asterisk
#     '''
#     no_word_stress = espeakoutput.replace("ˈ", "")    
#     no_stress = no_word_stress.replace("ˌ", "")
#     boundaries = no_stress.replace(" ","_*_")
#     no_ii = boundaries.replace("iːː", "iː")
#     phonemes = no_ii.split("_")
    
#     #remove 'empty' phonemes
#     while "" in phonemes:
#         phonemes.remove('')

#     return phonemes

# def espeak(text):
#     '''
#     Takes orthograpic sentence as input
#     returns list of phonemes and boundaries
#     boundaries are indicated by an asterisk 
#     '''
#     # remove punctuation and add quotes for espeak
#     chars = []
#     for c in text:
#         if c.isspace() or c.isalnum():
#             chars.append(c)
#     espeaktext = '"' + ''.join(chars) + '"'

#     espeakout = subprocess.check_output(['espeak', '-q', '--ipa=3', '-v', 'en', espeaktext])
#     # strip of newline characters etc    
#     espeakout = espeakout.strip()
#     phonemelist = clean_phonemes(espeakout)  
#     return phonemelist

men_ipa = json.load(open("/home/gchrupala/reimaginet/data/men_ipa.json"))

def espeak(words):
    return men_ipa[words]

# Cosine distance from precomuted representation table
def cos(R, word1, word2, layer=-1):
    return 1-cosine(R[word1][layer,:], R[word2][layer,:])

# Precompute representations

def reps(model, data, conv):
    R = {}
    vocab = set()
    for x in data:
        if x[0] not in vocab:
            vocab.add(x[0])
        if x[1] not in vocab:
            vocab.add(x[1])
    words = [ w for w in vocab]
    convs = [ conv(w) for w in vocab ]
    reps = [ imaginet.task.pile(model, [conv])[0] for conv in convs ]
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

# Christopher P. Matthews
# christophermatthews1985@gmail.com
# Sacramento, CA, USA

def levenshtein(s, t):
        ''' From Wikipedia article; Iterative with two matrix rows. '''
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]

def nlev(s, t):
    if len(s) >= len(t):
        norm = len(s)
    else:
        norm = len(t)
    return levenshtein(s, t)/norm
    
M_word = imaginet.task.load(path="/home/gchrupala/reimaginet/examples/emnlp-2016/word-gru.5.zip")
M_phon = imaginet.task.load(path="/home/gchrupala/reimaginet/examples/emnlp-2016/phon-gru.8.zip")

menurl = "https://raw.githubusercontent.com/kadarakos/wordsims.io/master/wordsims/word-sim-data/EN-MEN-TR-3k.txt"
MEN = judge(menurl)
human_m  = [ x[2] for x in MEN ]

R_phon = reps(M_phon, MEN, espeak)
R_word = reps(M_word, MEN, lambda x: [x])
distance = [ nlev(espeak(x[0]), espeak(x[1])) for x in MEN ]
# Print correlations for MEN

print "PHON"    

with open("cosines.json","w") as out:
    for layer in range(0,3):
        system_m = [ cos(R_phon, x[0], x[1], layer=layer) for x in MEN ]
        out.write(json.dumps(dict(model='phon', layer=layer, sims=system_m)))
        out.write("\n")
        print "Layer", layer+1
        print "MEN", scipy.stats.spearmanr(human_m, system_m)
        print "DIS", scipy.stats.spearmanr(distance, system_m)
    print
    
    print "WORD"
    for layer in range(0,1):
        system_m = [ cos(R_word, x[0], x[1], layer=layer) for x in MEN ]
        out.write(json.dumps(dict(model='word', layer=layer, sims=system_m)))
        out.write("\n")
        print "Layer", layer+1
        print "MEN", scipy.stats.spearmanr(human_m, system_m)
        print "DIS", scipy.stats.spearmanr(distance, system_m)

