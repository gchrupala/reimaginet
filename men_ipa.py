# coding: utf-8

from urllib2 import urlopen
from subprocess import check_output
import json

def espeak(words):
    return phon(check_output(["espeak", "-q", "--ipa=3",
                              "-v", "en", words]).decode("utf-8"))
def phon(inp):
    # get rid of accents
    return [ ph.replace(u"ˈ","").replace(u"ˌ","") for word in inp.split() for ph in word.split("_") ]

def judge(url):
    result = []
    f = urlopen(url)
    for line in f:
        fields = line.split("\t")
        if len(fields) == 3:
            result.append((fields[0], fields[1], float(fields[2])))
    return result

def make_ipa(word):
    D = {}
    for word in words:
        D[word]=espeak(word)
    return D

menurl = "https://raw.githubusercontent.com/kadarakos/wordsims.io/master/wordsims/word-sim-data/EN-MEN-TR-3k.txt"
words = []
for item in judge(menurl):
    words.append(item[0])
    words.append(item[1])

json.dump(make_ipa(words), open("data/men_ipa.json","w"))


