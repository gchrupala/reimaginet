#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import json
import sys
import gzip

##############################################################################

def clean_phonemes(espeakoutput):
    '''
    Takes espeak output as input
    returns list of phonemes and boundaries
    boundaries are indicated by an asterisk
    '''
    no_word_stress = espeakoutput.replace("ˈ", "")    
    no_stress = no_word_stress.replace("ˌ", "")
    boundaries = no_stress.replace(" ","_*_")
    no_ii = boundaries.replace("iːː", "iː")
    phonemes = no_ii.split("_")
    
    #remove 'empty' phonemes
    while "" in phonemes:
        phonemes.remove('')

    return phonemes

def texttophonemes(text):
    '''
    Takes orthograpic sentence as input
    returns list of phonemes and boundaries
    boundaries are indicated by an asterisk 
    '''
    # remove punctuation and add quotes for espeak
    chars = []
    for c in text:
        if c.isspace() or c.isalnum():
            chars.append(c)
    espeaktext = '"' + ''.join(chars) + '"'

    espeakout = subprocess.check_output(['espeak', '-q', '--ipa=3', '-v', 'en', espeaktext])
    # strip of newline characters etc    
    espeakout = espeakout.strip()
    phonemelist = clean_phonemes(espeakout)  
    return phonemelist
    
##############################################################################
def main():
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    with open(inputfile) as f:    
        data = json.load(f)

    with gzip.open(outputfile,"w") as f:
        for image in data['images']:
            for caption in image["sentences"]:
                phonemes = texttophonemes(caption["raw"])
                f.write("{}\n".format(json.dumps({'sentid':caption['sentid'], 'phonemes':phonemes})))
