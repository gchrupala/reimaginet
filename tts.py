import json
import gzip
from gtts import gTTS
import StringIO
import base64
import pydub
import features
import StringIO
import scipy.io.wavfile as wav
import numpy
import features
import time
from urllib2 import HTTPError

def speak(words):
    f = StringIO.StringIO()
    gTTS(text=words, lang='en-us').write_to_fp(f)
    return f.getvalue()

def tryspeak(words,i):
    try:
        return speak(words)
    except:
        print "sleeping {} seconds".format(60*i)
        time.sleep(60*i)
        return tryspeak(words,i*2)
        
def tts(dataset='flickr8k'):
    data = json.load(open('/home/gchrupala/repos/reimaginet/data/{}/dataset.json'.format(dataset)))

    with gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset.mp3.jsonl.gz".format(dataset),"w") as f:
        for img in data['images']:
            for s in img['sentences']:
                    audio = tryspeak(s['raw'],1)
                    f.write("{}\n".format(json.dumps({'sentid':s['sentid'], 'speech':base64.b64encode(audio)})))



def decodemp3(s):
    seg = pydub.AudioSegment.from_mp3(StringIO.StringIO(s))
    io = StringIO.StringIO()
    seg.export(io, format='wav')
    return io.getvalue()

# def wavdata(dataset='flickr8k'):
#     with gzip.open("/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.wav.jsonl.gz","w") as out:
#         for line in gzip.open("/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.mp3.jsonl.gz"):
#             sent = json.loads(line)
#             out.write("{}\n".format(json.dumps({'sentid':sent['sentid'],
#                                                 'wav': base64.b64encode(decodemp3(base64.b64decode(sent['speech'])))})))
            

def extract_mfcc(sound):
    (rate,sig) = wav.read(StringIO.StringIO(sound))
    mfcc_feat = features.mfcc(sig,rate)
    return mfcc_feat

def extract_fbank(sound):
    (rate,sig) = wav.read(StringIO.StringIO(sound))
    fbank_feat = features.logfbank(sig,rate)
    return fbank_feat

def featurefile(dataset='flickr8k'):
    result_mfcc = []
    result_fbank = []
    for line in gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset.mp3.jsonl.gz".format(dataset)):
        sent = json.loads(line)
        sound = decodemp3(base64.b64decode(sent['speech']))
        result_mfcc.append(extract_mfcc(sound))
#        result_fbank.append(extract_fbank(sound))
#        if len(result_mfcc) > 200:
#            break
    numpy.save("/home/gchrupala/repos/reimaginet/data/{}/dataset.mfcc.npy".format(dataset), result_mfcc)
#    numpy.save("/home/gchrupala/repos/reimaginet/data/flickr8k/speech-fbank.npy", result_fbank)
