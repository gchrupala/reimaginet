import os
import numpy
import json
import sys

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', extra_train=True):
    self.root = root
    self.dataset = dataset
    self.img = {}
    self.txt = {}
    self.img['train'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/train.npy".format(self.root, self.dataset)))
    self.img['val'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/val.npy".format(self.root, self.dataset)))
    self.img['test'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/test.npy".format(self.root, self.dataset)))

    self.txt['train'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/train.txt".format(self.root, self.dataset)) ]
    self.txt['val'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/val.txt".format(self.root, self.dataset)) ]
    self.txt['test'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/test.txt".format(self.root, self.dataset)) ]
  
    audio_path = "{}/data/{}/dataset.mfcc.npy".format(self.root, self.dataset)
    try:
      
        words = json.load(open("{}/data/{}/dataset.words.json".format(self.root, self.dataset)))
        self.AUDIO = numpy.load(audio_path)
        self.w2a = {}
        for i in range(0, len(words)):
            self.w2a[words[i]] = i 
    except IOError:
        sys.stderr.write("Could not read file {}: audio features not available\n".format(audio_path))


  def iterImages(self, split='train', shuffle=False):
    ix = range(0, self.img[split].shape[0])
    if shuffle:
      random.shuffle(ix)
    for i in ix:
      img = {}
      img['feat'] = self.img[split][i,:]
      img['sentences'] = []
      img['imgid'] = i
      for j in range(0,5):
        sent = {}
        sent['tokens'] = self.txt[split][i*5+j]
        sent['raw'] = ' '.join(sent['tokens'])
        sent['imgid'] = i
        sent['audio'] = numpy.log(numpy.exp(1+self.AUDIO[self.w2a[' '.join(sent['tokens'])]]))
        img['sentences'].append(sent)
      yield img

  def iterSentences(self, split='train', shuffle=False):
    for img in self.iterImages(split=split, shuffle=shuffle):
      for sent in img['sentences']:
        yield sent

        
      
    
