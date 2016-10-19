import numpy
import h5py

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', extra_train=True):
    self.root = root
    self.dataset = dataset
    self.audio_kind = audio_kind
    self.data = h5py.File("{}/data/{}/data.hdf5".format(self.root, self.dataset), "r", driver='core')


  def iterImages(self, split='train', shuffle=False):
    ix = range(0, self.data['img'][split].shape[0])
    if shuffle:
      random.shuffle(ix)
    for i in ix:
      img = {}
      img['feat'] = self.data['img'][split][i,:]
      img['sentences'] = []
      img['imgid'] = i
      for j in range(0,5):
        sent = {}
        sent['tokens'] = self.data['txt'][split][i*5+j].split()
        sent['raw'] = ' '.join(sent['tokens'])
        sent['imgid'] = i
        if self.audio_kind is None:
            sent['audio'] = None
        else:
            sent['audio'] = self.data['mfcc'][sent['raw']][...]
        if 'ipa' in self.data:
            sent['ipa'] = self.data['ipa'][sent['raw']][...]
        img['sentences'].append(sent)
      yield img

  def iterSentences(self, split='train', shuffle=False):
    for img in self.iterImages(split=split, shuffle=shuffle):
      for sent in img['sentences']:
        yield sent

  def close(self):
     """Close underlying data source"""
     self.data.close()
        
def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)

