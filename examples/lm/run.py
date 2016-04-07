import json
from imaginet.commands import train, evaluate
#from imaginet.defn.visual import Visual
from imaginet.defn.lm import LM
from imaginet.simple_data import phonemes, characters
from funktional.util import linear, clipped_rectify, CosineDistance
import numpy
dataset = 'coco'
datapath = "/home/gchrupala/repos/reimaginet"
epochs = 10
tokenize=phonemes

if True:
    train(dataset=dataset,
      datapath=datapath,
      model_path='.',
      task=LM,
      epochs=epochs,
      min_df=10,
      max_norm=5.0,
      scale=True,
      batch_size=64,
      shuffle=True,
      size_embed=256,
      size_hidden=1024,
      depth=3,
      tokenize=tokenize,
      validate_period=100,
      seed = 41)

