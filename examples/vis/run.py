import json
from imaginet.commands import train, evaluate
from imaginet.defn.visual import Visual
from imaginet.simple_data import phonemes, characters
from funktional.util import linear, clipped_rectify, CosineDistance
import numpy
dataset = 'coco'
datapath = "/home/gchrupala/repos/reimaginet"
epochs = 9
tokenize=phonemes

train(dataset=dataset,
      datapath=datapath,
      model_path='.',
      task=Visual,
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

for epoch in range(1,epochs+1):
    
    scores = evaluate(dataset=dataset,
                      datapath=datapath,
                      tokenize=tokenize,
                      batch_size=64,
                      model_path='model.{}.zip'.format(epoch))
    json.dump(scores, open('scores.{}.json'.format(epoch),'w'))
    print epoch, numpy.mean(scores['recall'][5])

