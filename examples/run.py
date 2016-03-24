import json
from imaginet.commands import train, evaluate
from imaginet.simple_data import phonemes
from funktional.util import linear, clipped_rectify, CosineDistance
dataset = 'coco'
datapath = "/home/gchrupala/repos/reimaginet"
epochs = 10


train(dataset=dataset,
      datapath=datapath,
      model_path='.',
      epochs=epochs,
      min_df=10,
      max_norm=5.0,
      scale=True,
      batch_size=64,
      shuffle=True,
      size_embed=256,
      size_hidden=1024,
      depth=3,
      tokenize=phonemes,
      validate_period=64*1000)

for epoch in range(7, 8):
    
    scores = evaluate(dataset=dataset,
                      datapath=datapath,
                      tokenize=phonemes,
                      batch_size=64,
                      model_path='model.{}.zip'.format(epoch))
    json.dump(scores, open('scores.{}.json'.format(epoch),'w'))
