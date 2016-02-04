import json
from imaginet.commands import train, evaluate, compressed
from funktional.util import linear, clipped_rectify, CosineDistance
dataset = 'flickr8k'
datapath = "."
epochs = 1


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
      tokenize=compressed,
      validate_period=64*1000)

for epoch in range(1, epochs+1):
    
    scores = evaluate(dataset=dataset,
                      datapath=datapath,
                      model_path='.',
                      model_name='model.{}.pkl.gz'.format(epoch))
    json.dump(scores, open('scores.{}.json'.format(epoch),'w'))
