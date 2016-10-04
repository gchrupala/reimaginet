import numpy
import imaginet.vendrov_provider as vendrov
import imaginet.simple_data as sd
import imaginet.task
import imaginet.defn.vectorsum2 as vs
from imaginet.evaluate import ranking
import json
from imaginet.simple_data import words
import random
from collections import Counter
import sys

def run_train(data, prov, model_config, run_config):
    seed  = run_config.get('seed')
    if  seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
    model = imaginet.task.GenericBundle(dict(scaler=data.scaler,
                                             batcher=data.batcher), model_config, run_config['task'])

    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            result.append(model.task.loss_test(*model.task.args(item)))
        return result

    costs = Counter()
    for epoch in range(1, run_config['epochs'] + 1):
        random.shuffle(data.data['train'])
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                cost = model.task.train(*model.task.args(item))
                costs += Counter({'cost':cost, 'N':1})
                print epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])])
                if j % run_config['validate_period'] == 0:
                        print epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))])
                sys.stdout.flush()
        model.save(path='model.{0}.zip'.format(epoch))
    model.save(path='model.zip')



def run_eval(prov, config):
    datapath='/home/gchrupala/repos/reimaginet'

    for epoch in range(1, 1+config['epochs']):
        scores = evaluate(prov,
                          datapath=datapath,
                          tokenize=config['tokenize'],
                          split=config['split'],
                          task=config['task'],
                          batch_size=config['batch_size'],
                          model_path='model.{}.zip'.format(epoch))
        json.dump(scores, open('scores.{}.json'.format(epoch),'w'))
        print epoch, numpy.mean(scores['recall'][5])



def evaluate(prov, 
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             task=vs.VectorSum,
             tokenize=words,
             split='val'
            ):
    model = imaginet.task.load(path=model_path)
    task = model.task
    scaler = model.scaler
    batcher = model.batcher
    mapper = batcher.mapper
    sents = list(prov.iterSentences(split=split))
    sents_tok =  [ tokenize(sent) for sent in sents ]
    predictions = imaginet.task.encode_sentences(model, sents_tok, batch_size=batch_size)
    images = list(prov.iterImages(split=split))
    img_fs = imaginet.task.encode_images(model, [ img['feat'] for img in images ])
    #img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sents)) ] )
    return ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)


