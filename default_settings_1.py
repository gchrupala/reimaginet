from imaginet.driver import cmd_train, cmd_predict_v, cmd_eval
from imaginet.models import MultitaskLMA
from funktional.util import linear, MeanSquaredError, clipped_rectify, CosineDistance
dataset = 'coco'
epochs = 10

cmd_train( dataset=dataset,
           datapath='/home/gchrupala/repos/neuraltalk/',
           model_path='.',
           epochs=epochs, depth=1, alpha=0.1, max_norm=5.0,
           cost_visual=CosineDistance,
           gru_activation=clipped_rectify,
           architecture=MultitaskLMA,
           hidden_size=1024,
           embedding_size=1024,
           with_para='auto',
           scaler='standard',
           visual_activation=linear,
           shuffle=True,
           pad_end=False,
           batch_size=64,
           validate_period=64*1000)
for epoch in range(1,epochs+1):
    cmd_predict_v(dataset=dataset,
                  datapath='/home/gchrupala/repos/neuraltalk/',
                  model_path='.',
                  model_name='model.{}.pkl.gz'.format(epoch),
                  output_v='predict_v.{}.npy'.format(epoch),
                  output_r='predict_r.{}.npy'.format(epoch))
    cmd_eval(dataset=dataset,
             datapath='/home/gchrupala/repos/neuraltalk/',
             input_v='predict_v.{}.npy'.format(epoch),
             input_r='predict_r.{}.npy'.format(epoch),
             output='eval.{}.json'.format(epoch))


    
