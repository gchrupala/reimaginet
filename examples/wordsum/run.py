import imaginet.simple_data as sd
import imaginet.experiment as E
import imaginet.vendrov_provider as vendrov
import imaginet.defn.vectorsum2 as vs
dataset = 'coco'
batch_size = 128
prov = vendrov.Provider(dataset, root='/home/gchrupala/repos/reimaginet/', audio_kind=None)
data = sd.SimpleData(prov, min_df=1, scale=False,
                     batch_size=batch_size, shuffle=True,
                     tokenize=sd.words, val_vocab=False)
model_config = dict(size_embed=300, size=300, depth=None, max_norm=2.0,
              lr=0.001, size_target=4096, contrastive=True, margin_size=0.2,
              fixed=True, init_img='xavier')
run_config = dict(seed=51, task=vs.VectorSum, epochs=20, validate_period=1000)

E.run_train(data, prov, model_config, run_config)

eval_config = dict(tokenize=sd.words, split='val', task=vs.VectorSum, batch_size=batch_size, epochs=20)

E.run_eval(prov, eval_config)
