import imaginet.defn.visual2_rhn as visual
import funktional.util as util
import imaginet.task as task
from funktional.layer import params
from funktional.util import autoassign

class Shared(Layer):

    def __init__(self, size_vocab, size_embed, size, depth,
                 recur_depth=1, drop_i=0.0, drop_s=0.0, residual=True, seed=0):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedRHN0(self.size_embed, self.size, depth=self.depth, recur_depth=self.recur_depth,
                                        drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)
    def params(self):
        return params(self.Embed, self.EncoderS, self.EncodeV, self.EncodeT)

    def __call__(self, input):
        return self.Encode(self.Embed(input))

class Visual(task.Task):
    def __init__(self, Shared, size,  ...):
        autoassign(locals())
        self.margin_size = config.get('margin_size', 0.2)
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.EncodeV = StackedRHN0(self.size, self.size, depth=self.depth_spec, recur_depth=self.recur_depth,
                                                drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)
        self.ImgEncode = Dense(config['size_target'], config['size'], init=eval(config.get('init_img', 'orthogonal')))
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()

    def compile(self):
        task.Task.compile(self)
        self.encode_images = self._make_encode_images()

    def params(self):
        return params(self.Shared, self.EncodeV, self.ImgEncoder)

    def __call__(self, input):
        return util.l2norm(last(self.EncodeV(self.Shared(input))))

    def args(self, item):
        return (item['input'], item['target_v'])



class Textual(task.Task):
    def __init__(self, Shared, ...):
        autoassign(locals())
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.EncodeT = StackedRHN0(self.size, self.size, depth=self.depth_spec, recur_depth=self.recur_depth,
                                                drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)
        self.DecodeT = StackedRHN(self.size, self.size, depth=self.depth_spec, recur_depth=self.recur_depth,
                                                drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)
        self.ToTxt = Dense(size, size_vocab)
        self.inputs = [T.imatrix(), T.imatrix()]
        self.target = T.matrix()

    def params(self):
        return params(self.Shared, self.EncodeT, self.DecodeT)

    def __call__(self, input, prox):
        h0 = last(self.EncodeT(self.Shared(input)))
        return softmax3d(self.ToText(self.DecodeT(h0, self.EncodeT(self.Shared(prox)))))

    def cost(self, target, prediction):
        oh = OneHot(size_in=self.size_vocab)
        return util.CrossEntropy(oh(target), prediction)

    def args(self, item):
        return (item['input'], item['target_prev_t'], item['target_t'])


class Vistext(task.Bundle):

    def __init__(self, data, config, weights=None):
        self.config = config
        self.data = data
        self.batcher = data['batcher']
        self.scaler = data['scaler']
        self.config['size_vocab'] = self.data['batcher'].mapper.size()
        self.network = Network(config['size_vocab'], config['size_embed'], config['size'],
                          config['size_target'], config['depth'])
        self.visual = Visual(self.network, config)
        self.textual = Textual(self.network, config)
        if weights is not None:
            assert len(self.network.params())==len(weights)
            for param, weight in zip(self.params(), weights):
                param.set_value(weight)
        self.visual.compile()
        self.textual.compile()
        self.visual.representation = self.visual._make_representation()
        self.visual.pile = self.visual._make_pile()

    def params(self):
        return self.network.params()

    def get_config(self):
        return self.config

    def get_data(self):
        return self.data

def load(path):
    """Load data and reconstruct model."""
    with zipfile.ZipFile(path,'r') as zf:
        buf = StringIO.StringIO(zf.read('weights.npy'))
        weights = numpy.load(buf)
        config  = json.loads(zf.read('config.json'))
        data  = pickle.loads(zf.read('data.pkl'))
    return Vistext(data, config, weights=weights)


def trainer(model, data, epochs, validate_period, model_path, prob_lm=0.1):
    def valid_loss():
        result = dict(lm=[], visual=[])
        for item in data.iter_valid_batches():
            result['lm'].append(model.lm.loss_test(*model.lm.args(item)))
            result['visual'].append(model.visual.loss_test(*model.visual.args(item)))
        return result
    costs = Counter(dict(cost_v=0.0, N_v=0.0, cost_t=0.0, N_t=0.0))
    print "LM: {} parameters".format(count_params(model.lm.params()))
    print "Vi: {} parameters".format(count_params(model.visual.params()))
    for epoch in range(1,epochs+1):
        for _j, item in enumerate(data.iter_train_batches()):
            j = _j +1
            if random.random() <= prob_lm:
                cost_t = model.lm.train(*model.lm.args(item))
                costs += Counter(dict(cost_t=cost_t, N_t=1))
            else:
                cost_v = model.visual.train(*model.visual.args(item))
                costs += Counter(dict(cost_v=cost_v, N_v=1))
            print epoch, j, j*data.batch_size, "train", \
                    numpy.divide(costs['cost_v'], costs['N_v']),\
                    numpy.divide(costs['cost_t'], costs['N_t'])
            if j % validate_period == 0:
                result = valid_loss()
                print epoch, j, 0, "valid", \
                    numpy.mean(result['visual']),\
                    numpy.mean(result['lm'])
                sys.stdout.flush()
        model.save(path='model.{0}.zip'.format(epoch))
    model.save(path='model.zip')
