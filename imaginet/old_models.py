from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             EncoderDecoderGRU, Embedding, OneHot, \
                             last, softmax3d, WithDropout
import funktional.util as util
from funktional.util import CosineDistance, CrossEntropy, tanh, linear, clipped_rectify
from funktional.util import autoassign, params
import funktional.context as context
import theano.tensor as T
import theano

class Activation(Layer):
    """Activation function object."""
    def __init__(self, activation):
        autoassign(locals())
        self.params = []

    def __call__(self, inp):
        return self.activation(inp)
    
class DenseStack(Layer):
    """Stack of fully connected layers."""
    def __init__(self, size_in, size, depth=2, dropout_prob=0.0, activation=tanh):
        autoassign(locals())
        self.bottom = Dense(self.size_in, self.size)
        layers = [ Dense(self.size, self.size) for _ in range(1, self.depth) ]
        self.stack = reduce(lambda z, x: \
                              x.compose(WithDropout(Activation(self.activation).compose(z), self.dropout_prob)), \
                            layers, \
                            self.bottom)
        self.params = self.stack.params

    def __call__(self, inp):
            return self.stack(inp)

class Visual(Layer):
    """Encode sequence of inputs into a visual vector."""

    def __init__(self, size_embed, size, size_out, depth, out_depth=1, gru_activation=tanh, dropout_prob=0.0):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth,
                                    activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Project = Dense(self.size, self.size_out)
        self.params = params(self.Encode, self.Project)

    def __call__(self, inp):
        return self.Project(last(self.Encode(inp)))

class LM(Layer):
    """Predict next word in sequence of outputs.

    Ignores input.
    """

    def __init__(self, size_embed, size, size_out, depth, gru_activation=tanh, dropout_prob=0.0):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth,
                                    activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Predict = Dense(self.size, self.size_embed)
        self.params = params(self.Encode, self.Predict)

    def __call__(self, _inp, out_prev, _img): # Decodes output from scratch (ignores input)
        return self.Predict(self.Encode(out_prev))

class LMC(Layer):
    """Predict next word in sequence of outputs, conditioned on image

    Ignores input.
    """

    def __init__(self, size_embed, size, size_out, depth, gru_activation=tanh, dropout_prob=0.0):
        autoassign(locals())
        self.Encode  = StackedGRU(self.size_embed, self.size, self.depth,
                                    activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.FromImg = Dense(self.size_out, self.size)
        self.Predict = Dense(self.size, self.size_embed)
        self.params = params(self.Encode, self.FromImg, self.Predict) 

    def __call__(self, _inp, out_prev, img): # Decodes output, conditioned on image
        return self.Predict(self.Encode(self.FromImg(img), out_prev))

class ED(Layer):
    """Encode a sequence of inputs, and decode into a sequence of outputs.

    Decoder is conditioned on the final state of the encoder, and output at position -1.
    """

    def __init__(self, size_embed, size, size_out, depth, gru_activation=tanh, dropout_prob=0.0):
        autoassign(locals())
        encoder = lambda size_in, size:\
                  StackedGRUH0(size_embed, size, self.depth,
                               activation=self.gru_activation, dropout_prob=self.dropout_prob)
        decoder = lambda size_in, size: \
                  StackedGRU(size_embed, size, self.depth,
                             activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Predict   = Dense(size_in=self.size, size_out=self.size_embed)
        self.params    = params(self.Encdec, self.Predict)

    def __call__(self, inp, out_prev, _img):
        return self.Predict(self.Encdec(inp, out_prev))


class Multitask(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth, textual,
                 out_depth=1,
                 gru_activation=tanh,
                 visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  = Visual(self.size_embed, self.size, self.size_out, self.depth, out_depth=self.out_depth,
                              gru_activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Textual = textual(self.size_embed, self.size, self.size_out, self.depth,
                               gru_activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.params  = params(self.Embed, self.Visual, self.Textual)

        
    def __call__(self, inp, output_prev, img):
        inp_e = self.Embed(inp)
        output_prev_e  = self.Embed(output_prev)
        img_pred   = self.visual_activation(self.Visual(inp_e))
        txt_pred   = softmax3d(self.Embed.unembed(self.Textual(inp_e, output_prev_e, img)))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input], self.visual_activation(self.Visual(self.Embed(input))))

class MultitaskY(Layer):
    """Joint Encode + separate pathways."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth, textual,
                 out_depth=1,
                 gru_activation=tanh,
                 visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed   = Embedding(self.size_vocab, self.size_embed)
        self.Joint   = StackedGRUH0(self.size_embed, self.size, self.depth,
                                    activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Visual  = Visual(self.size, self.size, self.size_out, self.depth, out_depth=self.out_depth,
                              gru_activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.Textual = textual(self.size, self.size, self.depth,
                               gru_activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.params  = params(self.Embed, self.Visual, self.Textual)

    def __call__(self, inp, output_prev, _img):
        inp_e = self.Joint(self.Embed(inp))
        output_prev_e  = self.Embed(output_prev)
        img   = self.visual_activation(self.Visual(inp_e))
        txt   = softmax3d(self.Embed.unembed(self.Textual(inp_e, output_prev_e, _img)))
        return (img, txt)
    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input], self.visual_activation(self.Visual(self.Embed(input))))
    
class MultitaskMM(Layer):
    """Shared recurrent encoder with visual decoder + textual decoder."""
    def __init__(self, size_vocab, size_embed, size, size_out, depth, out_depth=1, # FIXME USE THIS PARAM
                 gru_activation=tanh, visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedGRUH0(self.size_embed, self.size, self.depth,
                                   activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.DecodeT = StackedGRU(self.size_embed, self.size, self.depth,
                                  activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.PredictT   = Dense(size_in=self.size, size_out=self.size_embed)
        self.DecodeV = Dense(self.size, self.size_out)
        self.params = params(self.Embed, self.DecodeT, self.PredictT, self.DecodeV) 
        
    def __call__(self, inp, out_prev, _img):
        rep = last(self.Encode(self.Embed(inp)))
        img = self.visual_activation(self.DecodeV(rep))
        txt = softmax3d(self.Embed.unembed(self.PredictT(self.DecodeT(rep, self.Embed(out_prev)))))
        return (img, txt)
    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input], self.visual_activation(self.DecodeV(last(self.Encode(self.Embed(input))))))


class MultitaskBD(Layer):
    """Text to image and back."""
    def __init__(self, size_vocab, size_embed, size, size_out, depth, out_depth=1, # FIXME USE THIS PARAM
                 gru_activation=tanh, visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Encode = StackedGRUH0(self.size_embed, self.size, self.depth,
                                   activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.ToVis   = Dense(self.size, self.size_out)
        self.FromVis = Dense(self.size_out, self.size)
        self.Decode = StackedGRU(self.size_embed, self.size, self.depth,
                                 activation=self.gru_activation, dropout_prob=self.dropout_prob)
        self.PredictT   = Dense(size_in=self.size, size_out=self.size_embed)
        self.params = params(self.Embed, self.Encode, self.ToVis, self.FromVis, self.Decode, self.PredictT)
        
    def __call__(self, inp, out_prev, img):
        img_out = self.visual_activation(self.ToVis(last(self.Encode(self.Embed(inp)))))
        txt_out = softmax3d(self.Embed.unembed(self.PredictT(self.Decode(self.visual_activation(self.FromVis(img)),
                                                                   self.Embed(out_prev)))))
        return (img_out, txt_out)
    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input = T.imatrix()
        return theano.function([input], self.visual_activation(self.ToVis(last(self.Encode(self.Embed(input))))))

def MultitaskLM(size_vocab, size_embed, size, size_out, depth, **kwargs):
    """Visual encoder combined with a language model."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, LM, **kwargs)


def MultitaskLMC(size_vocab, size_embed, size, size_out, depth, **kwargs):
    """Visual encoder combined with a language model."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, LMC, **kwargs)

def MultitaskYLM(size_vocab, size_embed, size, size_out, depth, **kwargs):
    """Visual encoder combined with a language model."""
    return MultitaskY(size_vocab, size_embed, size, size_out, depth, LM, **kwargs)

def MultitaskED(size_vocab, size_embed, size, size_out, depth, **kwargs):
    """Visual encoder combined with a recurrent encoder-decoder."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, ED, **kwargs)

def MultitaskYED(size_vocab, size_embed, size, size_out, depth, **kwargs):
    """Visual encoder combined with a recurrent encoder-decoder."""
    return MultitaskY(size_vocab, size_embed, size, size_out, depth, ED, **kwargs)

        
class Imaginet(object):
    """Trainable imaginet model."""

    def __init__(self, size_vocab, size_embed, size, size_out, depth, network, alpha=0.5,
                 out_depth=1,
                 gru_activation=tanh, visual_activation=linear, cost_visual=CosineDistance,
                 max_norm=None, dropout_prob=0.0):
        autoassign(locals())
        self.network = network(self.size_vocab, self.size_embed, self.size, self.size_out, self.depth, 
                               out_depth=self.out_depth,
                               gru_activation=self.gru_activation, visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob )
                               
        input         = T.imatrix()
        output_t_prev = T.imatrix()
        output_t      = T.imatrix()
        output_v      = T.fmatrix()
        self.OH       = OneHot(size_in=self.size_vocab)
        output_t_oh   = self.OH(output_t)
        # TRAINING
        with context.context(training=True):
            output_v_pred, output_t_pred = self.network(input, output_t_prev, output_v)
            cost_T = CrossEntropy(output_t_oh, output_t_pred)
            cost_V = self.cost_visual(output_v, output_v_pred)
            cost = self.alpha * cost_T + (1.0 - self.alpha) * cost_V
        #TESTING
        with context.context(training=False):
            output_v_pred_test, output_t_pred_test = self.network(input, output_t_prev, output_v)
            cost_T_test = CrossEntropy(output_t_oh, output_t_pred_test)
            cost_V_test = self.cost_visual(output_v, output_v_pred_test)
            cost_test = self.alpha * cost_T_test + (1.0 - self.alpha) * cost_V_test
        self.updater = util.Adam(max_norm=self.max_norm)
        updates = self.updater.get_updates(self.network.params, cost)
        # TODO better way of dealing with needed/unneeded output_t_prev?
        self.train = theano.function([input, output_v, output_t_prev, output_t ], [cost, cost_T, cost_V],
                                     updates=updates, on_unused_input='warn')

        self.loss_test = theano.function([input, output_v, output_t_prev, output_t ],
                                         [cost_test, cost_T_test, cost_V_test],
                                    on_unused_input='warn')
#        self.predict = theano.function([input, output_t_prev], [output_v_pred_test, output_t_pred_test],
#                                       on_unused_input='warn')


# Functions added outside the class do not interfere with loading of older versions
def predictor_v(model):
    """Return function to predict image vector from input using `model`."""
    return model.network.predictor_v()

    
