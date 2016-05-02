from funktional.layer import Layer, Dense, BidiGRU, BidiGRUH0, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot, Dropout, Sum, \
                             first, last, softmax3d, params
import funktional.util as util
from funktional.util import CosineDistance, CrossEntropy, linear, clipped_rectify
from funktional.util import autoassign
import funktional.context as context
import theano.tensor as T
import theano

class Activation(Layer):
    """Activation function object."""
    def __init__(self, activation):
        autoassign(locals())

    def params():
        return []
    
    def __call__(self, inp):
        return self.activation(inp)

class SumAdapter(Layer):
    # Adapt Sum to be compatible with  with StackedGRUH0. Depth and activation 
    # will be ignored
    # FIXME nicer way of doing this?
    def __init__(self, size_embed, size, 
                 depth=None,               
                 activation=None,
                 dropout_prob=0.0):
        autoassign(locals())
        assert self.size_embed == self.size
        self.Dropout0 = Dropout(prob=self.dropout_prob)
        self.Sum = Sum(self.size)

    def params(self):
        return params(self.Dropout0, self.Sum)

    def __call__(self, seq):
        return self.Sum(self.Dropout0(seq))

class Visual(Layer):
    def __init__(self, size_embed, size, size_out, depth, encoder=StackedGRUH0, 
                 gru_activation=clipped_rectify,
                 visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Encode = encoder(self.size_embed, self.size, self.depth,
                              activation=self.gru_activation,
                              dropout_prob=self.dropout_prob)
        self.ToImg   = Dense(self.size, self.size_out)

    def params(self):
        return params(self.Encode, self.ToImg)

    def __call__(self, inp):
        return self.visual_activation(self.ToImg(self.Encode(inp)))

    def encode(self, inp):
        return self.Encode(inp)

class VisualBidi(Layer):
    def __init__(self, size_embed, size, size_out, 
                 gru_activation=clipped_rectify,
                 visual_activation=linear):
        autoassign(locals())
        self.Encode = BidiGRUH0(self.size_embed, self.size, 
                              activation=self.gru_activation)
        self.ToImg   = Dense(self.size, self.size_out)

    def params(self):
        return params(self.Encode, self.ToImg)

    def __call__(self, inp):
        return self.visual_activation(self.ToImg(self.encode(inp)))

    def encode(self, inp):
        return bidi_encode(self.Encode, inp)

def bidi_encode(layer, inp):
    """Return encoded input from a bidirectional encoder."""
    f,b = layer.bidi(inp)
    return last(f) + first(b)

    
class MultitaskLMA(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_encoder=StackedGRUH0, visual_activation=linear, dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth, 
                               encoder=self.visual_encoder,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.LM      =  StackedGRUH0(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation,
                                     dropout_prob=self.dropout_prob
                                     )
        self.ToTxt   =  Dense(self.size, self.size_vocab) # map to vocabulary

    def params(self):
        return params(self.Embed, self.Visual, self.LM, self.ToTxt)

    def grow(self, ps):
        self.LM.layer.grow(ps)
        self.Visual.Encode.layer.grow(ps)
        
    def __call__(self, inp, output_prev, _img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.ToTxt(self.LM(self.Embed(output_prev))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskBiLMA(Layer):
    """Visual encoder combined with a textual task (bidirectional GRUs)"""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_encoder=BidiGRUH0, visual_activation=linear, dropout_prob=0.0):
        autoassign(locals())
        assert self.depth == 1
        assert self.dropout_prob == 0.0
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  VisualBidi(self.size_embed, self.size, self.size_out, 
                                   gru_activation=self.gru_activation,
                                   visual_activation=self.visual_activation
                                   )
        self.LM      =  BidiGRUH0(self.size_embed, self.size, 
                                  activation=self.gru_activation
                                  )
        self.ToTxt   =  Dense(self.size, self.size_vocab) # map to vocabulary

    def params(self):
        return params(self.Embed, self.Visual, self.LM, self.ToTxt)

    def grow(self, ps):
        self.LM.layer.grow(ps)
        self.Visual.Encode.layer.grow(ps)
        
    def __call__(self, inp, output_prev, _img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.ToTxt(self.LM(self.Embed(output_prev))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskLMX(Layer):
    """Visual encoder, no textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_encoder=StackedGRUH0, visual_activation=linear, dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth, 
                               encoder=self.visual_encoder,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.OH      = OneHot(self.size_vocab)

    def params(self):
        return params(self.Embed, self.Visual)

    def grow(self, ps):
        self.Visual.Encode.layer.grow(ps)
        
    def __call__(self, inp, output_prev, _img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.OH(output_prev)) # fake output prediction
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskLM(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear, visual_encoder=StackedGRUH0, dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth,
                               encoder=self.visual_encoder,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.LM      =  StackedGRUH0(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation,
                                     dropout_prob=self.dropout_prob)
        self.ToTxt   =  Dense(self.size, self.size_embed) # map to embeddings

    def params(self):
        return params(self.Embed, self.Visual, self.LM, self.ToTxt)

                
    def __call__(self, inp, output_prev, _img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(self.Embed(output_prev)))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskLMC(Layer):
    """Visual encoder combined with a textual decoder."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear, visual_encoder=StackedGRUH0,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth,
                               encoder=self.visual_encoder,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.LM      =  StackedGRU(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation,
                                   dropout_prob=self.dropout_prob)
        self.FromImg =  Dense(self.size_out, self.size)
        self.ToTxt   =  Dense(self.size, self.size_embed) # try direct softmax

    def params(self):
        return params(self.Embed, self.Visual, self.LM, self.FromImg, self.ToTxt)

        
    def __call__(self, inp, output_prev, img):
        img_pred   = self.Visual(self.Embed(inp))
        txt_pred   = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(self.FromImg(img),
                                                                     self.Embed(output_prev)))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskLMD(Layer):
    """Alternative visual encoder combined with a textual decoder.

    Textual decoder starts from final state of encoder instead of from image.
"""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth,
                 gru_activation=clipped_rectify,
                 visual_activation=linear, visual_encoder=StackedGRUH0, dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  =  Visual(self.size_embed, self.size, self.size_out, self.depth,
                               encoder=self.visual_encoder,
                               gru_activation=self.gru_activation,
                               visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.LM      =  StackedGRU(self.size_embed, self.size, self.depth,
                                     activation=self.gru_activation,
                                   dropout_prob=self.dropout_prob)
        self.ToTxt   =  Dense(self.size, self.size_embed) # try direct softmax

    def params(self):
        return params(self.Embed, self.Visual, self.LM, self.ToTxt)

        
    def __call__(self, inp, output_prev, _img):
        rep        = self.Visual.encode(self.Embed(inp))
        img_pred   = self.Visual.visual_activation(self.Visual.ToImg(rep))
        txt_pred   = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(rep,
                                                                     self.Embed(output_prev)))))
        return (img_pred, txt_pred)

    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input],
                         self.Visual(self.Embed(input)))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], self.Visual.encode(self.Embed(input)))

class MultitaskLMY(Layer):
    """Alternative visual encoder combined with a textual decoder.

    Textual decoder starts from final state of encoder instead of from
    image. Shared hidden layer plus specialized layers.
    """
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth, depth_spec=1,
                 visual_encoder=StackedGRUH0,
                 gru_activation=clipped_rectify,
                 visual_activation=linear,
                 dropout_prob=0.0):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Shared  =  StackedGRUH0(self.size_embed, self.size, self.depth, activation=self.gru_activation,
                                     dropout_prob=self.dropout_prob)
        self.Visual  =  Visual(self.size, self.size, self.size_out, self.depth_spec,
                               encoder=self.visual_encoder,
                                gru_activation=self.gru_activation,
                                visual_activation=self.visual_activation,
                               dropout_prob=self.dropout_prob)
        self.LM      =  StackedGRU(self.size, self.size, self.depth_spec,
                                     activation=self.gru_activation,
                                   dropout_prob=self.dropout_prob)
        self.ToTxt   =  Dense(self.size, self.size_embed) # try direct softmax


    def params(self):
        return params(self.Embed, self.Shared, self.Visual, self.LM, self.ToTxt)


    def __call__(self, inp, output_prev, _img):
        shared = self.Shared(self.Embed(inp))
        img_pred = self.Visual(shared)
        txt_pred = softmax3d(self.Embed.unembed(self.ToTxt(self.LM(last(shared), self.Embed(output_prev)))))
        return (img_pred, txt_pred)
    
    def predictor_v(self):
        """Return function to predict image vector from input."""
        input    = T.imatrix()
        return theano.function([input], self.Visual(self.Shared(self.Embed(input))))

    def predictor_r(self):
        """Return function to predict representation from input."""
        input = T.imatrix()
        return theano.function([input], last(self.Shared(self.Embed(input))))
        
class Imaginet(object):
    """Trainable imaginet model."""

    def __init__(self, size_vocab, size_embed, size, size_out, depth, network, 
                 alpha=0.5,
                 gru_activation=clipped_rectify, 
                 visual_activation=linear, 
                 visual_encoder=StackedGRUH0, 
                 cost_visual=CosineDistance,
                 max_norm=None, 
                 lr=0.0002, 
                 dropout_prob=0.0):
        autoassign(locals())
        self.network = network(self.size_vocab, 
                               self.size_embed, 
                               self.size, 
                               self.size_out, 
                               self.depth,
                               gru_activation=self.gru_activation, 
                               visual_activation=self.visual_activation,
                               visual_encoder=self.visual_encoder,
                               dropout_prob=self.dropout_prob)
                               
        self.input         = T.imatrix()
        self.output_t_prev = T.imatrix()
        self.output_t      = T.imatrix()
        self.output_v      = T.fmatrix()
        self.OH       = OneHot(size_in=self.size_vocab)
        self.output_t_oh   = self.OH(self.output_t)
        self.updater = util.Adam(max_norm=self.max_norm, lr=self.lr)
        self.train = self._make_train()
        self.loss_test = self._make_loss_test()
                               
    def _make_train(self):
        with context.context(training=True):
            output_v_pred, output_t_pred = self.network(self.input, self.output_t_prev, self.output_v)
            cost_T = CrossEntropy(self.output_t_oh, output_t_pred)
            cost_V = self.cost_visual(self.output_v, output_v_pred)
            cost = self.alpha * cost_T + (1.0 - self.alpha) * cost_V
        return theano.function([self.input, self.output_v, self.output_t_prev, self.output_t ],
                               [cost, cost_T, cost_V],
                               updates=self.updates(cost), on_unused_input='warn')
        
    def _make_loss_test(self):
        with context.context(training=False):
            output_v_pred_test, output_t_pred_test = self.network(self.input, self.output_t_prev, self.output_v)
            cost_T_test = CrossEntropy(self.output_t_oh, output_t_pred_test)
            cost_V_test = self.cost_visual(self.output_v, output_v_pred_test)
            cost_test = self.alpha * cost_T_test + (1.0 - self.alpha) * cost_V_test
        return theano.function([self.input, self.output_v, self.output_t_prev, self.output_t ],
                               [cost_test, cost_T_test, cost_V_test],
                               on_unused_input='warn')

    def updates(self, cost):
        return self.updater.get_updates(self.network.params(), cost, disconnected_inputs='warn')
    
    def grow(self, ps):
        self.network.grow(ps)
        self.train  = self._make_train()
        self.loss_test = self._make_loss_test()
        
# Functions added outside the class do not interfere with loading of older versions
def predictor_v(model):
    """Return function to predict image vector from input using `model`."""
    return model.network.predictor_v()

def predictor_r(model):
    """Return function to predict representation from input using `model`."""
    return model.network.predictor_r()
