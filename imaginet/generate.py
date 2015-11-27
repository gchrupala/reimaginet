import theano.tensor as T
from funktional.util import softmax3d
from funktional.layer import last

def make_predict_next(net): 
    out_prev = T.imatrix()
    rep_prev = T.fmatrix()
    rep = net.LM(rep_prev, net.Embed(out_prev))
    out = softmax3d(net.Embed.unembed(net.ToTxt(rep)))
    return theano.function([rep_prev, out_prev], [last(rep), out])

def make_generate(M, temp=1.0):
    # Fixme: convert to make_generate
    predict_r = predictor_r(M['model'])
    predict_next = make_predict_next(M['model'].network)
    END = M['batcher'].mapper.END_ID 
    BEG = M['batcher'].mapper.BEG_ID
    def generate(sent):
        inp = batch_sents(M['batcher'], [sent])
        rep_0 = predict_r(inp)
        out_seq = [END] # SENTENCES ARE REVERSED!
        rep_seq = [rep_0]
        while len(out_seq)<15 and out_seq[-1] != BEG:
            rep, outp = predict_next(rep_seq[-1], numpy.array([[out_seq[-1]]], dtype='int32'))
            probs = temperature(numpy.asarray(outp[0,0,:], dtype='float64'), temp=temp)
            probs /= probs.sum()
            out = numpy.random.choice(len(probs), p=probs)
            out_seq.append(out)
            rep_seq.append(rep)
        return " ".join(reversed(list(M['batcher'].mapper.inverse_transform([out_seq]))[0]))
    return generate
        
