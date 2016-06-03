from __future__ import print_function
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from rnn_enc.utils.tools import GenWeight, GenBias, _p, np_floatX, MergeParams

trng = MRG_RandomStreams(777)

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, prefix='sm'):
        '''
        Softmax layer

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: dimensionality of output

        :type prefix: str
        :param prefix: prefix for symbolic variable names
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.prefix = prefix

        self.params = OrderedDict()
        self.regParams = OrderedDict()
        
        self.W = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W'))
        self.b = theano.shared(GenBias(n_out), name=_p(prefix, 'b'))

        self.params[_p(prefix, 'W')] = self.W
        self.params[_p(prefix, 'b')] = self.b

        self.regParams[_p(prefix, 'W')] = self.W

    def get_output(self, input):
        return T.nnet.softmax(T.dot(input, self.W) + self.b)

    def negative_log_likelihood(self, input, y):
        # return -T.mean(T.log(self.get_output(input))[T.arange(y.shape[0]), y])
        return -T.log(self.get_output(input))[T.arange(y.shape[0]), y]

    def errors(self, input, y):
        y_pred = T.argmax(self.get_output(input), axis=1)
        if y.ndim != y_pred.ndim:
            raise TypeError('dimension inconsistent')
        return T.neq(y_pred, y)

    def get_regParams(self):
        return self.regParams
    
    def get_params(self):
        return self.params

class HiddenLayer(object):
    def __init__(self, n_in, n_out, prefix='h', activation=T.tanh):
        '''
        Hidden layer

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type prefix: str
        :param prefix: prefix for symbolic variable names
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.prefix = prefix
        self.activation = activation

        self.params = OrderedDict()
        self.regParams = OrderedDict()

        self.W = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W'))
        self.b = theano.shared(GenBias(n_out), name=_p(prefix, 'b'))

        self.params[_p(prefix, 'W')] = self.W
        self.params[_p(prefix, 'b')] = self.b

        self.regParams[_p(prefix, 'W')] = self.W

    def get_output(self, input):
        lin_output = T.dot(input, self.W) + self.b
        output = lin_output if self.activation is None else self.activation(lin_output)
        return output

    def get_params(self):
        return self.params

    def get_regParams(self):
        return self.regParams

class DropoutLayer(object):
    def __init__(self, dr_val, prefix='dp'):
        '''
        Dropout layer

        :type ifDropout: theano.tensor.scalar
        :param ifDropout:

        :type dr_val: float
        :param dr_val: dropout rate
        '''
        self.params = OrderedDict()
        
        self.dr = theano.shared(np_floatX(dr_val), name=_p(prefix, 'dr'))
        self.params[_p(prefix, 'dr')] = self.dr

    def get_output(self, input, ifDropout):
        output = T.switch(ifDropout, 
                          input * trng.binomial(input.shape, p=(1. - self.dr), n=1, dtype=input.dtype),
                          input * (1. - self.dr))
        return output

    def get_params(self):
        return self.params

class WordEmbLayer(object):
    def __init__(self, n_word, embDim, prefix='wEmb'):
        self.params = OrderedDict()
        
        self.W = theano.shared(GenWeight(n_word, embDim), name=_p(prefix, 'W'))
        self.params[_p(prefix, 'W')] = self.W
    
    def set_embedding(self, emb):
        if not emb.shape[1] == self.W.get_value().shape[1]:
            raise ValueError('Embedding dimension does not match: {0} != {1}'.format(emb.shape[1], self.W.get_value().shape[1]))
        self.W.set_value(emb)
     
    def get_output(self, input):
        return self.W[input]

    def get_params(self):
        return self.params

    def get_regParams(self):
        return self.params

class LstmLayer(object):
    def __init__(self, n_in, n_out, prefix='lstm', activation=T.tanh):
        '''
        Long short-term memory layer

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of lstm units

        :type prefix: str
        :param prefix: prefix for symbolic variable names
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.prefix = prefix

        self.params = OrderedDict()
        self.regParams = OrderedDict()

        self.W_xi = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W_xi'))
        self.params[_p(prefix, 'W_xi')] = self.W_xi
        self.W_xo = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W_xo'))
        self.params[_p(prefix, 'W_xo')] = self.W_xo
        self.W_xf = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W_xf'))
        self.params[_p(prefix, 'W_xf')] = self.W_xf
        self.W_xc = theano.shared(GenWeight(n_in, n_out), name=_p(prefix, 'W_xc'))
        self.params[_p(prefix, 'W_xc')] = self.W_xc

        self.W_hi = theano.shared(GenWeight(n_out, n_out), name=_p(prefix, 'W_hi'))
        self.params[_p(prefix, 'W_hi')] = self.W_hi
        self.W_ho = theano.shared(GenWeight(n_out, n_out), name=_p(prefix, 'W_ho'))
        self.params[_p(prefix, 'W_ho')] = self.W_ho
        self.W_hf = theano.shared(GenWeight(n_out, n_out), name=_p(prefix, 'W_hf'))
        self.params[_p(prefix, 'W_hf')] = self.W_hf
        self.W_hc = theano.shared(GenWeight(n_out, n_out), name=_p(prefix, 'W_hc'))
        self.params[_p(prefix, 'W_hc')] = self.W_hc

        self.b_i = theano.shared(GenBias(n_out), name=_p(prefix, 'b_i'))
        self.params[_p(prefix, 'b_i')] = self.b_i
        self.b_o = theano.shared(GenBias(n_out), name=_p(prefix, 'b_o'))
        self.params[_p(prefix, 'b_o')] = self.b_o
        self.b_f = theano.shared(GenBias(n_out), name=_p(prefix, 'b_i'))
        self.params[_p(prefix, 'b_f')] = self.b_f
        self.b_c = theano.shared(GenBias(n_out), name=_p(prefix, 'b_c'))
        self.params[_p(prefix, 'b_c')] = self.b_c

        self.regParams[_p(prefix, 'W_xi')] = self.W_xi
        self.regParams[_p(prefix, 'W_xo')] = self.W_xo
        self.regParams[_p(prefix, 'W_xf')] = self.W_xf
        self.regParams[_p(prefix, 'W_xc')] = self.W_xc
        self.regParams[_p(prefix, 'W_hi')] = self.W_hi
        self.regParams[_p(prefix, 'W_ho')] = self.W_ho
        self.regParams[_p(prefix, 'W_hf')] = self.W_hf
        self.regParams[_p(prefix, 'W_hc')] = self.W_hc

    def get_output(self, input, inputMask, h0=None, c0=None):
        '''
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_steps, n_samples, n_in)

        :type inputMask: theano.tensor.dmatrix
        :param inputMask: a symbolic tensor of shape (n_steps, n_samples)

        :type h0: theano.tensor.dmatrix
        :param h0: 

        :type c0: theano.tensor.dmatrix
        :param c0:

        '''
        n_steps = input.shape[0]
        n_samples = input.shape[1]

        inputW_xib_i = T.dot(input, self.W_xi) + self.b_i
        inputW_xob_o = T.dot(input, self.W_xo) + self.b_o
        inputW_xfb_f = T.dot(input, self.W_xf) + self.b_f
        inputW_xcb_c = T.dot(input, self.W_xc) + self.b_c

        if h0 == None:
            h0 = T.alloc(np_floatX(0.), n_samples, self.n_out)

        if c0 == None:
            c0 = T.alloc(np_floatX(0.), n_samples, self.n_out)

        def Recurrence(m, xW_xib_i, xW_xob_o, xW_xfb_f, xW_xcb_c, prev_h, prev_c):
            i = T.nnet.sigmoid(xW_xib_i + T.dot(prev_h, self.W_hi))
            o = T.nnet.sigmoid(xW_xob_o + T.dot(prev_h, self.W_ho))
            f = T.nnet.sigmoid(xW_xfb_f + T.dot(prev_h, self.W_hf))

            c = f * prev_c + i * T.tanh(xW_xcb_c + T.dot(prev_h, self.W_hc))
            c_masked = m[:, None] * c + (1. - m)[:, None] * prev_c

            h = o * T.tanh(c_masked)
            h_masked = m[:, None] * h + (1. - m)[:, None] * prev_h
            
            return h_masked, c_masked

        rval, _ = theano.scan(Recurrence,
                              sequences=[inputMask,
                                         inputW_xib_i,
                                         inputW_xob_o,
                                         inputW_xfb_f,
                                         inputW_xcb_c],
                              outputs_info=[h0,
                                            c0],
                              name=_p(self.prefix, 'apply'),
                              n_steps=n_steps)

        output = rval[0]
        cell = rval[1]
        return output, cell

    def get_params(self):
        return self.params

    def get_regParams(self):
        return self.regParams

class Attention(object):
    def __init__(self, n_lstm, n_hidden, prefix='at', sharpen=None):
        '''
        Attention model

        :type encOutputs1: theano.tensor.matrix
        :param encOutputs1: a symbolic tensor of shape (n_steps1, n_samples, n_features)

        :type encMask1: theano.tensor.matrix
        :param encMask1: a symbolic tensor of shape (n_steps1, n_samples)

        :type encOutput2: theano.tensor.matrix
        :param encOutput2: a symbolic tensor of shape (n_samples, n_features)

        :type n_hidden: int
        :param n_hidden: number of hidden units
        '''
        self.h1 = HiddenLayer(n_lstm * 2, n_hidden, prefix=_p(prefix, 'h1'))
        self.h2 = HiddenLayer(n_hidden, 1, prefix=_p(prefix, 'h2'))

        self.params = MergeParams([self.h1.get_params(),
                                   self.h2.get_params()])
        self.regParams = MergeParams([self.h1.get_regParams(),
                                      self.h2.get_regParams()])
        self.sharpen = sharpen

    def get_output(self, encOutputs1, encMask1, encOutput2):
        # e = T.alloc(1, encOutputs1.shape[0])
        # tiledEncOutput2 = T.outer(e, encOutput2.flatten()).reshape([e.shape[0], encOutput2.shape[0], encOutput2.shape[1]])
        # attInput = T.concatenate([encOutputs1, tiledEncOutput2], axis=2)

        # A = self.h2.get_output(self.h1.get_output(attInput))[:,:,0]
        # maskedExpA = T.exp(A) * encMask1
        # weight = maskedExpA / T.sum(maskedExpA, axis=0)
        weight = self.get_weight(encOutputs1, encMask1, encOutput2)
        weightedEncOutput1 = T.sum(encOutputs1 * weight[:,:,None], axis=0)

        output = T.concatenate([weightedEncOutput1, encOutput2], axis=1)
        return output

    def get_weight(self, encOutputs1, encMask1, encOutput2):
        e = T.alloc(1, encOutputs1.shape[0])
        tiledEncOutput2 = T.outer(e, encOutput2.flatten()).reshape([e.shape[0], encOutput2.shape[0], encOutput2.shape[1]])
        attInput = T.concatenate([encOutputs1, tiledEncOutput2], axis=2)

        A = self.h2.get_output(self.h1.get_output(attInput))[:,:,0]
        maskedExpA = T.exp(A) * encMask1
        weight = maskedExpA / T.sum(maskedExpA, axis=0)
        
        weight = self.sharpen(weight) / T.sum(self.sharpen(weight), axis=0) if not (self.sharpen is None) else weight
        return weight
        

    def get_params(self):
        return self.params

    def get_regParams(self):
        return self.regParams

if __name__ == "__main__":
    layer = SoftmaxLayer(None, 3, 5)

            
