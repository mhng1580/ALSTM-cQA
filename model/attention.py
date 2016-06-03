from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from rnn_enc.core.layer import SoftmaxLayer, HiddenLayer, LstmLayer, WordEmbLayer, DropoutLayer, Attention
from rnn_enc.utils.tools import MergeParams, L2

class ALSTM_model(object):
    def __init__(self, n_word, n_in, n_lstm, n_attHidden, n_hidden, n_out, ifDropout, drVal=0.2, l2Decay=1e-3, embVal=None, ifFixEmb=False, ifRegEmb=True):

        if not (embVal is None):
            print('embVal is not None, setting n_in {0} -> {1}'.format(n_in, embVal.shape[1]))
            n_in = embVal.shape[1]

        self.hyperParams = locals()
        self.hyperParams.pop('self', None)
        self.hyperParams.pop('ifDropout', None)
        self.hyperParams['embVal'] = True if not (embVal is None) else False

        self.emb = WordEmbLayer(n_word, n_in, prefix='wEmb')
        if not (embVal is None):
            self.emb.set_embedding(embVal)

        self.enc1 = LstmLayer(n_in, n_lstm, prefix='enc1')
        self.enc2 = LstmLayer(n_in, n_lstm, prefix='enc2')
        self.att = Attention(n_lstm, n_attHidden)
        self.h = HiddenLayer(n_lstm * 2, n_hidden, prefix='h')
        self.s = SoftmaxLayer(n_hidden, n_out, prefix='s')
        self.dp = DropoutLayer(drVal, prefix='dp')
        self.ifDropout = ifDropout
        
        if ifFixEmb:
            self.params = MergeParams([self.enc1.get_params(), 
                                       self.enc2.get_params(),
                                       self.att.get_params(),
                                       self.h.get_params(),
                                       self.s.get_params()])
        else:
            self.params = MergeParams([self.emb.get_params(),
                                       self.enc1.get_params(), 
                                       self.enc2.get_params(),
                                       self.att.get_params(),
                                       self.h.get_params(),
                                       self.s.get_params()])

        if ifRegEmb:
            self.regParams = MergeParams([self.emb.get_params(),
                                          self.enc1.get_params(), 
                                          self.enc2.get_params(),
                                          self.att.get_params(),
                                          self.h.get_params(),
                                          self.s.get_params()])
        else:
            self.regParams = MergeParams([self.enc1.get_params(), 
                                          self.enc2.get_params(),
                                          self.att.get_params(),
                                          self.h.get_params(),
                                          self.s.get_params()])
        self.L2 = L2(l2Decay, self.regParams)


        self.allParams = MergeParams([self.emb.get_params(),
                                      self.enc1.get_params(), 
                                      self.enc2.get_params(),
                                      self.att.get_params(),
                                      self.h.get_params(),
                                      self.s.get_params(),
                                      self.dp.get_params()])

    def get_hOutput(self, input1, inputMask1, input2, inputMask2):
        inputEmb1 = self.dp.get_output(self.emb.get_output(input1), self.ifDropout)
        inputEmb2 = self.dp.get_output(self.emb.get_output(input2), self.ifDropout)

        encOutputs1, encCells1 = self.enc1.get_output(inputEmb1, inputMask1)
        sentVec1 = self.dp.get_output(encOutputs1[-1,:,:], self.ifDropout)
        sentCell1 = encCells1[-1,:,:]
        
        sentVec2 = self.dp.get_output(self.enc2.get_output(inputEmb2, inputMask2, sentVec1, sentCell1)[0][-1,:,:],
                                      self.ifDropout)
        mlpInput = self.att.get_output(encOutputs1, inputMask1, sentVec2)
        hOutput = self.h.get_output(mlpInput)
        return hOutput

    def get_output(self, input1, inputMask1, input2, inputMask2):
        hOutput = self.get_hOutput(input1, inputMask1, input2, inputMask2)
        return self.s.get_output(hOutput)

    def get_cost(self, input1, inputMask1, input2, inputMask2, y):
        hOutput = self.get_hOutput(input1, inputMask1, input2, inputMask2)
        return T.mean(self.s.negative_log_likelihood(hOutput, y)) + self.L2
    
    def get_loss(self, input1, inputMask1, input2, inputMask2, y):
        hOutput = self.get_hOutput(input1, inputMask1, input2, inputMask2)
        return self.s.negative_log_likelihood(hOutput, y)

    def get_errors(self, input1, inputMask1, input2, inputMask2, y):
        hOutput = self.get_hOutput(input1, inputMask1, input2, inputMask2)
        return self.s.errors(hOutput, y)

    def get_attention_weight(self, input1, inputMask1, input2, inputMask2):
        inputEmb1 = self.dp.get_output(self.emb.get_output(input1), self.ifDropout)
        inputEmb2 = self.dp.get_output(self.emb.get_output(input2), self.ifDropout)

        encOutputs1, encCells1 = self.enc1.get_output(inputEmb1, inputMask1)
        sentVec1 = self.dp.get_output(encOutputs1[-1,:,:], self.ifDropout)
        sentCell1 = encCells1[-1,:,:]
        
        sentVec2 = self.dp.get_output(self.enc2.get_output(inputEmb2, inputMask2, sentVec1, sentCell1)[0][-1,:,:],
                                      self.ifDropout)
        attWeight = self.att.get_weight(encOutputs1, inputMask1, sentVec2)
        return attWeight

    def get_params(self):
        return self.params

    def get_all_params(self):
        return self.allParams
