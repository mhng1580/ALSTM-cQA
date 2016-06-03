from __future__ import print_function
import time
import theano
import theano.tensor as T
import numpy as np
import sys
from rnn_enc.model2.layer import SoftmaxLayer, HiddenLayer, LstmLayer, WordEmbLayer, DropoutLayer, Attention
from rnn_enc.model2.optimizer import SGD, AdaDelta, AdaGrad
from rnn_enc.model2.runner import Runner, Tester
from rnn_enc.utils.tools import MergeParams, CopyParamsNumpyValue, LoadParamsToModel, PrintParamsValue, L2
from rnn_enc.utils.cqa import PrepareData, LoadData, get_minibatches_idx, SaveData

class SeqEncMlpWithAttentionModel(object):
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

def RunEncMlpModel(n_word=200000,
                   n_emb=300,
                   n_lstm=128,
                   n_att=64,
                   n_hidden=256,
                   max_epoch=8,
                   dr_val=0.4,
                   l2Decay=0,
                   lr_val=0.01,
                   optimizer=AdaGrad,
                   saveto='rnn_enc/result/model/attention.npz',
                   batchSize=50,
                   testBatchSize=200,
                   validFreq=10,
                   dataset='',
                   loadfrom=''):

    # load data here
    print('loadinig data...')
    train, valid, emb = LoadData(path=sys.argv[1], n_words=6319, valid_portion=0.05, maxlen=None)
    test, _, _ = LoadData(path=sys.argv[2], n_words=1000000, valid_portion=0.0, maxlen=None, sort_by_len=True)
    print(train[0][:5],train[1][:5],train[2][:5],)
    yDim = np.max(train[2]) + 1
    train[0], train[1] = train[1], train[0]
    valid[0], valid[1] = valid[1], valid[0]
    test[0], test[1] = test[1], test[0]
    print('Train Size: {0},\tValidation Size: {1},\tTest Size: {2},\t yDim = {3}'.format(len(train[0]),
                                                                                         len(valid[0]),
                                                                                         len(test[0]),
                                                                                         yDim))

    # init model
    print('building model...')
    x1 = T.matrix(name='x1', dtype='int64')
    m1 = T.matrix(name='m1', dtype=theano.config.floatX)
    x2 = T.matrix(name='x2', dtype='int64')
    m2 = T.matrix(name='m2', dtype=theano.config.floatX)
    lr = T.scalar(name='lr')
    y = T.ivector('y')

    ifDropout = theano.shared(np.array(1, dtype=theano.config.floatX), name='ifDr')
    classifier = SeqEncMlpWithAttentionModel(n_word=n_word,
                                             n_in=n_emb,
                                             n_lstm=n_lstm,
                                             n_attHidden=n_att,
                                             n_hidden=n_hidden,
                                             n_out=yDim,
                                             ifDropout=ifDropout,
                                             drVal=dr_val,
                                             l2Decay=l2Decay,
                                             embVal=emb,
                                             ifFixEmb=False)

    inputList = [x1, m1, x2, m2]
    output = y

    # load pretrained parameters
    if bool(loadfrom):
        LoadParamsToModel(classifier, loadfrom)

    saveto = Runner(train, valid, test, classifier, inputList, output, optimizer, lr, max_epoch, lr_val, batchSize, validFreq, saveto, ifDropout)

    # test model
    LoadParamsToModel(classifier, saveto)
    trainP = Tester(train, classifier, inputList, output, testBatchSize, ifDropout)
    testP = Tester(test, classifier, inputList, output, testBatchSize, ifDropout)
    output = saveto+'.pred'
    SaveData(test, testP, output)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        RunEncMlpModel(saveto=sys.argv[3])
    else:
        RunEncMlpModel()
