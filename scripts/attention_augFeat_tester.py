from __future__ import print_function
import time
import theano
import theano.tensor as T
import numpy as np
import sys
import os
import cPickle
from rnn_enc.model.attention_augFeat import ALSTM_augFeat_model
from rnn_enc.core.runner import Tester_augFeat
from rnn_enc.utils.tools import LoadParamsToModel, ArgParser
from rnn_enc.utils.cqa import PrepareData, LoadData, get_minibatches_idx, SaveData, ExtAugFeat

def test_model(testData,
               tag2rankData,
               loadfrom,
               outFile=None,
               n_word=200000,
               n_emb=300,
               n_lstm=128,
               n_att=64,
               n_hidden=256,
               dr_val=0.4,
               l2Decay=0,
               lr_val=0.01,
               batchSize=50,
               testBatchSize=200,
               validFreq=10,
               ifSwap=False):

    # load data here
    print('loadinig data...')
    test, test_valid, _ = LoadData(path=testData, n_words=1000000, valid_portion=0.0, maxlen=None, sort_by_len=True)
    test = [test[0]+test_valid[0], test[1]+test_valid[1], test[2]+test_valid[2], test[3]+test_valid[3]]
    yDim = np.max(test[2]) + 1
    print('Test Size: {0},\t yDim = {1}'.format(len(test[0]),yDim))

    if ifSwap:
        print('swapping the order of train/test pairs')
        test = list(test)
        test[0], test[1] = test[1], test[0]

    tag2rank = cPickle.load(open(tag2rankData, 'rb'))
    test_augFeat = ExtAugFeat(test, tag2rank)

    # init model
    print('building model...')
    x1 = T.matrix(name='x1', dtype='int64')
    m1 = T.matrix(name='m1', dtype=theano.config.floatX)
    x2 = T.matrix(name='x2', dtype='int64')
    m2 = T.matrix(name='m2', dtype=theano.config.floatX)
    lr = T.scalar(name='lr')
    y = T.ivector('y')
    f = T.matrix(name='f', dtype=test_augFeat.dtype)

    ifDropout = theano.shared(np.array(1, dtype=theano.config.floatX), name='ifDr')
    classifier = SeqEncMlpWithAttentionModel(n_word=n_word,
                                             n_in=n_emb,
                                             n_lstm=n_lstm,
                                             n_attHidden=n_att,
                                             n_hidden=n_hidden,
                                             n_out=yDim,
                                             augFeatDim=test_augFeat.shape[1],
                                             ifDropout=ifDropout,
                                             drVal=dr_val,
                                             l2Decay=l2Decay,
                                             embVal=None,
                                             ifFixEmb=False)

    inputList = [x1, m1, x2, m2]
    output = y
    augInput = f

    # test model
    LoadParamsToModel(classifier, loadfrom)
    testP = Tester_augFeat(test, test_augFeat, classifier, inputList, output, augInput, testBatchSize, ifDropout)
    if not outFile:
        outFile = loadfrom + '.pred'
    SaveData(test, testP, outFile)

if __name__ == "__main__":
    opts =ArgParser(train=False, augFeat=True)
    print(opts)
    test_model(**opts)
