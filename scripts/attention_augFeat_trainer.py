from __future__ import print_function
import time
import theano
import theano.tensor as T
import numpy as np
import sys
import cPickle
from rnn_enc.model.attention_augFeat import ALSTM_augFeat_model
from rnn_enc.core.optimizer import SGD, AdaDelta, AdaGrad
from rnn_enc.core.runner import Runner_augFeat
from rnn_enc.utils.tools import LoadParamsToModel, ArgParser
from rnn_enc.utils.cqa import PrepareData, LoadData, get_minibatches_idx, SaveData, ExtAugFeat

def train_model(trainData,
                testData,
                tag2rankData,
                n_word=200000,
                n_emb=300,
                n_lstm=128,
                n_att=64,
                n_hidden=256,
                max_epoch=15,
                dr_val=0.4,
                l2Decay=0,
                lr_val=0.01,
                optimizer=AdaGrad,
                saveto='attention_augFeat.npz',
                batchSize=50,
                testBatchSize=200,
                validFreq=10,
                loadfrom='',
                ifSwap=False):

    # load data here
    print('loadinig data...')
    train, valid, emb = LoadData(path=trainData, n_words=6319, valid_portion=0.05, maxlen=500)
    
    test, _, _ = LoadData(path=testData, n_words=1000000, valid_portion=0.0, maxlen=None, sort_by_len=True)
    yDim = np.max(train[2]) + 1
    if ifSwap:
        print('swapping the order of train/test pairs')
        train = list(train)
        valid = list(valid)
        test = list(test)
        train[0], train[1] = train[1], train[0]
        valid[0], valid[1] = valid[1], valid[0]
        test[0], test[1] = test[1], test[0]

    tag2rank = cPickle.load(open(tag2rankData, 'rb'))
    train_augFeat = ExtAugFeat(train, tag2rank)
    valid_augFeat = ExtAugFeat(valid, tag2rank)
    test_augFeat = ExtAugFeat(test, tag2rank)
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
    f = T.matrix(name='f', dtype=train_augFeat.dtype)

    ifDropout = theano.shared(np.array(1, dtype=theano.config.floatX), name='ifDr')
    classifier = ALSTM_augFeat_model(n_word=n_word,
                                     n_in=n_emb,
                                     n_lstm=n_lstm,
                                     n_attHidden=n_att,
                                     n_hidden=n_hidden,
                                     n_out=yDim,
                                     augFeatDim=train_augFeat.shape[1],
                                     ifDropout=ifDropout,
                                     drVal=dr_val,
                                     l2Decay=l2Decay,
                                     embVal=emb,
                                     ifFixEmb=False)

    inputList = [x1, m1, x2, m2]
    output = y
    augInput = f

    # load pretrained parameters
    if bool(loadfrom):
        LoadParamsToModel(classifier, loadfrom)

    saveto = Runner_augFeat(train, train_augFeat, valid, valid_augFeat, test, test_augFeat, classifier, inputList, output, augInput, optimizer, lr, max_epoch, lr_val, batchSize, validFreq, saveto, ifDropout)

if __name__ == "__main__":
    opts = ArgParser(augFeat=True)
    print(opts)
    train_model(**opts)
