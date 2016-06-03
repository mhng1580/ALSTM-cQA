import time
import sys
import cPickle
import numpy as np
import theano
import theano.tensor as T
from rnn_enc.model.attention import ALSTM_model
from rnn_enc.utils.tools import LoadParamsToModel
from rnn_enc.utils.cqa import PrepareData, LoadData, get_minibatches_idx, ExtAugFeat

def DumpSentAndAttWeight(testFile, modelFile, dictFile=None, batchSize=200, ifSwap=False):
    tmpTest, _, _ = LoadData(testFile, n_words=1000000, valid_portion=0., maxlen=None)
    test = (tmpTest[1], tmpTest[0], tmpTest[2], tmpTest[3]) if ifSwap else tmpTest
    model = np.load(modelFile)

    hypParams = model['hypParams'].item()
    if 'ifDropout' in hypParams:
        hypParams.pop('ifDropout')
    if 'embVal' in hypParams:
        hypParams.pop('embVal')

    # init model
    print('building model...')
    x1 = T.matrix(name='x1', dtype='int64')
    m1 = T.matrix(name='m1', dtype=theano.config.floatX)
    x2 = T.matrix(name='x2', dtype='int64')
    m2 = T.matrix(name='m2', dtype=theano.config.floatX)
    lr = T.scalar(name='lr')
    y = T.ivector('y')

    ifDropout = theano.shared(np.array(1, dtype=theano.config.floatX), name='ifDr')
    classifier = ALSTM_model(ifDropout=ifDropout, **hypParams)

    inputList = [x1, m1, x2, m2]
    output = y

    # load pretrained parameters
    LoadParamsToModel(classifier, modelFile)
 
    # building theano functions
    print('building theano functions...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    errors = classifier.get_errors(*(inputList + [output]))
    f_test = theano.function(inputs=inputList + [output],
                             outputs=errors,
                             givens=givens_test,
                             name='testModel')
    weights = classifier.get_attention_weight(*inputList)
    f_weight = theano.function(inputs=inputList,
                               outputs=weights,
                               givens=givens_test,
                               name='f_weight')

    # start testing and calc weights
    print('start testing...')
    kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)
    errs = np.concatenate([f_test(*PrepareData([test[0][t] for t in idx],
                                               [test[1][t] for t in idx],
                                               [test[2][t] for t in idx])) for _, idx in kf_test])
    print('Error: {0}'.format(np.mean(errs)))
    
    weights = [f_weight(*PrepareData([test[0][t] for t in idx],
                                     [test[1][t] for t in idx],
                                     [test[2][t] for t in idx])[:-1]) for _, idx in kf_test]
    weights = [weights[j][:,i] for j in xrange(len(weights)) for i in xrange(weights[j].shape[1])]
    weights = [weights[i][:len(test[0][i])] for i in xrange(len(test[0]))]
    
    # generate test data in text and with weights
    newTest = [None, None, None, None, None]
    if dictFile:
        i2w = cPickle.load(open(dictFile, 'rb'))[1]
        newTest[0] = [[i2w[test[0][i][j]] for j in xrange(len(test[0][i]))] for i in xrange(len(test[0]))]
        newTest[1] = [[i2w[test[1][i][j]] for j in xrange(len(test[1][i]))] for i in xrange(len(test[1]))]
    else:
        newTest[0] = [[test[0][i][j] for j in xrange(len(test[0][i]))] for i in xrange(len(test[0]))]
        newTest[1] = [[test[1][i][j] for j in xrange(len(test[1][i]))] for i in xrange(len(test[1]))]

    newTest[2] = test[2]
    newTest[3] = test[3]
    newTest[4] = weights

    return newTest

if __name__ == '__main__':
    pass
#     testFile = '/data/sls/scratch/yzhang87/NLP/rnn_enc/data2/cqa_taskA_dev.p.unified.0'
#     dictFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/dict.pkl'
# 
#     modelFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attention_taskA_unified_dr0.4_noreg_ifFixEmbFalse_drVal0.4_embValTrue_n_in300_n_word6319_l2Decay0_n_attHidden64_n_lstm128_ifRegEmbTrue_n_hidden256_n_out3_AdaGrad_AUXHVA.npz'
#     testWithWeights = DumpSentAndAttWeight(testFile, modelFile, dictFile)
#     cPickle.dump(testWithWeights, open('/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attWeight/taskA_relQrelC', 'wb'))
#     
#     modelFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attention_debug_ifFixEmbFalse_drVal0_embValTrue_n_in300_n_word6319_l2Decay0_n_attHidden64_n_lstm128_ifRegEmbTrue_n_hidden256_n_out3_AdaGrad_ZEUDYL.npz'
#     testWithWeights = DumpSentAndAttWeight(testFile, modelFile, dictFile, ifSwap=True)
#     cPickle.dump(testWithWeights, open('/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attWeight/taskA_relCrelQ', 'wb'))

    # testFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDev.pkl'
    # dictFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDict_2.pkl'
    # modelFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/model/arabic_attention100_AdaGrad_BCUMYQ.npz'
    # testWithWeights = DumpSentAndAttWeight(testFile, modelFile, dictFile)
    # dumpDir = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attWeight/matlab/arab2/'
    # cPickle.dump(testWithWeights, open(dumpDir + 'arabDev_weight.pkl', 'wb'))
    # with open(dumpDir + 'sentence.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([w.encode('utf-8') for w in toks]) for toks in testWithWeights[0]]))
    # with open(dumpDir + 'sentence2.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([w.encode('utf-8') for w in toks]) for toks in testWithWeights[1]]))
    # with open(dumpDir + 'weight.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([str(i) for i in toks]) for toks in testWithWeights[4]]))

    # testFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDev.pkl'
    # dictFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDict_2.pkl'
    # modelFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/model/arabic_swap_attention_AdaGrad_BSWIKZ.npz'
    # testWithWeights = DumpSentAndAttWeight(testFile, modelFile, dictFile, ifSwap=True)
    # dumpDir = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/result/attWeight/matlab/arab2_swap/'
    # cPickle.dump(testWithWeights, open(dumpDir + 'arabDev_swap_weight.pkl', 'wb'))
    # with open(dumpDir + 'sentence.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([w.encode('utf-8') for w in toks]) for toks in testWithWeights[0]]))
    # with open(dumpDir + 'sentence2.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([w.encode('utf-8') for w in toks]) for toks in testWithWeights[1]]))
    # with open(dumpDir + 'weight.txt', 'w') as f:
    #     f.write('\n'.join([' '.join([str(i) for i in toks]) for toks in testWithWeights[4]]))
