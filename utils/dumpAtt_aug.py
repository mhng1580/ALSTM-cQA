import time
import sys
import cPickle
import numpy as np
import theano
import theano.tensor as T
from rnn_enc.model.attention_augFeat import ALSTM_augFeat_model
from rnn_enc.utils.tools import LoadParamsToModel
from rnn_enc.utils.cqa import PrepareData, LoadData, get_minibatches_idx, ExtAugFeat

def DumpSentAndAttWeight_augFeat(testFile, tag2rankData, modelFile, dictFile=None, batchSize=200, ifSwap=False, isTrain=False):
    tmpTest, tmpValid, _ = LoadData(testFile, n_words=1000000)
    tmpTest = [tmpTest[0]+tmpValid[0], tmpTest[1]+tmpValid[1], tmpTest[2]+tmpValid[2], tmpTest[3]+tmpValid[3]]
    test = (tmpTest[1], tmpTest[0], tmpTest[2], tmpTest[3]) if ifSwap else tmpTest

    tag2rank = cPickle.load(open(tag2rankData,'rb'))
    test_augFeat = ExtAugFeat(test, tag2rank)

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
    f = T.matrix(name='f', dtype=test_augFeat.dtype)

    ifDropout = theano.shared(np.array(1, dtype=theano.config.floatX), name='ifDr')
    classifier = ALSTM_augFeat_model(ifDropout=ifDropout, **hypParams)

    inputList = [x1, m1, x2, m2]
    output = y
    augInput = f

    # load pretrained parameters
    LoadParamsToModel(classifier, modelFile)
 
    # building theano functions
    print('building theano functions...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    errors = classifier.get_errors(*(inputList + [output] + [augInput]))
    f_test = theano.function(inputs=inputList + [output] + [augInput],
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
    errs = np.concatenate([f_test(*(list(PrepareData([test[0][t] for t in idx],
                                                     [test[1][t] for t in idx],
                                                     [test[2][t] for t in idx])) +
                                    [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test])
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

    modelFile = sys.argv[1]
    testFile = sys.argv[2]
    tagFile = sys.argv[3]
    dictFile = sys.argv[4]
    dumpDir = sys.argv[5]
    testWithWeights = DumpSentAndAttWeight_augFeat(testFile, tagFile, modelFile, dictFile, ifSwap=True, isTrain=True)
    with open(dumpDir + 'sentence.txt', 'w') as f:
        for toks in testWithWeights[0]:
            s = ' '.join([w.decode('utf-8') for w in toks])
            f.write(s.encode('utf-8') + '\n')
    with open(dumpDir + 'sentence2.txt', 'w') as f:
        for toks in testWithWeights[1]:
            s = ' '.join([w.decode('utf-8') for w in toks])
            f.write(s.encode('utf-8') + '\n')
    with open(dumpDir + 'weight.txt', 'w') as f:
        f.write('\n'.join([' '.join([str(i) for i in toks]) for toks in testWithWeights[4]]))
    with open(dumpDir + 'tag.txt', 'w') as f:
        f.write('\n'.join(testWithWeights[3]))
