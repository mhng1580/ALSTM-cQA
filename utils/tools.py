import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

# randSeed = 777
# rng = np.random.RandomState(randSeed)
rng = np.random.RandomState()

def GenWeight(n_in, n_out):
    W_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    return W_values

def GenBias(n_out):
    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    return b_values

def L1(decay, params):
    return decay * T.sum([T.sum(T.abs_(param)) for param in params.values()])

def L2(decay, params):
    return decay * T.sum([T.sum(param ** 2) for param in params.values()])

def _p(pp, name):
    return "{0}_{1}".format(pp, name)

def np_floatX(val):
    return np.asarray(val, dtype=theano.config.floatX)

def MergeParams(paramsList):
    allParams = OrderedDict()
    for params in paramsList:
        for k, p in params.items():
            if k in allParams:
                raise NameError('Duplicated parameter name: {0}'.format(k))
            else:
                allParams[k] = p
    return allParams

def CopyParamsNumpyValue(params):
    paramsNumpyValue = OrderedDict()
    for k, p in params.items():
        paramsNumpyValue[k] = p.get_value()
    return paramsNumpyValue

def LoadParamsToModel(model, ptParamsFile):
    ptParams = np.load(ptParamsFile)
    params = model.get_all_params()
    for param in params.values():
        if not param.name in ptParams:
            raise NameError('{0} is not in pretrained parameter sets'.format(param.name))
        else:
            print('setting parameter: {0}'.format(param.name))
            param.set_value(ptParams[param.name])

def PrintParamsValue(params):
    for param in params.iteritems():
        print('Name: {0}\t Var: {1}\n {2}'.format(param[0], param[1].name, param[1].get_value()))

def GenDumpFilename(saveto, hypParams=None, optType=None):
    # import os
    # import string
    # import random
    # basename, ext = os.path.splitext(saveto)
    # filename = basename + '_'
    # if not (hypParams is None):
    #     for k, v in hypParams.iteritems():
    #         filename = filename + str(k) + str(v) + '_'
    # if not (optType is None):
    #     filename = filename + optType + '_'
    # filename = filename + ''.join(random.choice(string.uppercase) for _ in range(6)) + ext
    # return filename
    return saveto

def ArgParser(train=True, augFeat=False, augFeat2=False, sharpen=False):
    import argparse
    parser = argparse.ArgumentParser(description='Run rnn encoder model')
    if train:
        parser.add_argument('trainData', type=str, help='training dataset')
        parser.add_argument('testData', type=str, help='testing dataset')
        parser.add_argument('--saveto', type=str)
        parser.add_argument('--loadfrom', type=str)
        parser.add_argument('--max_epoch', type=int)
    else:
        parser.add_argument('loadfrom', type=str, help='RNN model')
        parser.add_argument('testData', type=str, help='testing dataset')
        parser.add_argument('--outFile', type=str, help='result dump path, default is <loadfrom>.pred')
    if augFeat:
        parser.add_argument('tag2rankData', type=str, help='pkl file mapping tag to rank')
    if augFeat2:
        parser.add_argument('tag2featData', type=str)
    if sharpen:
        parser.add_argument('--sharpen_deg', type=int)
    parser.add_argument('--swap', dest='ifSwap', action='store_true', help='swap the order of two sentences')
    parser.add_argument('--no-swap', dest='ifSwap', action='store_false', help='not swap the order of two sentences')
    parser.set_defaults(ifSwap=False)
    
    opts = vars(parser.parse_args())
    return {k:opts[k] for k in opts if not opts[k] is None}
