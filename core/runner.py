from __future__ import print_function
import time
import numpy as np
import sys
import theano
import os
import sys
from subprocess import call
from rnn_enc.utils.cqa import get_minibatches_idx, PrepareData
from rnn_enc.utils.tools import CopyParamsNumpyValue, PrintParamsValue, GenDumpFilename

def Runner(train, valid, test, classifier, inputList, output, optimizer, lr, max_epoch, lr_val, batchSize, validFreq, saveto, ifDropout=None):
    
    # building train/test theano function
    print('building theano train/test function...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    # ifDropoutSub = theano.printing.Print('USING ifDropoutSub')(theano.shared(np.array(0, dtype=theano.config.floatX), name='ifDrSub'))
    # givens_test = [(ifDropout, ifDropoutSub)] if not (ifDropout is None) else []
    errors = classifier.get_errors(*(inputList + [output]))
    testModel = theano.function(inputs=inputList + [output],
                                outputs=errors,
                                givens=givens_test,
                                name='testModel')
    
    cost = classifier.get_cost(*(inputList + [output]))
    opt = optimizer(inputList+[output], classifier.get_params(), cost, lr)
    trainModel = opt.trainModel()
    
    loss = classifier.get_loss(*(inputList + [output]))
    f_loss = theano.function(inputs=inputList + [output],
                             outputs=loss,
                             givens=givens_test,
                             name='f_loss')
    # start training
    print('start training...')
    print('Hyper Parameters: {0}'.format(classifier.hyperParams))
    print('Optimizer: {0}'.format(opt.get_optType()))

    epoch = 0
    n_samples = 0
    bestIter = 0
    bestValidError = np.inf
    bestParamsNumpyValue = None
    tStart = time.time()
    histError = []
    epochError = []
    # saveto = GenDumpFilename(saveto, classifier.hyperParams, opt.get_optType())
    saveto = GenDumpFilename(saveto, optType=opt.get_optType())

    try:
        while (epoch < max_epoch):
            epoch = epoch + 1
            batchThisEpoch = 0

            kf = get_minibatches_idx(len(train[0]), batchSize, shuffle=True)
            kf_valid = get_minibatches_idx(len(valid[0]), batchSize, shuffle=False)
            kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)

            sysout = call(["renew", "-R"])
            for _, trainIndex in kf:
                in1 = [train[0][t] for t in trainIndex]
                in2 = [train[1][t] for t in trainIndex]
                iny = [train[2][t] for t in trainIndex]
                batchAvgCost = trainModel(lr_val, *PrepareData(in1, in2, iny))
                n_samples += len(in1)
                batchThisEpoch += 1

                if (batchThisEpoch % validFreq == 0):
                    validLoss = np.mean(np.concatenate([f_loss(*PrepareData([valid[0][t] for t in idx],
                                                                            [valid[1][t] for t in idx],
                                                                            [valid[2][t] for t in idx])) for _, idx in kf_valid]))
                    validError = np.mean(np.concatenate([testModel(*PrepareData([valid[0][t] for t in idx],
                                                                                [valid[1][t] for t in idx],
                                                                                [valid[2][t] for t in idx])) for _, idx in kf_valid]))
                    testError = np.mean(np.concatenate([testModel(*PrepareData([test[0][t] for t in idx],
                                                                               [test[1][t] for t in idx],
                                                                               [test[2][t] for t in idx])) for _, idx in kf_test]))
                    histError.append([n_samples, n_samples * 1. / len(train[0]), batchAvgCost, validLoss, validError, testError])
                    print('[ Epoch {0} of {1} ]: {2} samples seen. TrainBatchLoss: {3}, ValidLoss: {4}, ValidErr: {5}, BestValidErr: {6}, TestErr: {7}'.format(epoch, 
                                                                                                                                                               max_epoch, 
                                                                                                                                                               n_samples, 
                                                                                                                                                               batchAvgCost, 
                                                                                                                                                               validLoss, 
                                                                                                                                                               validError,
                                                                                                                                                               bestValidError,
                                                                                                                                                               testError))
                    if validError < bestValidError:
                        print('\tBEST VAL ERR: {0}'.format(validError))
                        bestParamsNumpyValue = CopyParamsNumpyValue(classifier.get_all_params())
                        bestValidError = validError
                    sys.stdout.flush()

            trainError = np.mean(np.concatenate([testModel(*PrepareData([train[0][t] for t in idx],
                                                                        [train[1][t] for t in idx],
                                                                        [train[2][t] for t in idx])) for _, idx in kf]))
            validError = np.mean(np.concatenate([testModel(*PrepareData([valid[0][t] for t in idx],
                                                                        [valid[1][t] for t in idx],
                                                                        [valid[2][t] for t in idx])) for _, idx in kf_valid]))
            testError = np.mean(np.concatenate([testModel(*PrepareData([test[0][t] for t in idx],
                                                                       [test[1][t] for t in idx],
                                                                       [test[2][t] for t in idx])) for _, idx in kf_test]))
            epochError.append([trainError, validError, testError])
            print('>>>>>> Epoch {0} of {1} : Train: {3}, Validation: {4}, Test: {5}'.format(epoch, max_epoch, n_samples, trainError, validError, testError))

            if bool(saveto):
                np.savez(saveto,
                         hypParams=classifier.hyperParams,
                         optimizer=opt.get_optType(),
                         trainError=trainError,
                         validError=validError,
                         testError=testError,
                         histError=histError,
                         epochError=epochError,
                         **bestParamsNumpyValue)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED!")

    tEnd = time.time()
    trainError = np.mean(np.concatenate([testModel(*PrepareData([train[0][t] for t in idx],
                                                                [train[1][t] for t in idx],
                                                                [train[2][t] for t in idx])) for _, idx in kf]))
    validError = np.mean(np.concatenate([testModel(*PrepareData([valid[0][t] for t in idx],
                                                                [valid[1][t] for t in idx],
                                                                [valid[2][t] for t in idx])) for _, idx in kf_valid]))
    testError = np.mean(np.concatenate([testModel(*PrepareData([test[0][t] for t in idx],
                                                               [test[1][t] for t in idx],
                                                               [test[2][t] for t in idx])) for _, idx in kf_test]))
    print('Train: {0}\tValidation: {1}\tTest: {2}'.format(trainError, validError, testError))
    print('Hyper Parameters: {0}'.format(classifier.hyperParams))
    print('Optimizer: {0}'.format(opt.get_optType()))

    if bool(saveto):
        np.savez(saveto,
                 hypParams=classifier.hyperParams,
                 optimizer=opt.get_optType(),
                 trainError=trainError,
                 validError=validError,
                 testError=testError,
                 histError=histError,
                 epochError=epochError,
                 **bestParamsNumpyValue)
    print('The code run for {0} epochs, with {1} sec/epoch'.format(epoch, (tEnd - tStart) / (1. * epoch)))
    print('Total running time: {0}'.format(tEnd - tStart))
    return saveto

def Tester(test, classifier, inputList, output=None, batchSize=100, ifDropout=None):
    
    # building theano functions
    print('building theano functions...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    f_posteriors = theano.function(inputs=inputList,
                                   outputs=classifier.get_output(*inputList),
                                   givens=givens_test,
                                   name='f_posteriors')
    if not (output is None):
        errors = classifier.get_errors(*(inputList + [output]))
        testModel = theano.function(inputs=inputList + [output],
                                    outputs=errors,
                                    givens=givens_test,
                                    name='testModel')
    
    # start testing
    print('start testing...')
    tStart = time.time()

    
    kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)
    testPosteriors = np.vstack([f_posteriors(*PrepareData([test[0][t] for t in idx],
                                                          [test[1][t] for t in idx],
                                                          [test[2][t] for t in idx])[:-1]) for _, idx in kf_test])
    if not (output is None):
        errs = np.concatenate([testModel(*PrepareData([test[0][t] for t in idx],
                                                      [test[1][t] for t in idx],
                                                      [test[2][t] for t in idx])) for _, idx in kf_test])
        testError = np.mean(errs)
        print('Error: {0}'.format(testError))

    tEnd = time.time()
    print('Total running time: {0}'.format(tEnd - tStart))
    return testPosteriors

def Runner_augFeat(train, train_augFeat, valid, valid_augFeat, test, test_augFeat, classifier, inputList, output, augInput, optimizer, lr, max_epoch, lr_val, batchSize, validFreq, saveto, ifDropout=None):
    
    # building train/test theano function
    print('building theano train/test function...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    # ifDropoutSub = theano.printing.Print('USING ifDropoutSub')(theano.shared(np.array(0, dtype=theano.config.floatX), name='ifDrSub'))
    # givens_test = [(ifDropout, ifDropoutSub)] if not (ifDropout is None) else []
    errors = classifier.get_errors(*(inputList + [output] + [augInput]))
    testModel = theano.function(inputs=inputList + [output] + [augInput],
                                outputs=errors,
                                givens=givens_test,
                                name='testModel')
    
    cost = classifier.get_cost(*(inputList + [output] + [augInput]))
    opt = optimizer(inputList+[output]+[augInput], classifier.get_params(), cost, lr)
    trainModel = opt.trainModel()
    
    loss = classifier.get_loss(*(inputList + [output] + [augInput]))
    f_loss = theano.function(inputs=inputList + [output] + [augInput],
                             outputs=loss,
                             givens=givens_test,
                             name='f_loss')
    # start training
    print('start training...')
    print('Hyper Parameters: {0}'.format(classifier.hyperParams))
    print('Optimizer: {0}'.format(opt.get_optType()))

    epoch = 0
    n_samples = 0
    bestIter = 0
    bestValidError = np.inf
    bestParamsNumpyValue = None
    tStart = time.time()
    histError = []
    epochError = []
    # saveto = GenDumpFilename(saveto, classifier.hyperParams, opt.get_optType())
    saveto = GenDumpFilename(saveto, optType=opt.get_optType())

    try:
        while (epoch < max_epoch):
            epoch = epoch + 1
            batchThisEpoch = 0

            kf = get_minibatches_idx(len(train[0]), batchSize, shuffle=True)
            kf_valid = get_minibatches_idx(len(valid[0]), batchSize, shuffle=False)
            kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)

            sysout = call(["renew", "-R"])
            for _, trainIndex in kf:
                in1 = [train[0][t] for t in trainIndex]
                in2 = [train[1][t] for t in trainIndex]
                iny = [train[2][t] for t in trainIndex]
                inf = np.vstack([train_augFeat[t,:] for t in trainIndex])
                batchAvgCost = trainModel(lr_val, *(list(PrepareData(in1, in2, iny)) + [inf]))
                n_samples += len(in1)
                batchThisEpoch += 1

                if (batchThisEpoch % validFreq == 0):
                    validLoss = np.mean(np.concatenate([f_loss(*(list(PrepareData([valid[0][t] for t in idx],
                                                                                  [valid[1][t] for t in idx],
                                                                                  [valid[2][t] for t in idx])) +
                                                                 [np.vstack([valid_augFeat[t,:] for t in idx])])) for _, idx in kf_valid]))
                    validError = np.mean(np.concatenate([testModel(*(list(PrepareData([valid[0][t] for t in idx],
                                                                                      [valid[1][t] for t in idx],
                                                                                      [valid[2][t] for t in idx])) +
                                                                     [np.vstack([valid_augFeat[t,:] for t in idx])])) for _, idx in kf_valid]))
                    testError = np.mean(np.concatenate([testModel(*(list(PrepareData([test[0][t] for t in idx],
                                                                                     [test[1][t] for t in idx],
                                                                                     [test[2][t] for t in idx])) +
                                                                    [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test]))
                    histError.append([n_samples, n_samples * 1. / len(train[0]), batchAvgCost, validLoss, validError, testError])
                    print('[ Epoch {0} of {1} ]: {2} samples seen. TrainBatchLoss: {3}, ValidLoss: {4}, ValidErr: {5}, BestValidErr: {6}, TestErr: {7}'.format(epoch, 
                                                                                                                                                               max_epoch, 
                                                                                                                                                               n_samples, 
                                                                                                                                                               batchAvgCost, 
                                                                                                                                                               validLoss, 
                                                                                                                                                               validError,
                                                                                                                                                               bestValidError,
                                                                                                                                                               testError))
                    if validError < bestValidError:
                        print('\tBEST VAL ERR: {0}'.format(validError))
                        bestParamsNumpyValue = CopyParamsNumpyValue(classifier.get_all_params())
                        bestValidError = validError
                    sys.stdout.flush()

            trainError = np.mean(np.concatenate([testModel(*(list(PrepareData([train[0][t] for t in idx],
                                                                              [train[1][t] for t in idx],
                                                                              [train[2][t] for t in idx])) +
                                                             [np.vstack([train_augFeat[t,:] for t in idx])])) for _, idx in kf]))
            validError = np.mean(np.concatenate([testModel(*(list(PrepareData([valid[0][t] for t in idx],
                                                                              [valid[1][t] for t in idx],
                                                                              [valid[2][t] for t in idx])) +
                                                             [np.vstack([valid_augFeat[t,:] for t in idx])])) for _, idx in kf_valid]))
            testError = np.mean(np.concatenate([testModel(*(list(PrepareData([test[0][t] for t in idx],
                                                                             [test[1][t] for t in idx],
                                                                             [test[2][t] for t in idx])) +
                                                            [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test]))
            epochError.append([trainError, validError, testError])
            print('>>>>>> Epoch {0} of {1} : Train: {3}, Validation: {4}, Test: {5}'.format(epoch, max_epoch, n_samples, trainError, validError, testError))

            if bool(saveto):
                np.savez(saveto,
                         hypParams=classifier.hyperParams,
                         optimizer=opt.get_optType(),
                         trainError=trainError,
                         validError=validError,
                         testError=testError,
                         histError=histError,
                         epochError=epochError,
                         **bestParamsNumpyValue)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED!")

    tEnd = time.time()
    trainError = np.mean(np.concatenate([testModel(*(list(PrepareData([train[0][t] for t in idx],
                                                                      [train[1][t] for t in idx],
                                                                      [train[2][t] for t in idx])) +
                                                     [np.vstack([train_augFeat[t,:] for t in idx])])) for _, idx in kf]))
    validError = np.mean(np.concatenate([testModel(*(list(PrepareData([valid[0][t] for t in idx],
                                                                      [valid[1][t] for t in idx],
                                                                      [valid[2][t] for t in idx])) +
                                                     [np.vstack([valid_augFeat[t,:] for t in idx])])) for _, idx in kf_valid]))
    testError = np.mean(np.concatenate([testModel(*(list(PrepareData([test[0][t] for t in idx],
                                                                     [test[1][t] for t in idx],
                                                                     [test[2][t] for t in idx])) +
                                                    [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test]))
    print('Train: {0}\tValidation: {1}\tTest: {2}'.format(trainError, validError, testError))
    print('Hyper Parameters: {0}'.format(classifier.hyperParams))
    print('Optimizer: {0}'.format(opt.get_optType()))

    if bool(saveto):
        np.savez(saveto,
                 hypParams=classifier.hyperParams,
                 optimizer=opt.get_optType(),
                 trainError=trainError,
                 validError=validError,
                 testError=testError,
                 histError=histError,
                 epochError=epochError,
                 **bestParamsNumpyValue)
    print('The code run for {0} epochs, with {1} sec/epoch'.format(epoch, (tEnd - tStart) / (1. * epoch)))
    print('Total running time: {0}'.format(tEnd - tStart))
    return saveto

def Tester_augFeat(test, test_augFeat, classifier, inputList, output, augInput, batchSize=100, ifDropout=None):
    
    # building theano functions
    print('building theano functions...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    f_posteriors = theano.function(inputs=inputList + [augInput],
                                   outputs=classifier.get_output(*(inputList + [augInput])),
                                   givens=givens_test,
                                   name='f_posteriors')
    if not (output is None):
        errors = classifier.get_errors(*(inputList + [output] + [augInput]))
        testModel = theano.function(inputs=inputList + [output] + [augInput],
                                    outputs=errors,
                                    givens=givens_test,
                                    name='testModel')
    
    # start testing
    print('start testing...')
    tStart = time.time()

    
    kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)
    testPosteriors = np.vstack([f_posteriors(*(list(PrepareData([test[0][t] for t in idx],
                                                                [test[1][t] for t in idx],
                                                                [test[2][t] for t in idx]))[:-1] +  
                                               [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test])
    if not (output is None):
        errs = np.concatenate([testModel(*(list(PrepareData([test[0][t] for t in idx],
                                                            [test[1][t] for t in idx],
                                                            [test[2][t] for t in idx])) +
                                           [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test])
        testError = np.mean(errs)
        print('Error: {0}'.format(testError))

    tEnd = time.time()
    print('Total running time: {0}'.format(tEnd - tStart))
    return testPosteriors

def ReprExtractor_augFeat(test, test_augFeat, classifier, inputList, augInput, batchSize=100, ifDropout=None):
    
    # building theano functions
    print('building theano functions...')
    givens_test = [(ifDropout, np.array(0, dtype=ifDropout.dtype))] if not (ifDropout is None) else []
    f_extRepr = theano.function(inputs=inputList + [augInput],
                                outputs=classifier.get_mlpInput(*(inputList + [augInput])),
                                givens=givens_test,
                                name='f_extRepr')

    # start testing
    print('start extracting representation...')
    tStart = time.time()

    
    kf_test = get_minibatches_idx(len(test[0]), batchSize, shuffle=False)
    testRepr = np.vstack([f_extRepr(*(list(PrepareData([test[0][t] for t in idx],
                                                       [test[1][t] for t in idx],
                                                       [test[2][t] for t in idx]))[:-1] +  
                                      [np.vstack([test_augFeat[t,:] for t in idx])])) for _, idx in kf_test])

    tEnd = time.time()
    print('Total running time: {0}'.format(tEnd - tStart))
    return testRepr
