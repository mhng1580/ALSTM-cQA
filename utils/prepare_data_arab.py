from __future__ import print_function
import sys
import cPickle
import xml.etree.ElementTree as ET
from collections import OrderedDict

def CleanStr(s):
    if not type(s) is str:
        return ''
    import re, string
    regex_p = re.compile('[{0}]'.format(re.escape(string.punctuation)))
    regex_d = re.compile('[{0}]+'.format(string.digits))
    return regex_d.sub('<NUMBER>', regex_p.sub('', s))

def Xml2Pkl(xmlFile, w2i, pklFile, val_portion=0., ifDumpMt=False):
    print('generating pkl for: {0}'.format(xmlFile))
    tree = ET.parse(xmlFile)
    import numpy.random as nprandom
    numOriQ = len(tree.getroot())
    permutedIdx = nprandom.permutation(range(numOriQ))
    trainIdx = permutedIdx[round(numOriQ * val_portion):]
    valIdx = permutedIdx[:round(numOriQ * val_portion)]
    train = ExtQaObjList([tree.getroot()[i] for i in trainIdx], w2i)
    val = ExtQaObjList([tree.getroot()[i] for i in valIdx], w2i)
    print('Train size: {0}, Validation Size: {1}'.format(len(train[0]), len(val[0])))
    data_C = [train[0] + val[0],
              train[2] + val[2],
              train[3] + val[3],
              None,
              None,
              None,
              dict(zip(xrange(len(train[0]) + len(val[0])), train[4] + val[4])),
              [1] * len(train[0]) + [0] * len(val[0])]
    print('Dumping pkl file to: {0}...'.format(pklFile))
    cPickle.dump(data_C, open(pklFile, 'wb'))
    if ifDumpMt:
        import os
        data_D = [train[0] + val[0],
                  train[1] + val[1],
                  train[2] + val[2],
                  train[3] + val[3],
                  None,
                  None,
                  None,
                  dict(zip(xrange(len(train[0]) + len(val[0])), train[4] + val[4])),
                  [1] * len(train[0]) + [0] * len(val[0])]
        name, ext = os.path.splitext(pklFile)
        pklFile_mt = name + '_mt' + ext
        print('Dumping pkl file to: {0}...'.format(pklFile_mt))
        cPickle.dump(data_D, open(pklFile_mt, 'wb'))

def Xml2Pkl_QQ(xmlFile, w2i, pklFile, val_portion=0.):
    print('generating pkl for: {0}'.format(xmlFile))
    tree = ET.parse(xmlFile)
    import numpy.random as nprandom
    numOriQ = len(tree.getroot())
    permutedIdx = nprandom.permutation(range(numOriQ))
    trainIdx = permutedIdx[round(numOriQ * val_portion):]
    valIdx = permutedIdx[:round(numOriQ * val_portion)]
    train = ExtQaObjList([tree.getroot()[i] for i in trainIdx], w2i)
    val = ExtQaObjList([tree.getroot()[i] for i in valIdx], w2i)
    print('Train size: {0}, Validation Size: {1}'.format(len(train[0]), len(val[0])))
    data_C = [train[0] + val[0],
              train[1] + val[1],
              train[3] + val[3],
              None,
              None,
              None,
              dict(zip(xrange(len(train[0]) + len(val[0])), train[4] + val[4])),
              [1] * len(train[0]) + [0] * len(val[0])]
    print('Dumping pkl file to: {0}...'.format(pklFile))
    cPickle.dump(data_C, open(pklFile, 'wb'))

def ExtQaObjList(qaObjList, w2i):
    l2i = {'I':0, 'R':1, 'D':2}
    oriQList = []
    relQList = []
    relAList = []
    labelList = []
    keyList = []
    for qaObj in qaObjList:
        qId = qaObj.attrib['QID']
        oriQObj = qaObj[0]
        if oriQObj.tag != 'Qtext':
            sys.stderr.write('MISSING original question text: QID {0}\n'.format(qId))
            continue
        oriQ = [w2i[word] if word in w2i else 0 for word in CleanStr(oriQObj.text).split()]
        for relQaObj in qaObj[1:]:
            qaId = relQaObj.attrib['QAID']
            qaLabel = l2i[relQaObj.attrib['QArel']]
            if (len(relQaObj) < 2) or (relQaObj[0].tag != 'QAquestion') or (relQaObj[1].tag != 'QAanswer'):
                sys.stderr.write('MISSING related question or comment: QAID {0}\n'.format(qaId))
                continue
            relQ = [w2i[word] if word in w2i else 0 for word in CleanStr(relQaObj[0].text).split()]
            relA = [w2i[word] if word in w2i else 0 for word in CleanStr(relQaObj[1].text).split()]
            oriQList.append(oriQ)
            relQList.append(relQ)
            relAList.append(relA)
            labelList.append(qaLabel)
            keyList.append('{0}_{1}'.format(qId, qaId))
    return oriQList, relQList, relAList, labelList, keyList

def Xmls2Dict(xmlFileList, pklFile=None):
    w2i = OrderedDict()
    for xmlFile in xmlFileList:
        tree = ET.parse(xmlFile)
        for qaObj in tree.getroot():
            oriQ = qaObj[0]
            qaPairs = qaObj[1:]
            for word in CleanStr(oriQ.text).split():
                w2i[word] = len(w2i) + 1 if not (word in w2i) else w2i[word]
            for qa in qaPairs:
                words = []
                for qOrA in qa:
                    words = words + CleanStr(qOrA.text).split()
                for word in words:
                    w2i[word] = len(w2i) + 1 if not (word in w2i) else w2i[word]
    if not (pklFile is None):
        print('writing dictionary to: {0}'.format(pklFile))
        i2w = {i:w for w, i in w2i.iteritems()}
        i2w[0] = '<OOV>'
        cPickle.dump((w2i, i2w), open(pklFile, 'wb'))
    return w2i

if __name__ == '__main__':
    # trainFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/SemEval2016-Task3-CQA-Arabic-MD-train-v1.1/SemEval2016-Task3-CQA-MD-train.xml'
    # devFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/SemEval2016-Task3-CQA-Arabic-MD-train-v1.1/SemEval2016-Task3-CQA-MD-dev.xml'
    # w2i = Xmls2Dict([trainFile, devFile], '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDict.pkl')
    # print('vocab size: {0}'.format(len(w2i)))
    # Xml2Pkl(devFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicDev.pkl', ifDumpMt=True)
    # Xml2Pkl(trainFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arabicTrain.pkl', val_portion=0.1, ifDumpMt=True)
    
    trainFile = '/data/sls/scratch/sharedData/SemEval2016/taskD/SemEval2016-Task3-CQA-MD-train.lemma.xml'
    devFile = '/data/sls/scratch/sharedData/SemEval2016/taskD/SemEval2016-Task3-CQA-MD-dev.lemma.xml'
    # w2i = Xmls2Dict([trainFile, devFile], '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicDict.pkl')
    # print('vocab size: {0}'.format(len(w2i)))
    # Xml2Pkl(devFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicDev.pkl', ifDumpMt=True)
    # Xml2Pkl(trainFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicTrain.pkl', val_portion=0.1, ifDumpMt=True)

    # w2i = Xmls2Dict([trainFile, devFile], '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicQQDict.pkl')
    # print('vocab size: {0}'.format(len(w2i)))
    # Xml2Pkl_QQ(devFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicQQDev.pkl')
    # Xml2Pkl_QQ(trainFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicQQTrain.pkl', val_portion=0.1)
    
    testFile = '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/SemEval2016-Task3-CQA-MD-test-input-Arabic.lemma.xml'
    w2i = cPickle.load(open('/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicDict.pkl', 'rb'))[0]
    print('vocab size: {0}'.format(len(w2i)))
    Xml2Pkl_QQ(testFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicQQTest.pkl')
    Xml2Pkl(testFile, w2i, '/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/data/arab_lemma/arabicTest.pkl', ifDumpMt=True)
