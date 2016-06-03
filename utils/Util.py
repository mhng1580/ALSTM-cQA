import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import xml.etree.ElementTree as et

def build_data_cv_taskC(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None):
    revs = []
    tmp_revs = []
    if vocab == None:
        vocab = defaultdict(float) # default value of float is 0.0
    Keys = defaultdict(int) # default value of float is 0.0
    
    xml_root = et.parse(xml_filePath).getroot()
    for OrgQuestion in xml_root.findall('OrgQuestion'):
        ORGQ_ID = OrgQuestion.get('ORGQ_ID').strip()
        OrgQSubject = OrgQuestion.find('OrgQSubject').text
        if OrgQSubject is not None:
            OrgQSubject = OrgQSubject.strip()
        else:
            OrgQSubject = ""
        OrgQBody = OrgQuestion.find('OrgQBody').text
        if OrgQBody is not None:
            OrgQBody = OrgQBody.strip()
        else:
            OrgQBody = ""

        RelQuestion = OrgQuestion.find('Thread').find('RelQuestion')
        RELQ_ID = RelQuestion.get('RELQ_ID').strip()
        RelQSubject = RelQuestion.find('RelQSubject').text
        if RelQSubject is not None:
            RelQSubject = RelQSubject.strip()
        else:
            RelQSubject = ""
        RelQBody = RelQuestion.find('RelQBody').text
        if RelQBody is not None:
            RelQBody = RelQBody.strip()
        else:
            RelQBody = ""
        
        RELQ_RELEVANCE2ORGQ = RelQuestion.get('RELQ_RELEVANCE2ORGQ').strip()

        if clean_string:
            OrgQSubject = clean_str(OrgQSubject)
            OrgQBody = clean_str(OrgQBody)
            
            RelQSubject = clean_str(RelQSubject)
            RelQBody = clean_str(RelQBody)

        RelComments = OrgQuestion.find('Thread').findall('RelComment')
        for RelComment in RelComments:
            RELC_ID = RelComment.get('RELC_ID').strip()
            RelCBody = RelComment.find('RelCText').text
            if RelCBody is not None:
                RelCBody = RelCBody.strip()
            else:
                RelCBody = ""
        
            RELC_RELEVANCE2ORGQ = RelComment.get('RELC_RELEVANCE2ORGQ').strip()

            if clean_string:
                RelCBody = clean_str(RelCBody)
            
            questionPair = OrgQBody + " " + specialStr + " " + RelCBody
            words = set(questionPair.split())

            for word in words:
                vocab[word] += 1
        
            test_or_train = 1 # test = 0 and train = 1
            #if RELQ_ID in CV_info[mFold]:
            if ORGQ_ID in CV_info:
                if CV_info[ORGQ_ID]==1:
                    test_or_train = 0

            actual_Label = 1
            if RELC_RELEVANCE2ORGQ.lower() == "bad":
                actual_Label = 0    
            if RELC_RELEVANCE2ORGQ.lower() == "good":
                actual_Label = 2
            
            actualKey = ORGQ_ID + "\t" + RELC_ID
            intKey = len(Keys)
            Keys[intKey] = actualKey
            if len(OrgQBody.split()) == 0:
                print actualKey
            if len(RelCBody.split()) == 0:
                print actualKey
            tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelCBody.split())),
                  "pair_part1": RelCBody,
                  "pair_part2": OrgQBody,
                  "split": test_or_train,
                  "key": intKey}
            revs.append(tmp_datum)
    return revs, vocab, Keys

def build_data_cv_taskA(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None):
    revs = []
    tmp_revs = []
    if vocab == None:
        vocab = defaultdict(float) # default value of float is 0.0
    Keys = defaultdict(int) # default value of float is 0.0
    
    xml_root = et.parse(xml_filePath).getroot()
    for OrgQuestion in xml_root.findall('OrgQuestion'):
        ORGQ_ID = OrgQuestion.get('ORGQ_ID').strip()
        OrgQSubject = OrgQuestion.find('OrgQSubject').text
        if OrgQSubject is not None:
            OrgQSubject = OrgQSubject.strip()
        else:
            OrgQSubject = ""
        OrgQBody = OrgQuestion.find('OrgQBody').text
        if OrgQBody is not None:
            OrgQBody = OrgQBody.strip()
        else:
            OrgQBody = ""

        RelQuestion = OrgQuestion.find('Thread').find('RelQuestion')
        RELQ_ID = RelQuestion.get('RELQ_ID').strip()
        RelQSubject = RelQuestion.find('RelQSubject').text
        if RelQSubject is not None:
            RelQSubject = RelQSubject.strip()
        else:
            RelQSubject = ""
        RelQBody = RelQuestion.find('RelQBody').text
        if RelQBody is not None:
            RelQBody = RelQBody.strip()
        else:
            RelQBody = ""
        
        RELQ_RELEVANCE2ORGQ = RelQuestion.get('RELQ_RELEVANCE2ORGQ').strip()

        if clean_string:
            OrgQSubject = clean_str(OrgQSubject)
            OrgQBody = clean_str(OrgQBody)
            
            RelQSubject = clean_str(RelQSubject)
            RelQBody = clean_str(RelQBody)

        RelComments = OrgQuestion.find('Thread').findall('RelComment')
        for RelComment in RelComments:
            RELC_ID = RelComment.get('RELC_ID').strip()
            RelCBody = RelComment.find('RelCText').text
            if RelCBody is not None:
                RelCBody = RelCBody.strip()
            else:
                RelCBody = ""
        
            RELC_RELEVANCE2RELQ = RelComment.get('RELC_RELEVANCE2RELQ').strip()

            if clean_string:
                RelCBody = clean_str(RelCBody)
            
            questionPair = RelQBody + " " + specialStr + " " + RelCBody
            words = set(questionPair.split())

            for word in words:
                vocab[word] += 1
        
            test_or_train = 1 # test = 0 and train = 1
            #if RELQ_ID in CV_info[mFold]:
            if ORGQ_ID in CV_info:
                if CV_info[ORGQ_ID]==1:
                    test_or_train = 0

            actual_Label = 1
            if RELC_RELEVANCE2RELQ.lower() == "bad":
                actual_Label = 0    
            if RELC_RELEVANCE2RELQ.lower() == "good":
                actual_Label = 2
            
            actualKey = RELQ_ID + "\t" + RELC_ID
            intKey = len(Keys)
            Keys[intKey] = actualKey
            if len(RelQBody.split()) == 0:
                RelQBody = RelQSubject
                print "R empty",actualKey
            if len(RelCBody.split()) == 0:
                print actualKey
            tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelCBody.split())),
                  "pair_part1": RelCBody,
                  "pair_part2": RelQBody,
                  "pair_part3": RelQSubject,
                  "split": test_or_train,
                  "key": intKey}
            revs.append(tmp_datum)


    #max_l_pair_onePart = np.max(pd.DataFrame(tmp_revs)["max_num_words_pairs"])
    #for rev in tmp_revs:
    #    pair_part1 = rev["pair_part1"]
    #    pair_part1 = ' '.join(pair_part1.split() + [specialStr]*(max_l_pair_onePart-len(pair_part1.split())+4))

    #    pair_part2 = rev["pair_part2"]
    #    pair_part2 = ' '.join([specialStr]*4 + pair_part2.split() + [specialStr]*(max_l_pair_onePart-len(pair_part2.split())))

    #    questionPair = pair_part1 + " " + pair_part2
    #    words = set(questionPair.split())
    #    for word in words:
    #        vocab[word] += 1

    #    datum  = {"y":rev["y"],
    #              "text": questionPair,
    #              "num_words": len(questionPair.split()),
    #              "split": rev["split"],
    #              "key": rev["key"]}

    #    revs.append(datum)


    return revs, vocab, Keys
 
  
def build_data_cv(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None):
    revs = []
    tmp_revs = []
    if vocab == None:
        vocab = defaultdict(float) # default value of float is 0.0
    Keys = defaultdict(int) # default value of float is 0.0
    
    xml_root = et.parse(xml_filePath).getroot()
    for OrgQuestion in xml_root.findall('OrgQuestion'):
        ORGQ_ID = OrgQuestion.get('ORGQ_ID').strip()
        OrgQSubject = OrgQuestion.find('OrgQSubject').text
        if OrgQSubject is not None:
            OrgQSubject = OrgQSubject.strip()
        else:
            OrgQSubject = ""
        OrgQBody = OrgQuestion.find('OrgQBody').text
        if OrgQBody is not None:
            OrgQBody = OrgQBody.strip()
        else:
            OrgQBody = ""

        RelQuestion = OrgQuestion.find('Thread').find('RelQuestion')
        RELQ_ID = RelQuestion.get('RELQ_ID').strip()
        RelQSubject = RelQuestion.find('RelQSubject').text
        if RelQSubject is not None:
            RelQSubject = RelQSubject.strip()
        else:
            RelQSubject = ""
        RelQBody = RelQuestion.find('RelQBody').text
        if RelQBody is not None:
            RelQBody = RelQBody.strip()
        else:
            RelQBody = ""
        
        RELQ_RELEVANCE2ORGQ = RelQuestion.get('RELQ_RELEVANCE2ORGQ').strip()

        if clean_string:
            OrgQSubject = clean_str(OrgQSubject)
            OrgQBody = clean_str(OrgQBody)
            
            RelQSubject = clean_str(RelQSubject)
            RelQBody = clean_str(RelQBody)
        questionPair = OrgQBody + " " + specialStr + " " + RelQBody
        
        words = set(questionPair.split())
        for word in words:
            vocab[word] += 1
        
        test_or_train = 1 # test = 0 and train = 1
        #if RELQ_ID in CV_info[mFold]:
        if ORGQ_ID in CV_info:
            if CV_info[ORGQ_ID]:
                test_or_train = 0

        actual_Label = 2
        if RELQ_RELEVANCE2ORGQ.lower() == "irrelevant":
            actual_Label = 0    
        if RELQ_RELEVANCE2ORGQ.lower() == "relevant":
            actual_Label = 1

        actualKey = ORGQ_ID + "\t" + RELQ_ID
        intKey = len(Keys)
        Keys[intKey] = actualKey

        tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelQBody.split())),
                  "pair_part1": RelQSubject,
                  "pair_part2": OrgQBody,
                  "pair_part3": RelQSubject,
                  "split": test_or_train,
                  "key": intKey}
        revs.append(tmp_datum)


    #max_l_pair_onePart = np.max(pd.DataFrame(tmp_revs)["max_num_words_pairs"])
    #for rev in tmp_revs:
    #    pair_part1 = rev["pair_part1"]
    #    pair_part1 = ' '.join(pair_part1.split() + [specialStr]*(max_l_pair_onePart-len(pair_part1.split())+4))

    #    pair_part2 = rev["pair_part2"]
    #    pair_part2 = ' '.join([specialStr]*4 + pair_part2.split() + [specialStr]*(max_l_pair_onePart-len(pair_part2.split())))

    #    questionPair = pair_part1 + " " + pair_part2
    #    words = set(questionPair.split())
    #    for word in words:
    #        vocab[word] += 1

    #    datum  = {"y":rev["y"],
    #              "text": questionPair,
    #              "num_words": len(questionPair.split()),
    #              "split": rev["split"],
    #              "key": rev["key"]}

    #    revs.append(datum)


    return revs, vocab, Keys
  
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

    word_vecs['<tofixlength>'] = np.zeros(k) #add it to the tail of sentences to make their length equal to each other before their concatenation

def clean_str(string, ChangedToLower=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased if ChangedToLower is true
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower() if ChangedToLower else string.strip()

def packEmbData (revs, W, W2, word_idx_map, vocab, keys):
    x1 = []
    x2 = []
    y = []
    emb1 = []
    emb2 = []
    for rev in revs:
        q1 = []
        q2 = []
        word1 = []
        word2 = []
        for word in rev["pair_part1"].split():
            q1.append(vocab[word])
            word1.append(W2[vocab[word]])
        for word in rev["pair_part2"].split():
            q2.append(vocab[word])
            word2.append(W2[vocab[word]])
        x1.append(q1)
        x2.append(q2)
        y.append(rev["y"])
        emb1.append(word1)
        emb2.append(word2)
    return x1,x2,y,emb1, emb2

def packData (revs, W, W2, word_idx_map, vocab, keys):
    x1 = []
    x2 = []
    y = []
    cv = []
    for rev in revs:
        q1 = []
        q2 = []
        for word in rev["pair_part1"].split():
            q1.append(vocab[word])
        for word in rev["pair_part2"].split():
            q2.append(vocab[word])
        x1.append(q1)
        x2.append(q2)
        y.append(rev["y"])
        cv.append(rev["split"])
    return x1,x2,y,cv

def readTestSplit(xml_filepath, test_portion=0.10):
    id_info = []
    cv_info = {}
    xml_root = et.parse(xml_filePath).getroot()
    for OrgQuestion in xml_root.findall('OrgQuestion'):
        ORGQ_ID = OrgQuestion.get('ORGQ_ID').strip()
        if not ORGQ_ID in cv_info:
            cv_info[ORGQ_ID] = 1
            id_info.append(ORGQ_ID)
    np.random.seed(42)
    n_samples = len(id_info)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - test_portion)))
    for s in sidx[:n_train]:
        cv_info[id_info[s]] = 0
    return cv_info

    
if __name__=="__main__":
    w2v_file = sys.argv[1] #"/data/sls/scratch/mitra/mooc/ABSA/w2vec/GoogleNews-vectors-negative300.bin" #"/Users/mitra/Documents/QA/w2v-Materials/GoogleNews-vectors-negative300.bin" #sys.argv[1]
    xml_filePath = sys.argv[2] #"dataset_semeval2016/CQA-QL-train.xml" #sys.argv[2]
    xml_filePath_dev = sys.argv[3]
    #CV_infoPath = sys.argv[3] #"dataset_semeval2016/CV_subtaskB_ver2.csv" #sys.argv[3]
    foldNo = 0
    
    
    CV_info = readTestSplit(xml_filePath) # CV_info[0] gives the first column
    #CV_info = np.recfromcsv(CV_infoPath, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ') # CV_info[0] gives the first row

    revs, vocab, keys = build_data_cv(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo)
    revs_dev, vocab, keys_dev = build_data_cv(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocab)

    revsC, vocabC, keysC = build_data_cv_taskC(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo)
    revsC_dev, vocabC, keysC_dev = build_data_cv_taskC(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabC)

    revsA, vocabA, keysA = build_data_cv_taskA(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo)
    revsA_dev, vocabA, keysA_dev = build_data_cv_taskA(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabA)


    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    x1,x2,y,cv = packData(revs, W, W2, word_idx_map, vocab, keys)
    cPickle.dump([x1,x2,y,vocab,W,W2,keys,cv], open("cqa_taskB.p_" + str(foldNo), "wb"))
 
    x1,x2,y,cv = packData(revs_dev, W, W2, word_idx_map, vocab, keys_dev)
    cPickle.dump([x1,x2,y,vocab,W,W2,keys_dev,cv], open("cqa_taskB_dev.p_" + str(foldNo), "wb"))
 
    max_l = np.max(pd.DataFrame(revsC)["num_words"])

    print "data loaded!"
    print "number of sentences: " + str(len(revsC))
    print "vocab size: " + str(len(vocabC))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocabC)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))

    add_unknown_words(w2v, vocabC)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocabC)
    W2, _ = get_W(rand_vecs)

    xC1,xC2,yC,cv = packData(revsC, W, W2, word_idx_map, vocabC, keysC)
    cPickle.dump([xC1,xC2,yC,vocabC,W,W2, keysC,cv], open("cqa_taskC.p_" + str(foldNo), "wb"))

    xC1,xC2,yC,cv = packData(revsC_dev, W, W2, word_idx_map, vocabC, keysC_dev)
    cPickle.dump([xC1,xC2,yC,vocabC,W,W2, keysC_dev,cv], open("cqa_taskC_dev.p_" + str(foldNo), "wb"))


    max_l = np.max(pd.DataFrame(revsA)["num_words"])

    print "data loaded!"
    print "number of sentences: " + str(len(revsA))
    print "vocab size: " + str(len(vocabA))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocabA)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))

    add_unknown_words(w2v, vocabA)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocabA)
    W2, _ = get_W(rand_vecs)

    xC1,xC2,yC,cv = packData(revsA, W, W2, word_idx_map, vocabA, keysA)
    cPickle.dump([xC1,xC2,yC,vocabA,W,W2, keysA,cv], open("cqa_taskA.p_" + str(foldNo), "wb"))

    xC1,xC2,yC,cv = packData(revsA_dev, W, W2, word_idx_map, vocabA, keysA_dev)
    cPickle.dump([xC1,xC2,yC,vocabA,W,W2, keysA_dev,cv], open("cqa_taskA_dev.p_" + str(foldNo), "wb"))



    #x1,x2,y,emb1,emb2= packEmbData(revs, W, W2, word_idx_map, vocab, keys)
    #cPickle.dump([x1,x2,y,emb1,emb2], open("cqa_taskA_emb.p_" + str(foldNo), "wb"))
 

    #vocab contains df of the words (present or absent in the docs)
    #revs -> is a list in which each element has 4 fields: y(actual label, e.g., pos or neg), text(text, e.g, sentence or review), num_words, split(the instance is randomly assigned to one CV from 10) 
    print "dataset created!"
    
