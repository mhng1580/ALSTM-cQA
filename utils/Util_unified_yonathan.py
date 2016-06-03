import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import xml.etree.ElementTree as et
def build_data_cv_taskU(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None, test = False):
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
            RELC_RELEVANCE2ORGQ = RelComment.get('RELC_RELEVANCE2ORGQ').strip()

            if clean_string:
                RelCBody = clean_str(RelCBody)
            
            questionPair = RelQBody + " " + specialStr + " " + RelCBody
            words = set(questionPair.split())

            if test == False:
                for word in words:
                    vocab[word] += 1
        
            test_or_train = 1 # test = 0 and train = 1
            #if RELQ_ID in CV_info[mFold]:
            if ORGQ_ID in CV_info:
                if CV_info[ORGQ_ID]==1:
                    test_or_train = 0
            #Task A
            actual_LabelA = 1
            if RELC_RELEVANCE2RELQ.lower() == "bad":
                actual_LabelA = 0    
            if RELC_RELEVANCE2RELQ.lower() == "good":
                actual_LabelA = 2
            
            #Task C
            actual_LabelC = 1
            if RELC_RELEVANCE2ORGQ.lower() == "bad":
                actual_LabelC = 0    
            if RELC_RELEVANCE2ORGQ.lower() == "good":
                actual_LabelC = 2
 
            #Task B
            actual_LabelB = 2
            if RELQ_RELEVANCE2ORGQ.lower() == "irrelevant":
                actual_LabelB = 0    
            if RELQ_RELEVANCE2ORGQ.lower() == "relevant":
                actual_LabelB = 1

            actualKey = ORGQ_ID + "\t" + RELQ_ID + "\t" + RELC_ID
            intKey = len(Keys)
            Keys[intKey] = actualKey
            if len(RelQBody.split()) == 0:
                RelQBody = RelQSubject
                print "R empty",actualKey
            if len(RelCBody.split()) == 0:
                print actualKey
            tmp_datum  = {"yA":actual_LabelA,
                    "yB":actual_LabelB,
                    "yC":actual_LabelC,
                  "num_words": max(len(OrgQBody.split()), len(RelCBody.split())),
                  "RelC": RelCBody,
                  "RelQ": RelQBody,
                  "OrgQ": OrgQBody,
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
 
def build_data_cv_taskC(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None, test = False):
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
            if test == False:
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

def build_data_cv_taskA(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None, test = False):
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
        if test:
            RelQBody = RelQSubject+" "+RelQBody
       
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
            
            if test == False:
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
                print "R empty",actualKey, RelQSubject
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
 
def build_data_cv_taskD(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None):
    revs = []
    tmp_revs = []
    revsTmp = []
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

        actual_Label = 1
        if RELQ_RELEVANCE2ORGQ.lower() == "irrelevant":
            actual_Label = 0    
        if RELQ_RELEVANCE2ORGQ.lower() == "relevant":
            actual_Label = 1

        actualKey = ORGQ_ID + "\t" + RELQ_ID
        intKey = len(Keys)
        Keys[intKey] = actualKey

        tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelQBody.split())),
                  "pair_part1": RelQBody,
                  "pair_part2": OrgQBody,
                  "pair_part3": RelQSubject,
                  "split": test_or_train,
                  "key": intKey,
                  "keyOrg": ORGQ_ID,
                  "keyRel": RELQ_ID}
        revs.append(tmp_datum)
        revsTmp.append(tmp_datum)
    
    for rev1 in revsTmp:
        for rev2 in revsTmp:
            if rev1["keyOrg"] == rev2["keyOrg"]:
                if rev1["y"] == rev2["y"] and rev1["y"] == 1:
                    intKey = len(Keys)
                    Keys[intKey] = rev1["keyRel"] + "\t" + rev2["keyRel"]
                    tmp_datum = {"y":1,
                            "num_words":rev1["num_words"],
                            "pair_part1": rev1["pair_part1"],
                            "pair_part2": rev2["pair_part1"],
                            "split": rev1["split"],
                            "key": intKey}
                    revs.append(tmp_datum)
                else:
                    intKey = len(Keys)
                    Keys[intKey] = rev1["keyRel"] + "\t" + rev2["keyRel"]
                    tmp_datum = {"y":0,
                            "num_words":rev1["num_words"],
                            "pair_part1": rev1["pair_part1"],
                            "pair_part2": rev2["pair_part1"],
                            "split": rev1["split"],
                            "key": intKey}
                    revs.append(tmp_datum)

    return revs, vocab, Keys
  
def build_data_cv(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None, test = False):
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
        
        if test:
            RelQBody = RelQSubject+" "+RelQBody
       
        RELQ_RELEVANCE2ORGQ = RelQuestion.get('RELQ_RELEVANCE2ORGQ').strip()

        if clean_string:
            OrgQSubject = clean_str(OrgQSubject)
            OrgQBody = clean_str(OrgQBody)
            
            RelQSubject = clean_str(RelQSubject)
            RelQBody = clean_str(RelQBody)
        questionPair = OrgQBody + " " + specialStr + " " + RelQBody
        
        words = set(questionPair.split())
        if test == False:
            for word in words:
                vocab[word] += 1
        
        test_or_train = 1 # test = 0 and train = 1
        #if RELQ_ID in CV_info[mFold]:
        if ORGQ_ID in CV_info:
            if CV_info[ORGQ_ID]:
                test_or_train = 0

        actual_Label = 1
        if RELQ_RELEVANCE2ORGQ.lower() == "irrelevant":
            actual_Label = 0    
        if RELQ_RELEVANCE2ORGQ.lower() == "relevant":
            actual_Label = 1

        actualKey = ORGQ_ID + "\t" + RELQ_ID
        intKey = len(Keys)
        Keys[intKey] = actualKey

        if len(RelQBody.split()) == 0:
            RelQBody = RelQSubject
            print "R empty",actualKey, RelQSubject

        tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelQBody.split())),
                  "pair_part1": RelQBody,
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

def build_data_cv_y(orig_file, rel_file, label_file, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = 0, vocab = None, test = False):
    revs = []
    tmp_revs = []
    if vocab == None:
        vocab = defaultdict(float) # default value of float is 0.0
    Keys = defaultdict(int) # default value of float is 0.0
    
    origs = open(orig_file).readlines()
    rels = open(rel_file).readlines()
    labels = open(label_file).readlines()
    for (i,orig) in enumerate(origs):
        orig = orig.strip()
        orig_c = orig.split()
        ORGQ_ID = orig_c[0]
        OrgQBody = " ".join(orig_c[1:])

        rel = rels[i].strip()
        rel_c = rel.split()
        RELQ_ID = rel_c[0]
        RelQBody = " ".join(rel_c[1:])

        
        label = labels[i].strip()
        label_c = label.split()
        RELQ_RELEVANCE2ORGQ = label_c[len(label_c)-1]
       
        test_or_train = 1 # test = 0 and train = 1
        #if RELQ_ID in CV_info[mFold]:
        if ORGQ_ID in CV_info:
            if CV_info[ORGQ_ID]:
                test_or_train = 0

        questionPair = OrgQBody + " " + specialStr + " " + RelQBody
        words = set(questionPair.split())
        
        if test == False:
            for word in words:
                vocab[word] += 1
 

        actual_Label = 1
        if RELQ_RELEVANCE2ORGQ.lower() == "false":
            actual_Label = 0    
        if RELQ_RELEVANCE2ORGQ.lower() == "true":
            actual_Label = 1

        actualKey = ORGQ_ID + "\t" + RELQ_ID
        intKey = len(Keys)
        Keys[intKey] = actualKey

        if len(RelQBody.split()) == 0:
            RelQBody = RelQSubject
            print "R empty",actualKey, RelQSubject

        tmp_datum  = {"y":actual_Label,
                  "num_words": max(len(OrgQBody.split()), len(RelQBody.split())),
                  "pair_part1": RelQBody,
                  "pair_part2": OrgQBody,
                  "pair_part3": RelQBody,
                  "split": test_or_train,
                  "key": intKey}
        revs.append(tmp_datum)
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
    print fname
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
    np.random.seed(42)

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
            if word in vocab:
                q1.append(vocab[word])
                word1.append(W2[vocab[word]])
            else:
                q1.append(0)
                word1.append(W2[0])

        for word in rev["pair_part2"].split():
            if word in vocab:
                q2.append(vocab[word])
                word2.append(W2[vocab[word]])
            else:
                q2.append(0)
                word2.append(W2[0])

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
    oovNum = 0
    for rev in revs:
        q1 = []
        q2 = []
        for word in rev["pair_part1"].split():
            if word in vocab:
                q1.append(vocab[word])
            else:
                q1.append(0)
                print word
        for word in rev["pair_part2"].split():
            if word in vocab:
                q2.append(vocab[word])
            else:
                q2.append(0)
                print word
        x1.append(q1)
        x2.append(q2)
        y.append(rev["y"])
        cv.append(rev["split"])
    return x1,x2,y,cv, oovNum

def packDataU (revs, W, W2, word_idx_map, vocab, keys):
    x1 = []
    x2 = []
    x3 = []
    y = []
    cv = []
    for rev in revs:
        q1 = []
        q2 = []
        c  = []
        yTmp = []
        for word in rev["OrgQ"].split():
            q1.append(vocab[word])
        for word in rev["RelQ"].split():
            q2.append(vocab[word])
        for word in rev["RelC"].split():
            c.append(vocab[word])

        x1.append(q1)
        x2.append(q2)
        x3.append(c)
        yTmp.append(rev["yC"])
        yTmp.append(rev["yA"])
        yTmp.append(rev["yB"])
        y.append(yTmp)
        cv.append(rev["split"])
    return x1,x2,x3,y,cv




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
    #xml_filePath_dev = sys.argv[3]
    #xml_filePath_test = sys.argv[4]
    name = sys.argv[3]
    orig_file = sys.argv[4]
    rel_file = sys.argv[5]
    label_file = sys.argv[6]

    orig_dev_file = sys.argv[7]
    rel_dev_file = sys.argv[8]
    label_dev_file = sys.argv[9]


    #CV_infoPath = sys.argv[3] #"dataset_semeval2016/CV_subtaskB_ver2.csv" #sys.argv[3]
    foldNo = 0
    
    
    CV_info = readTestSplit(xml_filePath) # CV_info[0] gives the first column
    #CV_info = np.recfromcsv(CV_infoPath, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ') # CV_info[0] gives the first row

    #revs, vocab, keys = build_data_cv(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo)
    #revs_D, vocab, keysD = build_data_cv_taskD(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo)
    #revs_dev, vocab, keys_dev = build_data_cv(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocab)

    #revsC, vocabC, keysC = build_data_cv_taskC(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocab)
    #revsC_dev, vocabC, keysC_dev = build_data_cv_taskC(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabC)

    #revsA, vocabA, keysA = build_data_cv_taskA(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabC)
    #revsA_dev, vocabA, keysA_dev = build_data_cv_taskA(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabA)

    #revsU, vocabU, keysU = build_data_cv_taskU(xml_filePath, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabA)
    #revsU_dev, vocabU, keysU_dev = build_data_cv_taskU(xml_filePath_dev, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabA)


    revs_train_y, vocab_y, keys_train_y = build_data_cv_y(orig_file, rel_file, label_file, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, test = False)
    revs_dev_y, vocabA, keys_dev_y = build_data_cv_y(orig_dev_file, rel_dev_file, label_dev_file, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocab_y, test = True)
    #revs_train_y, vocab_y, keys_train_y = build_data_cv_y(xml_filePath_test, CV_info, specialStr = "<tofixlength>", clean_string = True, mFold = foldNo, vocab = vocabA, test = True)


    max_l = np.max(pd.DataFrame(revs_train_y)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs_train_y))
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

    # TaskB_y
    x1,x2,y,cv,oovNum= packData(revs_dev_y, W, W2, word_idx_map, word_idx_map, keys_dev_y)
    cPickle.dump([x1,x2,y,word_idx_map,W,W2,keys_dev_y,cv], open("cqa_taskB.p." + name + '.'  + str(foldNo), "wb"))
    
    print "Number of OOV: ", oovNum
    


    #x1,x2,y,emb1,emb2= packEmbData(revs, W, W2, word_idx_map, vocab, keys)
    #cPickle.dump([x1,x2,y,emb1,emb2], open("cqa_taskA_emb.p_" + str(foldNo), "wb"))
 

    #vocab contains df of the words (present or absent in the docs)
    #revs -> is a list in which each element has 4 fields: y(actual label, e.g., pos or neg), text(text, e.g, sentence or review), num_words, split(the instance is randomly assigned to one CV from 10) 
    print "dataset created!"
    
