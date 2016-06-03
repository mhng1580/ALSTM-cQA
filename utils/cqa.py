import cPickle
import gzip
import os

import numpy as np
import theano
import string
# def PrepareData2(seqs1, seqs2, seqs3, labels, maxlen=None):
#     """Create the matrices from the datasets.
# 
#     This pad each sequence to the same lenght: the lenght of the
#     longuest sequence or maxlen.
# 
#     if maxlen is set, we will cut all sequence to this maximum
#     lenght.
# 
#     This swap the axis!
#     """
#     # x: a list of sentences
#     lengths1 = [len(s) for s in seqs1]
#     lengths2 = [len(s) for s in seqs2]
#     lengths3 = [len(s) for s in seqs3]
# 
# 
#     n_samples = len(seqs1)
#     maxlen1 = np.max(lengths1)
#     maxlen2 = np.max(lengths2)
#     maxlen3 = np.max(lengths3)
#     #maxlen = np.max(maxlen1,maxlen2)
# 
#     x1 = np.zeros((maxlen1, n_samples)).astype('int64')
#     x_mask1 = np.zeros((maxlen1, n_samples)).astype(theano.config.floatX)
#     x2 = np.zeros((maxlen2, n_samples)).astype('int64')
#     x_mask2 = np.zeros((maxlen2, n_samples)).astype(theano.config.floatX)
#     x3 = np.zeros((maxlen3, n_samples)).astype('int64')
#     x_mask3 = np.zeros((maxlen3, n_samples)).astype(theano.config.floatX)
#     for idx, s in enumerate(seqs1):
#         x1[:lengths1[idx], idx] = s
#         x_mask1[:lengths1[idx], idx] = 1.
#     for idx, s in enumerate(seqs2):
#         x2[:lengths2[idx], idx] = s
#         x_mask2[:lengths2[idx], idx] = 1.
#     for idx, s in enumerate(seqs3):
#         x3[:lengths3[idx], idx] = s
#         x_mask3[:lengths3[idx], idx] = 1.
# 
# 
# 
#     return x1, x_mask1, x2, x_mask2, x3, x_mask3,  labels


def PrepareData(seqs1, seqs2, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths1 = [len(s) for s in seqs1]
    lengths2 = [len(s) for s in seqs2]

    if maxlen is not None:
        new_seqs1 = []
        new_seqs2 = []
        new_labels = []
        new_lengths1 = []
        new_lengths2 = []
        for l, s, y in zip(lengths1, seqs1, labels):
            if l < maxlen:
                new_seqs1.append(s)
                new_labels.append(y)
                new_lengths1.append(l)
        lengths1 = new_lengths1
        labels = new_labels
        seqs1 = new_seqs1

        for l, s, y in zip(lengths2, seqs2, labels):
            if l < maxlen:
                new_seqs2.append(s)
                new_lengths2.append(l)
        lengths2 = new_lengths2
        seqs2 = new_seqs2


        if len(lengths1) < 1:
            return None, None, None

    n_samples = len(seqs1)
    maxlen1 = np.max(lengths1)
    maxlen2 = np.max(lengths2)
    #maxlen = np.max(maxlen1,maxlen2)

    x1 = np.zeros((maxlen1, n_samples)).astype('int64')
    x_mask1 = np.zeros((maxlen1, n_samples)).astype(theano.config.floatX)
    x2 = np.zeros((maxlen2, n_samples)).astype('int64')
    x_mask2 = np.zeros((maxlen2, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs1):
        x1[:lengths1[idx], idx] = s
        x_mask1[:lengths1[idx], idx] = 1.
    for idx, s in enumerate(seqs2):
        x2[:lengths2[idx], idx] = s
        x_mask2[:lengths2[idx], idx] = 1.


    return x1, x_mask1, x2, x_mask2, labels

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def SaveData(test, pos, path, lang=None):
    results = []
    for i,j in enumerate(test[3]):
        res = 'false'
        if lang==None:
            if (pos[i][len(pos[i])-1]) > (pos[i][0]):
                res = 'true'
            results.append([j,str(0),str(pos[i][len(pos[i])-1]),res])
        if lang=='Arabic':
            if (pos[i][len(pos[i])-1]+pos[i][len(pos[i])-2]) > (pos[i][0]):
                res = 'true'
            results.append([j,str(0),str(pos[i][len(pos[i])-1]),res])
            # results.append([j,str(0),str(pos[i][len(pos[i])-1]+pos[i][len(pos[i])-2]),res])


    def getKey1(item):
        [org, Rel] = item[0].split('\t')
        org = org.split('_R')[0]
        return org

    def getKey2(item):
        [org, Rel] = item[0].split('\t')
        [_, num] = Rel.split('_R')
        num = num.split('_C')[0] 
        return num
    def getKey3(item):
        [org, Rel] = item[0].split('\t')
        [_, num] = Rel.split('_R')
        [_, num] = num.split('_C') 
        return num

    def cmp(x,y):
        if getKey1(x) == getKey1(y):
            if int(getKey2(x)) == int(getKey2(y)):
                if int(getKey3(x)) < int(getKey3(y)):
                    return -1
                else:
                    return 1
            elif int(getKey2(x)) < int(getKey2(y)):
                return -1
            else:
                return 1 
        else:
            if getKey1(x) < getKey1(y):
                return -1
            else:
                return 1
    #results = ['\t'.join(i) for i in sorted(results,cmp=cmp)]
    results = ['\t'.join(i) for i in results]
    f = open(path, "w")
    f.write('\n'.join(results))
    #cPickle.dump([test[3],pos], open(path, "wb"))

# def SaveData2(test, pos, path):
#     results = []
#     for i,j in enumerate(test[3]):
#         res = 'false'
#         if (pos[i][len(pos[i])-1]+pos[i][len(pos[i])-2]) > (pos[i][0]):
#             res = 'true'
#         results.append([j,str(0),str(pos[i][len(pos[i])-1]+pos[i][len(pos[i])-2]),res])
# 
#     def getKey1(item):
#         [org, Rel] = item[0].split('\t')
#         org = org.split('_R')[0]
#         return org
# 
#     def getKey2(item):
#         [org, Rel] = item[0].split('\t')
#         [_, num] = Rel.split('_R')
#         num = num.split('_C')[0] 
#         return num
#     def getKey3(item):
#         [org, Rel] = item[0].split('\t')
#         [_, num] = Rel.split('_R')
#         [_, num] = num.split('_C') 
#         return num
# 
#     def cmp(x,y):
#         if getKey1(x) == getKey1(y):
#             if int(getKey2(x)) == int(getKey2(y)):
#                 if int(getKey3(x)) < int(getKey3(y)):
#                     return -1
#                 else:
#                     return 1
#             elif int(getKey2(x)) < int(getKey2(y)):
#                 return -1
#             else:
#                 return 1 
#         else:
#             if getKey1(x) < getKey1(y):
#                 return -1
#             else:
#                 return 1
#     results = ['\t'.join(i) for i in sorted(results,cmp=cmp)]
#     #results.append(str(j)+'\t'+str(0)+'\t'+str(pos[i][1])+'\t'+res)
#     #results.sort()
#     f = open(path, "w")
#     f.write('\n'.join(results))

def LoadData(path=None, n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=False):

    '''
    valid_portion: deprecated
    '''

    np.random.seed(777)

    # Load the dataset
    print path
    f = open(path, 'rb')
    train_set = cPickle.load(f)
    #test_set = cPickle.load(f)
    f.close()
    emb = train_set[4].astype(np.float32) if not (train_set[4] is None) else train_set[4]
    keys = train_set[6]
    cv = train_set[7]
    
    new_train_set_x1 = []
    new_train_set_x2 = []
    new_train_set_y = []
    new_train_set_key = []
    new_train_set_cv = []
    for x1, x2, y, key, c in zip(train_set[0], train_set[1], train_set[2], train_set[6],cv):
        if maxlen:
            if len(x1)==0 or len(x2) ==0:
                continue
        new_train_set_x1.append(x1)
        new_train_set_x2.append(x2)
        new_train_set_y.append(y)
        new_train_set_key.append(train_set[6][key])
        new_train_set_cv.append(c)
    train_set = (new_train_set_x1, new_train_set_x2, new_train_set_y, new_train_set_key)
    cv = new_train_set_cv
    del new_train_set_x1, new_train_set_x2, new_train_set_y, new_train_set_key
    # split training set into validation set
    train_set_x1, train_set_x2, train_set_y, train_set_key = train_set
    n_samples = len(train_set_x1)
    # fix the random seed
    np.random.seed(42)

    sidx = np.nonzero(cv)[0]
    cvsidx = np.where(np.array(cv)==0)[0]
    valid_set_x1 = [train_set_x1[s] for s in cvsidx]
    valid_set_x2 = [train_set_x2[s] for s in cvsidx]
    valid_set_y = [train_set_y[s] for s in cvsidx]
    valid_set_key = [train_set_key[s] for s in cvsidx]
    train_set_x1 = [train_set_x1[s] for s in sidx]
    train_set_x2 = [train_set_x2[s] for s in sidx]
    train_set_y = [train_set_y[s] for s in sidx]
    train_set_key = [train_set_key[s] for s in sidx]

    train_set = (train_set_x1, train_set_x2, train_set_y, train_set_key)
    valid_set = (valid_set_x1, valid_set_x2, valid_set_y, valid_set_key)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]
    def chop_utt(set1, set2, sety, setkey, maxlen):
        new_set1 = []
        new_set2 = []
        new_sety = []
        new_setkey = []

        for x1, x2, y, key in zip(set1, set2, sety, setkey):
            if len(x2) <= maxlen:
                new_set1.append(x1)
                new_set2.append(x2)
                new_sety.append(y)
                new_setkey.append(key)
            else:
                while len(x2) > maxlen:
                    new_set1.append(x1)
                    new_set2.append(x2[:maxlen])
                    new_sety.append(y)
                    new_setkey.append(key)
                    x2 = x2[maxlen+1:]
        return new_set1, new_set2, new_sety, new_setkey
    #test_set_x, test_set_y = test_set
    if maxlen:
        valid_set_x1, valid_set_x2, valid_set_y, valid_set_key = valid_set
        train_set_x1, train_set_x2, train_set_y, train_set_key = chop_utt(train_set[0], train_set[1], train_set[2], train_set[3], maxlen)
    
    #train_set_x1 = remove_unk(train_set_x1)
    #train_set_x2 = remove_unk(train_set_x2)
    #valid_set_x1 = remove_unk(valid_set_x1)
    #valid_set_x2 = remove_unk(valid_set_x2)
    #test_set_x = remove_unk(test_set_x)
    n_samples = len(train_set_x1)
    sidx = np.random.permutation(n_samples)
    train_set_x1 = [train_set_x1[s] for s in sidx[:n_samples]]
    train_set_x2 = [train_set_x2[s] for s in sidx[:n_samples]]
    train_set_y = [train_set_y[s] for s in sidx[:n_samples]]
    train_set_key = [train_set_key[s] for s in sidx[:n_samples]]


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        #sorted_index = len_argsort(test_set_x)
        #test_set_x = [test_set_x[i] for i in sorted_index]
        #test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x1)
        valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
        valid_set_x2 = [valid_set_x2[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_key = [valid_set_key[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x1)
        train_set_x1 = [train_set_x1[i] for i in sorted_index]
        train_set_x2 = [train_set_x2[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_key = [train_set_key[i] for i in sorted_index]

    train = (train_set_x2, train_set_x1, train_set_y, train_set_key)
    valid = (valid_set_x2, valid_set_x1, valid_set_y, valid_set_key)
    #test = (test_set_x, test_set_y)
    return train, valid, emb

# def LoadData2(path="rnn_enc/data/imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
#               sort_by_len=False):
#     np.random.seed(777)
#     #############
#     # LOAD DATA #
#     #############
# 
#     # Load the dataset
#     print path
#     f = open(path, 'rb')
#     train_set = cPickle.load(f)
#     f.close()
# 
#     emb = train_set[5].astype(np.float32) if not (train_set[5] is None) else train_set[5]
#     keys = train_set[7]
#     cv = train_set[8]
#     train_set_x1 = train_set[0]
#     train_set_x2 = train_set[1]
#     train_set_x3 = train_set[2]
#     train_set_y = train_set[3] 
#     train_set_key = train_set[7]
#     n_samples = len(train_set_x1)
#     # fix the random seed
#     np.random.seed(42)
# 
#     sidx = np.nonzero(cv)[0]
#     cvsidx = np.where(np.array(cv)==0)[0]
#     valid_set_x1 = [train_set_x1[s] for s in cvsidx]
#     valid_set_x2 = [train_set_x2[s] for s in cvsidx]
#     valid_set_x3 = [train_set_x3[s] for s in cvsidx]
#     valid_set_y = [train_set_y[s] for s in cvsidx]
#     valid_set_key = [train_set_key[s] for s in cvsidx]
#     train_set_x1 = [train_set_x1[s] for s in sidx]
#     train_set_x2 = [train_set_x2[s] for s in sidx]
#     train_set_x3 = [train_set_x3[s] for s in sidx]
#     train_set_y = [train_set_y[s] for s in sidx]
#     train_set_key = [train_set_key[s] for s in sidx]
# 
# 
#     def remove_unk(x):
#         return [[1 if w >= n_words else w for w in sen] for sen in x]
#     def chop_utt(set1, set2, sety, setkey, maxlen):
#         new_set1 = []
#         new_set2 = []
#         new_set3 = []
#         new_sety = []
#         new_setkey = []
# 
#         for x1, x2, y, key in zip(set1, set2, sety, setkey):
#             if len(x2) <= maxlen:
#                 new_set1.append(x1)
#                 new_set2.append(x2)
#                 new_set3.append(x3)
#                 new_sety.append(y)
#                 new_setkey.append(key)
#             else:
#                 while len(x3) > maxlen:
#                     new_set1.append(x1)
#                     new_set2.append(x2[:maxlen])
#                     new_set3.append(x3[:maxlen])
#                     new_sety.append(y)
#                     new_setkey.append(key)
#                     x3 = x3[maxlen+1:]
#         return new_set1, new_set2, new_set3, new_sety, new_setkey
#     #test_set_x, test_set_y = test_set
#     #valid_set_x1, valid_set_x2, valid_set_x3, valid_set_y, valid_set_key = valid_set
#     #train_set_x1, train_set_x2, train_set_x3, train_set_y, train_set_key = chop_utt(train_set[0], train_set[1], train_set[2], train_set[3], maxlen)
#     
#     #train_set_x1 = remove_unk(train_set_x1)
#     #train_set_x2 = remove_unk(train_set_x2)
#     #valid_set_x1 = remove_unk(valid_set_x1)
#     #valid_set_x2 = remove_unk(valid_set_x2)
#     #test_set_x = remove_unk(test_set_x)
#     n_samples = len(train_set_x1)
#     sidx = np.random.permutation(n_samples)
#     train_set_x1 = [train_set_x1[s] for s in sidx[:n_samples]]
#     train_set_x2 = [train_set_x2[s] for s in sidx[:n_samples]]
#     train_set_x3 = [train_set_x3[s] for s in sidx[:n_samples]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_samples]]
#     train_set_key = [train_set_key[s] for s in sidx[:n_samples]]
# 
# 
#     def len_argsort(seq):
#         return sorted(range(len(seq)), key=lambda x: len(seq[x]))
# 
#     if sort_by_len:
#         #sorted_index = len_argsort(test_set_x)
#         #test_set_x = [test_set_x[i] for i in sorted_index]
#         #test_set_y = [test_set_y[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(valid_set_x1)
#         valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
#         valid_set_x2 = [valid_set_x2[i] for i in sorted_index]
#         valid_set_x3 = [valid_set_x3[i] for i in sorted_index]
#         valid_set_y = [valid_set_y[i] for i in sorted_index]
#         valid_set_key = [valid_set_key[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(train_set_x1)
#         train_set_x1 = [train_set_x1[i] for i in sorted_index]
#         train_set_x2 = [train_set_x2[i] for i in sorted_index]
#         train_set_x3 = [train_set_x3[i] for i in sorted_index]
#         train_set_y = [train_set_y[i] for i in sorted_index]
#         train_set_key = [train_set_key[i] for i in sorted_index]
# 
#     train = (train_set_x1, train_set_x2, train_set_x3, train_set_y, train_set_key)
#     valid = (valid_set_x1, valid_set_x2, valid_set_x3, valid_set_y, valid_set_key)
#     #test = (test_set_x, test_set_y)
# 
#     return train, valid, emb#, test
# def LoadData_Merge(path="rnn_enc/data/imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
#               sort_by_len=False):
#     '''Loads the dataset
# 
#     :type path: String
#     :param path: The path to the dataset (here IMDB)
#     :type n_words: int
#     :param n_words: The number of word to keep in the vocabulary.
#         All extra words are set to unknow (1).
#     :type valid_portion: float
#     :param valid_portion: The proportion of the full train set used for
#         the validation set.
#     :type maxlen: None or positive int
#     :param maxlen: the max sequence length we use in the train/valid set.
#     :type sort_by_len: bool
#     :name sort_by_len: Sort by the sequence lenght for the train,
#         valid and test set. This allow faster execution as it cause
#         less padding per minibatch. Another mechanism must be used to
#         shuffle the train set at each epoch.
# 
#     '''
# 
#     #############
#     # LOAD DATA #
#     #############
# 
#     # Load the dataset
#     print path
#     f = open(path, 'rb')
#     train_set = cPickle.load(f)
#     #test_set = cPickle.load(f)
#     f.close()
#     emb = train_set[4].astype(np.float32) if not (train_set[4] is None) else train_set[4]
#     keys = train_set[6]
#     cv = train_set[7]
#     
#     new_train_set_x1 = []
#     new_train_set_x2 = []
#     new_train_set_y = []
#     new_train_set_key = []
#     for x1, x2, y, key in zip(train_set[0], train_set[1], train_set[2], train_set[6]):
#         new_train_set_x1.append(x1)
#         new_train_set_x2.append(x2)
#         new_train_set_y.append(y)
#         new_train_set_key.append(train_set[6][key])
#     train_set = (new_train_set_x1, new_train_set_x2, new_train_set_y, new_train_set_key)
#     del new_train_set_x1, new_train_set_x2, new_train_set_y, new_train_set_key
#     # split training set into validation set
#     train_set_x1, train_set_x2, train_set_y, train_set_key = train_set
#     n_samples = len(train_set_x1)
#     # fix the random seed
#     np.random.seed(42)
#     sidx = np.where(np.array(train_set_y)==1)[0]
#     for s in sidx:
#         train_set_y[s] = train_set_y[s]+1
# 
#     sidx = np.nonzero(cv)[0]
#     cvsidx = np.where(np.array(cv)==0)[0]
#     valid_set_x1 = [train_set_x1[s] for s in cvsidx]
#     valid_set_x2 = [train_set_x2[s] for s in cvsidx]
#     valid_set_y = [train_set_y[s] for s in cvsidx]
#     valid_set_key = [train_set_key[s] for s in cvsidx]
#     train_set_x1 = [train_set_x1[s] for s in sidx]
#     train_set_x2 = [train_set_x2[s] for s in sidx]
#     train_set_y = [train_set_y[s] for s in sidx]
#     train_set_key = [train_set_key[s] for s in sidx]
# 
#     train_set = (train_set_x1, train_set_x2, train_set_y, train_set_key)
#     valid_set = (valid_set_x1, valid_set_x2, valid_set_y, valid_set_key)
# 
#     def remove_unk(x):
#         return [[1 if w >= n_words else w for w in sen] for sen in x]
#     def chop_utt(set1, set2, sety, setkey, maxlen):
#         new_set1 = []
#         new_set2 = []
#         new_sety = []
#         new_setkey = []
# 
#         for x1, x2, y, key in zip(set1, set2, sety, setkey):
#             if len(x2) <= maxlen:
#                 new_set1.append(x1)
#                 new_set2.append(x2)
#                 new_sety.append(y)
#                 new_setkey.append(key)
#             else:
#                 while len(x2) > maxlen:
#                     new_set1.append(x1)
#                     new_set2.append(x2[:maxlen])
#                     new_sety.append(y)
#                     new_setkey.append(key)
#                     x2 = x2[maxlen+1:]
#         return new_set1, new_set2, new_sety, new_setkey
#     #test_set_x, test_set_y = test_set
#     #valid_set_x1, valid_set_x2, valid_set_y, valid_set_key = valid_set
#     #train_set_x1, train_set_x2, train_set_y, train_set_key = chop_utt(train_set[0], train_set[1], train_set[2], train_set[3], maxlen)
#     
#     #train_set_x1 = remove_unk(train_set_x1)
#     #train_set_x2 = remove_unk(train_set_x2)
#     #valid_set_x1 = remove_unk(valid_set_x1)
#     #valid_set_x2 = remove_unk(valid_set_x2)
#     #test_set_x = remove_unk(test_set_x)
#     n_samples = len(train_set_x1)
#     sidx = np.random.permutation(n_samples)
#     train_set_x1 = [train_set_x1[s] for s in sidx[:n_samples]]
#     train_set_x2 = [train_set_x2[s] for s in sidx[:n_samples]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_samples]]
#     train_set_key = [train_set_key[s] for s in sidx[:n_samples]]
# 
# 
#     def len_argsort(seq):
#         return sorted(range(len(seq)), key=lambda x: len(seq[x]))
# 
#     if sort_by_len:
#         #sorted_index = len_argsort(test_set_x)
#         #test_set_x = [test_set_x[i] for i in sorted_index]
#         #test_set_y = [test_set_y[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(valid_set_x1)
#         valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
#         valid_set_x2 = [valid_set_x2[i] for i in sorted_index]
#         valid_set_y = [valid_set_y[i] for i in sorted_index]
#         valid_set_key = [valid_set_key[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(train_set_x1)
#         train_set_x1 = [train_set_x1[i] for i in sorted_index]
#         train_set_x2 = [train_set_x2[i] for i in sorted_index]
#         train_set_y = [train_set_y[i] for i in sorted_index]
#         train_set_key = [train_set_key[i] for i in sorted_index]
# 
#     train = (train_set_x2, train_set_x1, train_set_y, train_set_key)
#     valid = (valid_set_x2, valid_set_x1, valid_set_y, valid_set_key)
#     #test = (test_set_x, test_set_y)
# 
#     return train, valid, emb#, test
# 
# 
# def LoadData_balance(path="rnn_enc/data/imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None, sort_by_len=False, repeat=1):
#     #############
#     # LOAD DATA #
#     #############
# 
#     # Load the dataset
#     print path
#     f = open(path, 'rb')
#     train_set = cPickle.load(f)
#     f.close()
# 
#     cv = train_set[7]
#     emb = train_set[4].astype(np.float32) if not (train_set[4] is None) else train_set[4]
#     train_set_x1 = train_set[0]
#     train_set_x2 = train_set[1]
#     train_set_y = train_set[2]
#     train_set_key = train_set[6]
#     
#     sposidx = np.where(np.array(train_set_y)>1)[0]
#     postrain_set_x1 = [train_set_x1[s] for s in sposidx]
#     postrain_set_x2 = [train_set_x2[s] for s in sposidx]
#     postrain_set_y = [train_set_y[s] for s in sposidx]
#     postrain_set_cv = [cv[s] for s in sposidx]
#     postrain_set_key = [train_set_key[s] for s in sposidx]
#     snegidx = np.where(np.array(train_set_y)==0)[0]
#     negtrain_set_x1 = [train_set_x1[s] for s in snegidx]
#     negtrain_set_x2 = [train_set_x2[s] for s in snegidx]
#     negtrain_set_y = [train_set_y[s] for s in snegidx]
#     negtrain_set_cv = [cv[s] for s in snegidx]
#     #negtrain_set_key = [train_set_key[s] for s in snegidx]
# 
#     while (repeat>0):
#         neg_samples = len(negtrain_set_x1)
#         print neg_samples
#         sidx = np.random.permutation(neg_samples)
#         print sidx
#         len(sidx)
#         train_set_x1 = [negtrain_set_x1[s] for s in sidx[:len(postrain_set_x1)]]
#         train_set_x2 = [negtrain_set_x2[s] for s in sidx[:len(postrain_set_x1)]]
#         train_set_y = [negtrain_set_y[s] for s in sidx[:len(postrain_set_x1)]]
#         cv = [negtrain_set_cv[s] for s in sidx[:len(postrain_set_x1)]]
# 
#         train_set_x1 = train_set_x1 + postrain_set_x1
#         train_set_x2 = train_set_x2 + postrain_set_x2
#         train_set_y = train_set_y + postrain_set_y
#         cv = cv + postrain_set_cv
#         #for val in postrain_set_key:
#         #    intkey = len(train_set_key)
#         #    train_set_key[intkey]=val
#         repeat = repeat-1
# 
#     n_samples = len(train_set_x1)
#     print n_samples
#     # fix the random seed
#     np.random.seed(42)
# 
#     sidx = np.nonzero(cv)[0]
#     cvsidx = np.where(np.array(cv)==0)[0]
#     valid_set_x1 = [train_set_x1[s] for s in cvsidx]
#     valid_set_x2 = [train_set_x2[s] for s in cvsidx]
#     valid_set_y = [train_set_y[s] for s in cvsidx]
#     valid_set_key = [train_set_key[s] for s in cvsidx]
#     train_set_x1 = [train_set_x1[s] for s in sidx]
#     train_set_x2 = [train_set_x2[s] for s in sidx]
#     train_set_y = [train_set_y[s] for s in sidx]
#     train_set_key = [train_set_key[s] for s in sidx]
# 
# 
#     def remove_unk(x):
#         return [[1 if w >= n_words else w for w in sen] for sen in x]
#     def chop_utt(set1, set2, sety, setkey, maxlen):
#         new_set1 = []
#         new_set2 = []
#         new_sety = []
#         new_setkey = []
# 
#         for x1, x2, y, key in zip(set1, set2, sety, setkey):
#             if len(x2) <= maxlen:
#                 new_set1.append(x1)
#                 new_set2.append(x2)
#                 new_sety.append(y)
#                 new_setkey.append(key)
#             else:
#                 while len(x2) > maxlen:
#                     new_set1.append(x1)
#                     new_set2.append(x2[:maxlen])
#                     new_sety.append(y)
#                     new_setkey.append(key)
#                     x2 = x2[maxlen+1:]
#         return new_set1, new_set2, new_sety, new_setkey
#     #test_set_x, test_set_y = test_set
#     #valid_set_x1, valid_set_x2, valid_set_x3, valid_set_y, valid_set_key = valid_set
#     #train_set_x1, train_set_x2, train_set_x3, train_set_y, train_set_key = chop_utt(train_set[0], train_set[1], train_set[2], train_set[3], maxlen)
#     
#     #train_set_x1 = remove_unk(train_set_x1)
#     #train_set_x2 = remove_unk(train_set_x2)
#     #valid_set_x1 = remove_unk(valid_set_x1)
#     #valid_set_x2 = remove_unk(valid_set_x2)
#     #test_set_x = remove_unk(test_set_x)
#     n_samples = len(train_set_x1)
#     sidx = np.random.permutation(n_samples)
#     train_set_x1 = [train_set_x1[s] for s in sidx[:n_samples]]
#     train_set_x2 = [train_set_x2[s] for s in sidx[:n_samples]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_samples]]
#     train_set_key = [train_set_key[s] for s in sidx[:n_samples]]
# 
# 
#     def len_argsort(seq):
#         return sorted(range(len(seq)), key=lambda x: len(seq[x]))
# 
#     if sort_by_len:
#         #sorted_index = len_argsort(test_set_x)
#         #test_set_x = [test_set_x[i] for i in sorted_index]
#         #test_set_y = [test_set_y[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(valid_set_x1)
#         valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
#         valid_set_x2 = [valid_set_x2[i] for i in sorted_index]
#         valid_set_y = [valid_set_y[i] for i in sorted_index]
#         valid_set_key = [valid_set_key[i] for i in sorted_index]
# 
#         sorted_index = len_argsort(train_set_x1)
#         train_set_x1 = [train_set_x1[i] for i in sorted_index]
#         train_set_x2 = [train_set_x2[i] for i in sorted_index]
#         train_set_y = [train_set_y[i] for i in sorted_index]
#         train_set_key = [train_set_key[i] for i in sorted_index]
# 
#     train = (train_set_x2, train_set_x1, train_set_y, train_set_key)
#     valid = (valid_set_x2, valid_set_x1, valid_set_y, valid_set_key)
#     #test = (test_set_x, test_set_y)
# 
#     return train, valid, emb#, test

def ExtAugFeat(data, tag2rank, isOnehot=True):
    if isOnehot:
        rankFeat = np.zeros((len(data[0]), max(tag2rank.values())), dtype=int)
        for dIdx, rawTag in enumerate(data[3]):
            tag = rawTag.split('\t')[1]
            rankFeat[dIdx, tag2rank[tag] - 1] = 1
    else:
        rankFeat = np.zeros((len(data[0]), 1), dtype=theano.config.floatX)
        for dIdx, rawTag in enumerate(data[3]):
            tag = rawTag.split('\t')[1]
            rankFeat[dIdx, 0] = 1. / tag2rank[tag]

    return rankFeat

# def ExtMitraFeat(data, tag2feat):
#     mitraFeat = np.zeros((len(data[0]), tag2feat.values()[0].shape[0]), dtype=theano.config.floatX)
#     for dIdx, rawTag in enumerate(data[3]):
#         mitraFeat[dIdx, :] = tag2feat[rawTag]
#     return mitraFeat

def GenTag2RankData(dataList, tag2rankData):
    tag2rank = dict()
    for path in dataList:
        data = cPickle.load(open(path, 'rb'))
        keys = data[6].values()
        curTag1 = ''
        for key in keys:
            tag1, tag2 = key.split('\t')
            if tag1 != curTag1:
                curTag1 = tag1
                rank = 1
            tag2rank[tag2] = rank
            rank = rank + 1
    cPickle.dump(tag2rank, open(tag2rankData, 'wb'))

# def SaveRepr(test, testRepr, path):
#     results = []
#     for i,j in enumerate(test[3]):
#         results.append([j, ' '.join([str(f) for f in testRepr[i,:]])])
# 
#     def getKey1(item):
#         [org, Rel] = item[0].split('\t')
#         org = org.split('_R')[0]
#         return org
# 
#     def getKey2(item):
#         [org, Rel] = item[0].split('\t')
#         [_, num] = Rel.split('_R')
#         num = num.split('_C')[0] 
#         return num
#     def getKey3(item):
#         [org, Rel] = item[0].split('\t')
#         [_, num] = Rel.split('_R')
#         [_, num] = num.split('_C') 
#         return num
# 
#     def cmp(x,y):
#         if getKey1(x) == getKey1(y):
#             if int(getKey2(x)) == int(getKey2(y)):
#                 if int(getKey3(x)) < int(getKey3(y)):
#                     return -1
#                 else:
#                     return 1
#             elif int(getKey2(x)) < int(getKey2(y)):
#                 return -1
#             else:
#                 return 1 
#         else:
#             if getKey1(x) < getKey1(y):
#                 return -1
#             else:
#                 return 1
#     results = ['\t'.join(i) for i in sorted(results,cmp=cmp)]
#     f = open(path, "w")
#     f.write('\n'.join(results))
