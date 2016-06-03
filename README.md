Attention-Based Long Short-Term Memory Enocder for Community Question Answering
=======
## Data Formatting

### trainData/testData

Data is stored using cPickle package, and formatted as follows:

* `data[0]`: list of related questions/comments, each question/comment is represented as a list of index
* `data[1]`: list of original questions, each question is represented as a list of index
* `data[2]`: list of labels
* `data[3]`: None (deprecated)
* `data[4]`: pretrained word embedding, represented as a vocabSize-by-wordVecDim numpy matrix
* `data[5]`: None (deprecated)
* `data[6]`: list of tags (e.g. 'Q1\tQ1\_R1')
* `data[7]`: list of indictor, where 0 refers to valid set; 1 refers to train set

### tag2rank
Tag2rank is a dictionary mapping tags to ranks, and is stored using cPickle package.

Example: `{'Q1\tQ1\_R1': 1, 'Q1\tQ1\_R4': 2, 'Q2\tQ2\_R7': 1, 'Q2\tQ2\_R10': 2}`

## How to run

### Training
These scripts take training set and test set as input, and generate ALSTM encoder models.

`--saveto` specifies where to save model; `--loadfrom` loads existing model

#### Sequential LSTM Encoder with Attention
`python rnn_enc/scripts/attention_trainer.py [--saveto <saveto>] [--loadfrom <loadfrom>] [--max_epoch <max_epoch>] [--swap | --no-swap] <trainData> <testData>`

#### Sequential LSTM Encoder with Attention + Rank Feature
`python rnn_enc/scripts/attention_augFeat_trainer.py [--saveto <saveto>] [--loadfrom <loadfrom>] [--max_epoch <max_epoch>] [--swap | --no-swap] <trainData> <testData> <tag2rank>`


### Testing
These scripts load trained models and predict relevance scores of testData.

`--outFile` specifies where to save prediction results

#### Sequential LSTM Encoder with Attention
`python rnn_enc/scripts/attention_tester.py [--outFile <outFile>] [--swap | --no-swap] <loadfrom> <testData>`

#### Sequential LSTM Encoder with Attention + Rank Feature
`python rnn_enc/scripts/attention_augFeat_tester.py [--outFile <outFile>] [--swap | --no-swap] <loadfrom> <testData> <tag2rank>`


### Retrieve Attention Weights
These scripts load trained models and output attention weights of sentences fed to first LSTM encoder in testData.
Dict helps convert index to word. Sentences in word and weights of sentences fed to first LSTM encoder will stores in dumpDir.

#### Sequential LSTM Encoder with Attention
`python rnn_enc/scripts/attention_extWeight.py <loadfrom> <testData> <dict> <dumpDir>`

#### Sequential LSTM Encoder with Attention + Rank Feature
`python rnn_enc/scripts/attention_augFeat_extWeight.py <loadfrom> <testData> <tag2rank> <dict> <dumpDir>`
