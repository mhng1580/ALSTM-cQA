Long Short-Term Memory Enocder with Attention for Community Question Answering
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

### tag2rank
Tag2rank is a dictionary mapping tags to ranks, and is stored using cPickle package.

Example: 

`{'Q1\tQ1\_R1': 1, 'Q1\tQ1\_R4': 2, 'Q2\tQ2\_R7': 1, 'Q2\tQ2\_R10': 2}`

## How to run

### Training

#### Sequential LSTM Encoder with Attention
`python rnn_enc/model2/attention.py [--saveto <saveto>] [--loadfrom <loadfrom>] [--max_epoch <max_epoch>] [--swap | --no-swap] <trainData> <testData>`

#### Sequential LSTM Encoder with Attention + Rank Feature
`python rnn_enc/model/attention_augFeat.py [--saveto <saveto>] [--loadfrom <loadfrom>] [--max_epoch <max_epoch>] [--swap | --no-swap] <trainData> <testData> <tag2rank>`
