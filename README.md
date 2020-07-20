# Sentence Embedding for Korean using translated NLI data
Unofficial implementation of [Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf) for korean sentence embedding

## Model
The encoder for embedding two sentences during NLI task is Korean Distilled BERT from [DistilKoBERT](https://github.com/monologg/DistilKoBERT). The embedding for each sentence is concatenated along with their absolute mean difference which then enters to 3-way classifier (entail, neutral, contradict)

## Data
I have used Korean NLI dataset, which was released from Kakao Brain

### Train
First, run `preprocess.py` to tokenize, convert into indicies and save into `.pt` files
Then, run `train.py` for training


### Requirements
torch == 1.1.0
transformers == 2.3.0
gluonnlp == 0.8.1
tensorflow == 2.0.0 (for `tensorflow.keras.preprocessing.sequence` only)

### Result
After around 20 epochs, the accuracy for test dataset was 73.7%

## Acknowledgments

* [monologg](https://github.com/monologg)