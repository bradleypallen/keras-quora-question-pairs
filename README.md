# keras-quora-question-pairs

A Keras model that addresses the Quora Question Pairs
[[1]](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
dyadic prediction task.

## Model implementation

The Keras model architecture is shown below:

![[Keras model architecture for Quora Question Pairs dyadic prediction]](quora-q-pairs-model.png)

The model architecture is based on the Stanford Natural Language
Inference [[2]](http://nlp.stanford.edu/pubs/snli_paper.pdf) benchmark
model developed by Stephen Merity
[[3]](https://github.com/Smerity/keras_snli), specifically the version
using a simple summation of GloVe word embeddings
[[4]](http://nlp.stanford.edu/pubs/glove.pdf) to represent each
question in the pair. A difference between this and the Merity SNLI
benchmark is that our final layer is Dense with sigmoid activation, as
opposed to softmax. Another key difference is that we are using the
max operator as opposed to sum to combine word embeddings into a
question representation. We use binary cross-entropy as a loss
function and Adam for optimization.

## Evaluation

We partition the Quora question pairs into a 90/10 train/test
split. We run training for 25 epochs with a further 90/10
train/validation split, saving the weights from the model checkpoint
with the maximum validation accuracy. Training takes approximately 120
secs/epoch, using Tensorflow as a backend for Keras on an Amazon Web
Services EC2 p2-xlarge GPU compute instance. We finally evaluate the
best checkpointed model to obtain a test set accuracy of
0.8291. The table below places this in the context of other work
on the dataset reported to date:

| Model | Source of Word Embeddings | Accuracy |
| --- | --- | --- |
| "BiMPM model" [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | **0.88** |
| "LSTM with concatenation" [[6]](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) | "Quora's text corpus" | 0.87 |
| "LSTM with distance and angle" [[6]](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) | "Quora's text corpus" | 0.87 |
| "Decomposable attention" [[6]](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) | "Quora's text corpus" | 0.86 |
| "L.D.C." [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | 0.86 |
| Max bag-of-embeddings (*this work*) | GloVe Common Crawl (840B tokens, 300D) | 0.83 |
| "Multi-Perspective-LSTM" [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | 0.83 |
| "Siamese-LSTM" [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | 0.83 |
| "Neural bag-of-words" (max) [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification) | GloVe Common Crawl pruned to 1M vocab. (spaCy default) | 0.83 |
| "Neural bag-of-words" (max & mean) [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification) | GloVe Common Crawl pruned to 1M vocab. (spaCy default) | 0.83 |
| "Max-out Window Encoding" with depth 2 [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification) | GloVe Common Crawl pruned to 1M vocab. (spaCy default) | 0.83 |
| "Neural bag-of-words" (mean) [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification) | GloVe Common Crawl pruned to 1M vocab. (spaCy default) | 0.81 |
| "Multi-Perspective-CNN" [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | 0.81 |
| "Siamese-CNN" [[5]](https://arxiv.org/pdf/1702.03814) | GloVe Common Crawl (840B tokens, 300D) | 0.80 |
| "Spacy + TD-IDF + Siamese" [[8]](http://www.erogol.com/duplicate-question-detection-deep-learning/) | GloVe (6B tokens, 300D) | 0.79 |


## Discussion

An initial pass at hyperparameter tuning by evaluating possible
settings a hyperparameter at a time led to the following observations:

* Computing the question representation by applying the max operator to the word embeddings slightly outperformed using mean and sum, which is consistent with what is reported in [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification).
* Computing the question representation using max also slightly outperformed the use of bidirectional LSTM and GRU recurrent layers, again as discussed in [[7]](https://explosion.ai/blog/quora-deep-text-pair-classification).
* Batch normalization improved accuracy, as observed by [[8]](http://www.erogol.com/duplicate-question-detection-deep-learning/).
* Any amount of dropout decreased accuracy, as also observed by [[8]](http://www.erogol.com/duplicate-question-detection-deep-learning/).
* Four hidden layers in the fully-connected component had the best accuracy, with between zero and six hidden layers evaluated.
* Using 200 dimensions for the layers in the fully-connected component showed the best accuracy among tested dimensions 50, 100, 200, and 300.

## Future work

A more principled (and computationally-intensive) campaign of
randomized search over the space of hyperparameter configurations is
planned.

## Requirements

* Python 3.5.2
* jupyter 4.2.1

## Package dependencies

* numpy 1.11.3
* pandas 0.19.2
* matplotlib 1.5.3
* Keras 1.2.1
* scikit-learn 0.18.1
* h5py 2.6.0
* hdf5 1.8.17

## Usage

This repository contains two different ways to create and run the model.

### From the command line

    $ python3 keras-quora-question-pairs.py

On first execution, this will download the required Quora and GloVe datasets and generate files that cache the training data and related word count and embedding data for subsequent runs.

### As Jupyter notebooks

Simply run the notebook server using the standard Jupyter command:

    $ jupyter notebook

First run 

    quora-question-pairs-data-prep.ipynb

As with the script above, this will generate files for training the Keras model. Then run

    quora-question-pairs-training.ipynb
    
next to train and evaluate the model.

## License

MIT. See the LICENSE file for the copyright notice.

## References

[[1]](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) Shankar Iyar, Nikhil Dandekar, and Kornél Csernai. “First Quora Dataset Release: Question Pairs,” 24 January 2016. Retrieved at https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs on 31 January 2017.

[[2]](http://nlp.stanford.edu/pubs/snli_paper.pdf)  Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. "A large annotated corpus for learning natural language inference," in Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015), September 2015.

[[3]](https://github.com/Smerity/keras_snli) Stephen Merity. "Keras SNLI baseline example,” 4 September 2016. Retrieved at https://github.com/Smerity/keras_snli on 31 January 2017.

[[4]](http://nlp.stanford.edu/pubs/glove.pdf) Jeffrey Pennington, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation," in Proceedings of the 2014 Conference on Empirical Methods In Natural Language Processing (EMNLP 2014), October 2014.

[[5]](https://arxiv.org/pdf/1702.03814) Zhiguo Wang, Wael Hamza and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences," 13 February 2017.  Retrieved at https://arxiv.org/pdf/1702.03814.pdf on 14 February 2017.

[[6]](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) Lili Jiang, Shuo Chang, and Nikhil Dandekar. "Semantic Question Matching with Deep Learning," 13 February 2017. Retrieved at https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning on 13 February 2017.

[[7]](https://explosion.ai/blog/quora-deep-text-pair-classification) Matthew Honnibal. "Deep text-pair classification with Quora's 2017 question dataset," 13 February 2017. Retreived at https://explosion.ai/blog/quora-deep-text-pair-classification on 13 February 2017.

[[8]](http://www.erogol.com/duplicate-question-detection-deep-learning/) Eren Golge. "Duplicate Question Detection with Deep Learning on Quora Dataset," 12 February 2017. Retreived at http://www.erogol.com/duplicate-question-detection-deep-learning/ on 13 February 2017.
