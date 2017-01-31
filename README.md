# keras-quora-question-pairs

A Keras model that addresses the Quora Question Pairs [[1]](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) classification task.

## Model implementation

The model architecture is based on the Stanford Natural Language Inference [[2]](http://nlp.stanford.edu/pubs/snli_paper.pdf) benchmark model developed by Stephen Merity [[3]](https://github.com/Smerity/keras_snli), specifically the version using a simple summation of GloVe word embeddings [[4]](http://nlp.stanford.edu/pubs/glove.pdf) to represent each question in the pair. A difference between this and the Merity SNLI benchmark is that our final layer is Dense with sigmoid activation, as opposed to softmax. We use binary cross-entropy as a loss function and Adam for optimization. 

## Evaluation

We partition the Quora question pairs into a 90/10 train/test split. We run training for 25 epochs with a further 90/10 train/validation split, saving the weights from the model checkpoint with the maximum validation accuracy. Training takes approximately 150 secs/epoch, using Tensorflow as a backend for Keras on an Amazon Web Services EC2 p2-xlarge GPU compute instance. We finally evaluate the best checkpointed model to obtain a test set accuracy of 0.8181.

## Discussion

Much work remains, specifically:

* Tuning hyperparameters
* As in [[3]](https://github.com/Smerity/keras_snli), evaluating variant architectures that use recurrent layers in place of the summation of embeddings

## Requirements

* Python 3.5.2
* jupyter 4.2.0

## Package dependencies

* numpy 1.11.1
* pandas 0.19.2
* matplotlib 1.5.3
* Keras 1.2.1
* scikit-learn 0.17.1
* h5py 2.6.0
* hdf5 1.8.17

## Usage

This repository contains two different ways to create and run the model.

### From the command line

    $ python3 keras-question-pairs.py

On first execution, this will download the required Quora and GloVe datasets and generate files that cache the training data and related word count and embedding data for subsequent runs.

### As Jupyter notebooks

Simply run the notebook server using the standard Jupyter command:

    $ jupyter notebook

First run quora-question-pairs-data-prep.ipynb; as with the script above, this will generate files for training the Keras model. Then run quora-question-pairs-training next to train and evaluate the model.

## License

MIT. See the LICENSE file for the copyright notice.

## References

[[1]](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) Shankar Iyar, Nikhil Dandekar, and Kornél Csernai. “First Quora Dataset Release: Question Pairs,” 24 January 2016. Retrieved at https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs on 31 January 2017.

[[2]](http://nlp.stanford.edu/pubs/snli_paper.pdf)  Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. "A large annotated corpus for learning natural language inference," in Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015), September 2015.

[[3]](https://github.com/Smerity/keras_snli) Stephen Merity. "Keras SNLI baseline example,” 4 September 2016. Retrieved at https://github.com/Smerity/keras_snli on 31 January 2017.

[[4]](http://nlp.stanford.edu/pubs/glove.pdf) Jeffrey Pennington, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation," in Proceedings of the 2014 Conference on Empirical Methods In Natural Language Processing (EMNLP 2014), October 2014.

## License

MIT. See the LICENSE file for the copyright notice.
