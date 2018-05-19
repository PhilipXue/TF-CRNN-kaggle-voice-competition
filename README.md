# CRNN model for keyword spotting
This is the implementation of some methods I applied in the [Tensorflow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) held by Google Brain on Kaggle.com. 
## Model Structure
The goal of the challenge is to spot some simple key word occuring in voice record,  which is well stated by the `Keyword spotting` problem in voice / speech recognition area.
Inspired by [This paper](https://arxiv.org/abs/1703.05390) and some other [articles](http://machine-listening.eecs.qmul.ac.uk/wp-content/uploads/sites/26/2017/01/cakir.pdf) by an attendent of [BirdCLEF 2016
](http://www.imageclef.org/lifeclef/2016/bird), I transformed audio sources into mel-spectrum images, construct a network with CNN top extracting features and RNN tail processing time relation. (Though pure CNN also works as well as CRNN method). I use InceptionNet and Resnet like network to construct the CNN top, and GRU for recurrent part.
The implementation here end up 0.87 on private board for single model without ensemble. The best ensemble record I had was 0.89 on private with a bunch of networks with similar architechture.
## prerequisite
```
python>=3.5
tensorflow>=1.8 (for some new features added when doning refactor)
keras>=2.1
librosa>=0.5.1
opencv
```
## Usage
### Data precess
I used librosa to do the audio-spetrum transformation and opencv to handle image data.
Modifyt the data root in `config.py` to your position that holds the data and run `data_preprocessor.py`
### Training with keras
### Training with tensorflow