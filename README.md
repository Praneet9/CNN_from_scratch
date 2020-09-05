## Implementation of ConvNet in Numpy

Started out as an implementation of CNN Layer and turned out to be a complete end to end deep learning architecture implementation. There might be many more improvements and corrections here. Do open an issue if you find anything. Would be great to get contributions on this. If you want to learn/understand some other implementations end to end you can use this repo as a starting point.

*As numpy is built to optimize single core operations, it was difficult to include a CNN example run file. If you know how to parallelize numpy operations let me know!*

#### Requirements

* numpy 1.18.x
* matplotlib 3.2.x
* scikit-learn 0.23.x

#### Implementation

I've tried to implement all of these from scratch, ofcourse with the help from all the freely available articles and codebases.

- [x] Convolution Layer
- [x] Dense Layer
- [x] Pooling Layer
- [x] Flatten Layer
- [x] Forward Function
- [x] Backpropagation Function
- [x] Sigmoid Activation Function
- [x] ReLU Activation Function
- [x] Binary Cross Entropy Loss Function
- [x] Adam Optimizer
- [x] Weights Initialization

There is a lot of scope to implement many more things in this repo. For example - `Global Average Pooling Layer`, `Categorical Cross Entropy Loss`, `Other Activation Functions`, etc. Would love to get pull requests for corrections/updates and new implementations.

#### Example

```sh
$ python fcn_model.py
```

#### References

* https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
* https://github.com/AyushExel/Neo/blob/master/src/nn.py
* http://neuralnetworksanddeeplearning.com/chap2.html
* https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
* https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4
* http://www.songho.ca/dsp/convolution/convolution2d_example.html
* https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
* https://victorzhou.com/blog/intro-to-cnns-part-2/
* https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
* https://quantdare.com/create-your-own-deep-learning-framework-using-numpy/
* https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
* https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
* https://medium.com/binaryandmore/beginners-guide-to-deriving-and-implementing-backpropagation-e3c1a5a1e536
* https://gluon.mxnet.io/chapter06_optimization/adam-scratch.html
