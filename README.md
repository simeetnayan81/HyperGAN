HyperGAN
=============

This is a slightly modified implementation of the paper [HyperGAN: A Generative Model for Diverse, Performant Neural Networks](https://arxiv.org/pdf/1901.11058) by Ratzlaff et. al.

I propose a slight modification to the implementation by replacing the single LeNet target network with an ensemble of LeNet networks. The ensemble size can be set to 1 to align with the original paper.

To run the python script, just use python model.py.
Following are the args that may be used for tuning the model while training:

```--noise_batch_size``` : the batch size of random noise to input in the generator
<br>
```--image_batch_size``` : batch size of image data that will be passed through the model
<br>
```--lr```: learning rate
<br>
```--ensemble_size``` : Size of ensemble(N according to problem statement)
<br>
```--epochs``` : epochs
<br>
```--code_size``` : the dimension of intermediate latent code

Different components of the network:

* Mixer: A fully connected network, takes input noise, produces a latent code as output.
* Generator: Takes the latent code, produces the weights for the target network. It is also
fully connected as of now. I’ll work on replacing mixer and generator with transformer for
the next submission
* Discriminator: It takes latent code as the input, tries to make the codes as diverse as
possible so that the mixer does not end up collapsing to produce the same code each
time. This is important in order to maintain diversity among the weights produced for
different sets of noises.
* Target Network: In this case, it is a LeNet5 model for classifying 10 classes based on
cifar10 dataset.

Simple LeNet5 test accuracy on Cifar10: 0.55
Generator Based Ensemble of LeNet5 accuracy on test: Close to 0.6

The logs file contains the training logs.