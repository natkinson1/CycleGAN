# CycleGAN

Tensorflow implentation of a CycleGAN.

In this repository we train 2 generators:
- A generator that turns real images into the style of a painting.
- A generator that turns paintings into the style of real images.

Firstly, we define two models, the generator and the discriminator which will be used for the 2 GAN's in the CycleGAN.

Once you have trained the network in the Jupyter Notebook, one can then use the script to apply the model to any image of choice.
