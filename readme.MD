# GAN

Pytorch GAN implementation done to learn all the caveats behind this model

## Notes:

- Hard to train because there is not a clear metric that tells when the
  train is going well. Only qualitative samples seen by human can tell 
  the goodness of the generator
- I can see the potentiality of GAN: adversarial training can be used as an engine
  to accomplish hard tasks like image generation with a simple decoder
- For me, normalization layer before activation stapled the image generation process
- Following various articles I:
  - Replaced ReLU with LeakyReLU to improve gradient flowing
  - Use SGD with generator and Adam with discriminator
    - Apply Adam to both generator and discriminator made prevail the discriminator loss during the training

# How to use

    git clone https://github.com/Michedev/Simple-GAN
    cd Simple-GAN
    poetry install
    poetry run python gan/train.py