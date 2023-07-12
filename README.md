# CRT First Year Placement 2023 - Met Éireann
## Synthetic Generation of Satellite Imagery

The architecture described here originally comes from this paper: **https://arxiv.org/pdf/1710.10196.pdf**, while the PyTorch implementation of a similar ProGAN archictecture comes from here: **https://www.kaggle.com/code/paddeaux/pggan-progressive-growing-gan-pggan-pytorch/edit**
The use of a ProGAN for synthetic satellite imagery is also described by Abady et. al (https://www.researchgate.net/publication/346821973_GAN_generation_of_synthetic_multispectral_satellite_images), and the initial aim of this work is to reproduce this GAN, in this case using PyTorch in favour of TensorFlow.

A means of expanding on this original work is to bring seasonality into the network - allowing the generation of synthetic images representative of seasonality.

### Running the tool
Command line arguments will be added in due course, but for now to run the training simply run:
`python progan.py`

