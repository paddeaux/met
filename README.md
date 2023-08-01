# CRT First Year Placement 2023 - Met Éireann
## Synthetic Generation of Satellite Imagery

The architecture described here originally comes from this paper: **https://arxiv.org/pdf/1710.10196.pdf**, while the PyTorch implementation of a similar ProGAN archictecture comes from here: **https://www.kaggle.com/code/paddeaux/pggan-progressive-growing-gan-pggan-pytorch/edit**
The use of a ProGAN for synthetic satellite imagery is also described by Abady et. al (https://www.researchgate.net/publication/346821973_GAN_generation_of_synthetic_multispectral_satellite_images), and the initial aim of this work is to reproduce this GAN, in this case using PyTorch in favour of TensorFlow.

A means of expanding on this original work is to bring seasonality into the network - allowing the generation of synthetic images representative of seasonality

### Datasets & Checkpoints
Datasets and model checkpoints should be located in the parent folder of `met`, as **"Input"** and **"checkpoints"** respectively.

* The SEN12MS dataset is available here: **https://mediatum.ub.tum.de/1474000**
* Checkpoints for an attempted partial train:
   * Generator checkpoint trained on the Spring portion of the SEN12MS dataset (partially trained): **https://maynoothuniversity-my.sharepoint.com/:u:/g/personal/patrick_gorry_2015_mumail_ie/ERhn7hIiokdJkp0mJ_hgvEMByG-4SrzeFcvE5ZKSjwfEgg?e=yFWuef&download=1**
   * Critic checkpoint for the above: **https://maynoothuniversity-my.sharepoint.com/:u:/g/personal/patrick_gorry_2015_mumail_ie/EVzkXxTOevFPh9-73e8LCDoBHwBfbA7pFe6QF5rCVg88tg?e=sG9Ipy&download=1**
* Checkpoints for an overfitted train:
   * Generator: **https://maynoothuniversity-my.sharepoint.com/:u:/r/personal/patrick_gorry_2015_mumail_ie/Documents/CRT/Met%20%C3%89ireann/met_resources/checkpoints/generator_sen12_full_overfit.pth?csf=1&web=1&e=gCfxlJ**
   * Critic: **https://maynoothuniversity-my.sharepoint.com/:u:/r/personal/patrick_gorry_2015_mumail_ie/Documents/CRT/Met%20%C3%89ireann/met_resources/checkpoints/generator_sen12_full_overfit.pth?csf=1&web=1&e=gCfxlJ**


### Running the tool
* Training a new model or loading the last checkpoints is done using the `--mode/-m` flag:
 `python progan.py --mode train` or `python progan.py -m load`
* Example plots (either in RGB or full band) can be produced from the trained/loaded model with the `--plot/-p` flag:
`python progan.py -m load -p rgb` or `python progan.py -m load -p full` 
 

