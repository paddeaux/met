# The original ProGAN archictecutre comes from here: https://www.kaggle.com/code/paddeaux/pggan-progressive-growing-gan-pggan-pytorch/edit

# The Conditional GAN archictecture I am attempting to adapt is from here: https://www.kaggle.com/code/arturlacerda/pytorch-conditional-gan

# Base packages
import numpy as np
from math import log2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, getopt
import argparse
# Torch packages
import torch

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

# ProGAN libraries
from src.layers import *
from src.loaders import SEN12MS_RGB, SEN12MS_FULL, overfit, overfit_sen12
from src.utils import *

# SEN12MS dataloader
from src.sen12ms_dataLoader import *

torch.backends.cudnn.benchmarks = True
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_TRAIN_IMG_SIZE = 4
DATASET = os.path.join(os.path.dirname(os.getcwd()), "Input/ROIs1158_spring/train/")
OVERFIT_DATASET = os.path.join(os.path.dirname(os.getcwd()), "Input/sen12.tif")
OVERFIT_DATASET = "/data/pgorry/inputs/sen12.tif"

#os.makedirs("checkpoints", exist_ok = True)
CHECKPOINT_GEN = os.path.join(os.path.dirname(os.getcwd()),"checkpoints/gen_sen12_overfit_c.pth")
CHECKPOINT_CRITIC = os.path.join(os.path.dirname(os.getcwd()),"checkpoints/crit_sen12_overfit_c.pth")
SAVE_MODEL = True
LOAD_MODEL = False

LR = 1e-3
BATCH_SIZES = [256, 256, 128, 64, 32, 16, 8] #[256,256,128,64,32,16,8]  ## modifiable/ Batch_sizes for each step
IMAGE_SIZE = 128 ## 1024 for paper
IMG_CHANNELS = 13
Z_DIM = 256 ## 512 for paper
IN_CHANNELS = 256 ## 512 for paper
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/4)) + 1

PROGRESSIVE_EPOCHS = [1] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8,Z_DIM,1,1).to(DEVICE)
# NUM_WORKERS = 4
NUM_WORKERS = 2
GENERATE_EXAMPLES_AT = [1,4,8,12,16,20,24,28,32]#[1,50,100,150,200,250,300,350,400,450,500]##[1000,2000,3000,4000,5000,6000,7000,8000]

def get_loader(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],[0.5 for _ in range(IMG_CHANNELS)])
    ])
        
    transform_sen = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size),antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],[0.5 for _ in range(IMG_CHANNELS)])
    ])
    
    batch_size = BATCH_SIZES[int(log2(img_size/4))]
    # SEN12MS Dataset
    #dataset = SEN12MS_RGB(targ_dir=DATASET, transform=transform_sen)
    #dataset = SEN12MS_FULL(targ_dir=DATASET, transform=transform_sen)
    
    # Overfit of a single SEN12MS Image
    dataset = overfit_sen12(image_path=OVERFIT_DATASET, transform=transform_sen)

    # Loading from original CelebA dataset
    #dataset = datasets.ImageFolder(root=DATASET,transform=transform)

    # Loading from single image overfit
    #dataset = overfit(image_path=OVERFIT_DATASET, length=500, transform=transform)

    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    
    return loader,dataset

def train_fn(gen,critic,loader,dataset,step,alpha,opt_gen,opt_critic,tensorboard_step,writer,scaler_gen,scaler_critic):
    loop = tqdm(loader,leave=True)
    
    i = 0
    for batch_idx,(real,labels) in enumerate(loop):
        i += 1
        if i%2 == 0:
            continue
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size,Z_DIM,1,1).to(DEVICE)

        fake_labels = torch.randint(0, 4, (cur_batch_size,)).to(DEVICE)
        ## Train Critic
        ## Wasserstein Loss : Maximize "E[Critic(real)] - E[Critic(fake)]"   ==   Minimize "-(E[Critic(real)] - E[Critic(fake)])"
        with torch.cuda.amp.autocast():
            fake = gen(noise,alpha,step,fake_labels).to(DEVICE)
            critic_real = critic(real,alpha,step,labels)
            critic_fake = critic(fake.detach(),alpha,step,fake_labels)
            gp = gradient_penalty(critic,real,fake,alpha,step,device=DEVICE)
            loss_critic = -1 * (torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp + 0.001 * torch.mean(critic_real**2)
        
        critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()
        
        ## Train Generator
        ## Maximize "E[Critic(fake)]"   ==   Minimize "- E[Critic(fake)]"
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake,alpha,step, fake_labels)
            loss_gen = -1 * torch.mean(gen_fake)
            
        gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
    
        alpha += (cur_batch_size/len(dataset)) * (1/PROGRESSIVE_EPOCHS[step]) * 2
        alpha = min(alpha,1)
        
        #if batch_idx % 500 == 0:
        #    with torch.no_grad():
        #        fixed_fakes = gen(FIXED_NOISE,alpha,step) * 0.5 + 0.5
        #        save_on_tensorboard(writer,loss_critic.item(),loss_gen.item(),real.detach(),fixed_fakes.detach(),tensorboard_step)
        tensorboard_step += 1
    
    return tensorboard_step,alpha

def train_model():      
    ## build model
    gen = Generator_C(Z_DIM,IN_CHANNELS,IMG_CHANNELS).to(DEVICE)
    critic = Discriminator_C(IN_CHANNELS,IMG_CHANNELS).to(DEVICE)

    ## initialize optimizer,scalers (for FP16 training)
    opt_gen = optim.Adam(gen.parameters(),lr=LR,betas=(0.0,0.99))
    opt_critic = optim.Adam(critic.parameters(),lr=LR,betas=(0.0,0.99))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

    ## tensorboard writer
    writer = SummaryWriter(f"runs/PG_GAN")
    tensorboard_step = 0

    step = int(log2(START_TRAIN_IMG_SIZE/4)) ## starts from 0
    global_epoch = 0

    ## if checkpoint files exist, load model
    if LOAD_MODEL:
        step, global_epoch = load_checkpoint(CHECKPOINT_GEN,gen,opt_gen,LR)
        step, global_epoch = load_checkpoint(CHECKPOINT_CRITIC,critic,opt_critic,LR)
        step += 1
        
    gen.train()
    critic.train()
    
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-4
        loader,dataset = get_loader(4*2**step)
        print(f"Image size:{4*2**step} | Current step:{step}")
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}] Global Epoch:{global_epoch}")
            tensorboard_step,alpha = train_fn(gen,critic,loader,dataset,step,alpha,opt_gen,opt_critic,tensorboard_step,writer,scaler_gen,scaler_critic)
            global_epoch += 1
            if global_epoch in GENERATE_EXAMPLES_AT:
                generate_examples_tif(gen,global_epoch,step,Z_DIM, DEVICE, n=3)
            
            # has a habit of running out of memory going into step 5 and step 6 so increasing the frequency of checkpoints here
            if SAVE_MODEL:
                if step < 5 and (epoch+1)%4==0:
                    print("Saving generator checkpoint...")
                    save_checkpoint(gen,opt_gen, step, global_epoch, filename=CHECKPOINT_GEN)
                    print("Saving critic checkpoint...")
                    save_checkpoint(critic,opt_critic, step, global_epoch, filename=CHECKPOINT_CRITIC)
                elif step > 4:
                    print("Saving generator checkpoint...")
                    save_checkpoint(gen,opt_gen, step, global_epoch, filename=CHECKPOINT_GEN)
                    print("Saving critic checkpoint...")
                    save_checkpoint(critic,opt_critic, step, global_epoch, filename=CHECKPOINT_CRITIC)
        step += 1 ## Progressive Growing
    if SAVE_MODEL:
        print("Saving model...")
        save_checkpoint(gen,opt_gen, step, global_epoch, filename=CHECKPOINT_GEN)
        save_checkpoint(critic,opt_critic, step, global_epoch, filename=CHECKPOINT_CRITIC)
        print("Model saved.")

    print("Training finished")
    return gen, critic

def load_model():      
    ## build model
    print("Loading model...")
    gen = Generator(Z_DIM,IN_CHANNELS,IMG_CHANNELS).to(DEVICE)
    critic = Discriminator(IN_CHANNELS,IMG_CHANNELS).to(DEVICE)

    ## initialize optimizer,scalers (for FP16 training)
    opt_gen = optim.Adam(gen.parameters(),lr=LR,betas=(0.0,0.99))
    opt_critic = optim.Adam(critic.parameters(),lr=LR,betas=(0.0,0.99))

    ## if checkpoint files exist, load model
    print("* Loading Generator...")
    load_checkpoint(CHECKPOINT_GEN,gen,opt_gen,LR)
    print("* Loading Critic...")
    load_checkpoint(CHECKPOINT_CRITIC,critic,opt_critic,LR)
        
    gen.eval()
    critic.eval()

    print("Model loaded")
    return gen, critic


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--mode", help="train or load a model: 'train' or 'load'")
    argParser.add_argument("-p", "--plot", help="plot example image: 'rgb' or 'full'")
    argParser.add_argument("-t", "--test", help="for saving tifs: 'tif'")
    args = argParser.parse_args()
    
    # Choosing model mode
    if args.mode in ('train', 'TRAIN'):
        gen, critic = train_model()
    elif args.mode in ('load', 'LOAD'):
        gen, critic = load_model()
    else:
        print("Invalid input: --mode <train OR load>")

    # Plotting sample
    if args.plot in ('rgb', 'RGB'):
        print("Plotting example RGB image...")
        plot_sample(gen,DEVICE)
    elif args.plot in ('full', 'FULL'):
        plot_bands_all(gen,DEVICE)
    else:
        print("Invalid input: --plot rgb/full")
    
    # Saving tif
    if args.test in ('tif', 'TIF'):
        print("help me")

if __name__ == '__main__':
    #main(sys.argv[1:])
    main()
