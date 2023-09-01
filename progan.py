# This original ProGAN architecture comes from here: https://www.kaggle.com/code/paddeaux/pggan-progressive-growing-gan-pggan-pytorch/edit

# Base packages
import numpy as np
import pandas as pd
from math import log2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, getopt
import argparse
# Torch packages
import torch
from torchinfo import summary

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ProGAN libraries
from src.layers import *
from src.loaders import SEN12MS_RGB, SEN12MS_FULL, overfit, overfit_sen12
from src.utils import *

# SEN12MS dataloader
from src.sen12ms_dataLoader import *

torch.backends.cudnn.benchmarks = True
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_loader(img_size, DATASET, BATCH_SIZES, NUM_WORKERS):
    IMG_CHANNELS = 13
    transform_sen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size),antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],[0.5 for _ in range(IMG_CHANNELS)])
    ])
    
    batch_size = BATCH_SIZES[int(log2(img_size/4))]
    # SEN12MS Dataset
    #dataset = SEN12MS_RGB(targ_dir=DATASET, transform=transform_sen)
    dataset = SEN12MS_FULL(targ_dir=DATASET, transform=transform_sen)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    
    return loader,dataset

def train_fn(gen,critic,loader,dataset,step,alpha,epoch,opt_gen,opt_critic,scaler_gen,scaler_critic,DEVICE,Z_DIM,LAMBDA_GP,PROGRESSIVE_EPOCHS):
    loop = tqdm(loader,leave=True)
    generator_losses = pd.DataFrame({"size":[], "epoch":[], "batch_number":[], "loss":[]})
    critic_losses = pd.DataFrame({"size":[], "epoch":[], "batch_number":[], "loss":[]})
    i = 0
    for batch_idx,(real,_) in enumerate(loop):
        i += 1
        if i%2 == 0:
            continue
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size,Z_DIM,1,1).to(DEVICE)
        
        ## Train Critic
        ## Wasserstein Loss : Maximize "E[Critic(real)] - E[Critic(fake)]"   ==   Minimize "-(E[Critic(real)] - E[Critic(fake)])"
        with torch.cuda.amp.autocast():
            fake = gen(noise,alpha,step).to(DEVICE)
            critic_real = critic(real,alpha,step)
            critic_fake = critic(fake.detach(),alpha,step)
            gp = gradient_penalty(critic,real,fake,alpha,step,device=DEVICE)
            loss_critic = -1 * (torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp + 0.001 * torch.mean(critic_real**2)
        
        critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()
        
        ## Train Generator
        ## Maximize "E[Critic(fake)]"   ==   Minimize "- E[Critic(fake)]"
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake,alpha,step)
            loss_gen = -1 * torch.mean(gen_fake)
            
        gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
    
        loop.set_postfix({'Critic loss': loss_critic.item(),
                  'Gen loss': loss_gen.item()})

        generator_losses = pd.concat([generator_losses, pd.DataFrame({"size":[(4*2**step)],"epoch":[epoch],"batch_number":[batch_idx], "loss":[loss_gen.item()]})], ignore_index=True)
        critic_losses = pd.concat([critic_losses, pd.DataFrame({"size":[(4*2**step)],"epoch":[epoch],"batch_number":[batch_idx], "loss":[loss_critic.item()]})], ignore_index=True)
        alpha += (cur_batch_size/len(dataset)) * (1/PROGRESSIVE_EPOCHS[step]) * 2
        alpha = min(alpha,1)

    return alpha,generator_losses,critic_losses

def train_model(DEVICE, START_TRAIN_IMG_SIZE, IMG_CHANNELS, Z_DIM, IN_CHANNELS, LAMBDA_GP, DATASET, 
                CHECKPOINT_GEN, CHECKPOINT_CRITIC, SAVE_MODEL, LOAD_MODEL, LR, BATCH_SIZES, PROGRESSIVE_EPOCHS, NUM_WORKERS, GENERATE_EXAMPLES_AT):      
    ## build model
    gen = Generator(Z_DIM,IN_CHANNELS,IMG_CHANNELS).to(DEVICE)
    critic = Discriminator(IN_CHANNELS,IMG_CHANNELS).to(DEVICE)

    ## initialize optimizer,scalers (for FP16 training)
    opt_gen = optim.Adam(gen.parameters(),lr=LR,betas=(0.0,0.99))
    opt_critic = optim.Adam(critic.parameters(),lr=LR,betas=(0.0,0.99))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

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
        loader,dataset = get_loader(4*2**step, DATASET, BATCH_SIZES, NUM_WORKERS)
        print(f"Image size:{4*2**step} | Current step:{step}")
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}] Global Epoch:{global_epoch}")
            alpha,gen_losses, crit_losses = train_fn(gen,critic,loader,dataset,step,alpha,epoch,opt_gen,opt_critic,scaler_gen,scaler_critic,DEVICE,Z_DIM,LAMBDA_GP,PROGRESSIVE_EPOCHS)
            global_epoch += 1
            
            # Generating TIF files (not working)
            if global_epoch in GENERATE_EXAMPLES_AT:
                save_sample(gen, global_epoch, DEVICE, Z_DIM, step, 5)
            
            # has a habit of running out of memory going into step 5 and step 6 so increasing the frequency of checkpoints here
            if SAVE_MODEL:
                if step < 5 and (epoch+1)%4==0:
                    print("Saving generator checkpoint...")
                    save_checkpoint(gen,opt_gen, step, global_epoch, filename=CHECKPOINT_GEN)
                    print("Saving critic checkpoint...")
                    save_checkpoint(critic,opt_critic, step, global_epoch, filename=CHECKPOINT_CRITIC)

                    print("Saving loss data...")
                    gen_losses.to_csv(f'gen_loss_size{4*2**step}_step{step}_epoch{epoch}.csv', index=False)
                    crit_losses.to_csv(f'critic_loss_size{4*2**step}_step{step}_epoch{epoch}.csv', index=False)
                    print(f"Saved to 'gen_loss_step{step}_epoch{epoch}.csv' and 'critic_loss_step{step}_epoch{epoch}.csv' ")
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
    #summary(gen, input_data=[torch.randn(1,256,1,1).to(DEVICE), torch.tensor(1).int().to(DEVICE), torch.tensor(6).int().to(DEVICE)])
    summary(critic, input_data=[torch.randn(1,13,256,256).to(DEVICE), torch.tensor(1).int().to(DEVICE), torch.tensor(6).int().to(DEVICE)])

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
    cwd = os.path.dirname(os.getcwd())
    os.makedirs(os.path.join(cwd,"checkpoints"), exist_ok = True)

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--input_folder', type=str, default='sen12ms', help='name of input folder located in the "input" folder')
    argParser.add_argument('-g', '--checkpoint_gen', type=str, default='gen_sen12_test.pth', help='name of the checkpoint file for Generator')
    argParser.add_argument('-c', '--checkpoint_critic', type=str, default='critic_sen12_test.pth', help='name of the checkpoint file for Critic')
    argParser.add_argument('-lc', '--load_checkpoint', type=bool, default=False, help='Boolean for loading checkpoint in training')
    argParser.add_argument('-sc', '--save_checkpoint', type=bool, default=False, help='Boolean for saving checkpoint in training')
    argParser.add_argument("-m", "--mode", type=str, default='load', help="train or load a model: 'train' or 'load'")
    argParser.add_argument("-p", "--plot", type=str, default='full', help="plot example image: 'rgb' or 'full'")
    argParser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    argParser.add_argument('-b', '--batch_sizes', type=list, default=[256, 256, 128, 64, 32, 16, 8], help='list of batch sizes for each image size')
    argParser.add_argument('-e', '--epochs', type=list, default=[10,10,25,25,50,50,100], help="list of epochs used for each progressive image size")
    argParser.add_argument('-w', '--workers', type=int, default=2, help="number of workers used for dataloader")
    argParser.add_argument('-ex', '--examples', type=list, default=[5,10,15,20,25,50,100,150,200,250], help='list of epochs at which to generate examples')
    args = argParser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    START_TRAIN_IMG_SIZE = 4
    IMG_CHANNELS = 13
    Z_DIM = 256 ## 512 for paper
    IN_CHANNELS = 256 ## 512 for paper
    LAMBDA_GP = 10

    DATASET = os.path.join(cwd, "input", args.input_folder)
    CHECKPOINT_GEN = os.path.join(cwd,"checkpoints", args.checkpoint_gen)
    CHECKPOINT_CRITIC = os.path.join(cwd,"checkpoints", args.checkpoint_critic)
    SAVE_MODEL = args.save_checkpoint
    LOAD_MODEL = args.load_checkpoint
    LR = args.learning_rate
    BATCH_SIZES = args.batch_sizes
    PROGRESSIVE_EPOCHS = args.epochs
    NUM_WORKERS = args.workers
    GENERATE_EXAMPLES_AT = args.examples

    # Choosing model mode
    if args.mode in ('train', 'TRAIN'):
        gen, critic = train_model(DEVICE, START_TRAIN_IMG_SIZE, IMG_CHANNELS, Z_DIM, IN_CHANNELS, 
                                  LAMBDA_GP, DATASET, CHECKPOINT_GEN, CHECKPOINT_CRITIC,
                                  SAVE_MODEL, LOAD_MODEL, LR, BATCH_SIZES, PROGRESSIVE_EPOCHS, NUM_WORKERS, GENERATE_EXAMPLES_AT)
    elif args.mode in ('load', 'LOAD'):
        gen, critic = load_model()
    else:
        print("Invalid input: --mode <train OR load>")

    # Plotting sample
    if args.plot in ('rgb', 'RGB'):
        print("Plotting example RGB image...")
        plot_sample(gen,999,DEVICE)
    elif args.plot in ('full', 'FULL'):
        plot_bands_all(gen,DEVICE)
    else:
        print("Invalid input: --plot rgb/full")

if __name__ == '__main__':
    main()
