import random
import torch
import torchvision
from torchvision.utils import save_image
import rasterio
import os
import matplotlib.pyplot as plt
import numpy as np
       
def gradient_penalty(critic,real,fake,alpha,train_step,device="cpu"):
    BATCH_SIZE,C,H,W = real.shape
    beta = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    
    interpolated_images = real * beta + fake.detach() * (1-beta)
    interpolated_images.requires_grad_(True)
    
    ## Calculate critic scores
    mixed_scores = critic(interpolated_images,alpha,train_step)
    
    ## Take the gradient of the scores with respect to the image
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True
    )[0]
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

def save_checkpoint(model,optimizer,step,epoch,filename="my_checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "step" : step,
        "epoch" : epoch
    }
    torch.save(checkpoint,filename)
    print("Checkpoint saved.")

    
def load_checkpoint(checkpoint_file,model,optimizer,lr):
    checkpoint = torch.load(checkpoint_file,map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("* Checkpoint loaded.")
    return checkpoint["step"], checkpoint["epoch"]

        
def generate_examples(gen,current_epoch,steps,z_dim,device,n=16):
    gen.eval()
    alpha = 1.0
    
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1,z_dim,1,1).to(device)
            generated_img = gen(noise,alpha=alpha,steps=steps)
            save_image(generated_img*0.5+0.5,f"generated_images/step{steps}_epoch{current_epoch}_{i}.png")
    
    gen.train()

def save_tif(input_img, filename):
    input_img = input_img.cpu().detach()[0]
    with rasterio.open(
                os.path.join(f"generated_images/{filename}.tif"), 
                mode="w",
                driver="GTiff",
                count=input_img.shape[0],
                height=input_img.shape[1], 
                width=input_img.shape[2],
                crs='EPSG:32737',
                dtype=rasterio.uint16,
                transform=rasterio.Affine(1, 0, 0, 0, 1, 0)
            ) as new_file:
                new_file.write(input_img, [x for x in range(1,14)]) 

def plot_sample(gen,epoch,device,z_dim=256,steps=6,n=10,training=True):
    if n % 5 != 0:
        n += n%5
    rows = int(n/5)
    cols = int(n / rows)
    fig, axs = plt.subplots(rows, cols, figsize=(12,5))
    fig.suptitle("Synthetic SEN12MS RGB Images", fontsize=16)
    plt.axis('off')
    alpha=1
    for i, ax in enumerate(axs if n <=1 else axs.flatten()):
        gen.eval()
        with torch.no_grad():
            noise = torch.randn(1,z_dim,1,1).to(device)
            generated_img = gen(noise,alpha=alpha,steps=steps)
            img = np.transpose((generated_img*0.5+0.5)[0].detach().cpu().numpy(), (1,2,0))
            ax.imshow(img[:, :, 1:4])
            ax.set_title(f'Image #{i}')
            ax.axis('off')
        gen.train()
    plt.show()
    fig_title = f"sen12_epoch_{epoch}_step_{steps}.png" if training else f"sen12_sample_rgb.png"
    #plt.savefig(f"generated_images/{fig_title}")
    
def plot_bands_all(gen,device,z_dim=256,steps=6, training=True):
    band_names = [
        "B1: Coastal aerosol", "B2: Blue", "B3: Green", "B4: Red",
        "B5: Vegetation red edge", "B6: Vegetation red edge", "B7: Vegetation red edge",
        "B8: NIR", "B8A: Narrow NIR", "B9: Water Vapour", "B10: SWIR - Cirrus",
        "B11: SWIR", "B12: SWIR"
    ]
    # Generate image from GAN
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(1,z_dim,1,1).to(device)
        generated_img = gen(noise,alpha=1,steps=steps)
        img = np.transpose((generated_img*0.5+0.5)[0].detach().cpu().numpy(), (1,2,0))
        fig, axs = plt.subplots(3,5,figsize=(15,9.5))
        plt.axis('off')
        fig.suptitle("Synthetic SEN12MS Image Bands", fontsize=20)
        for i, ax in enumerate(axs.flatten()):
            if i < 13:
                ax.imshow(img[:,:,i])
                ax.set_title(band_names[i])
            else:
                ax.set_axis_off()
    gen.train()
    plt.show()
