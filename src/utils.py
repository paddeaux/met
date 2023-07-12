import random
import torch
import torchvision
from torchvision.utils import save_image

def save_on_tensorboard(writer,loss_critic,loss_gen,real,fake,tensorboard_step):
    writer.add_scalar("Loss Critic",loss_critic,global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    
    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:8],normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8],normalize=True)
        
        writer.add_image("Real",img_grid_real,global_step = tensorboard_step)
        writer.add_image("Fake",img_grid_fake,global_step = tensorboard_step)
        
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

def save_checkpoint(model,optimizer,filename="my_checkpoint.pth"):
    print("Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(checkpoint,filename)
    
def load_checkpoint(checkpoint_file,model,optimizer,lr):
    print("Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def generate_examples(gen,current_epoch,steps,z_dim,device,n=16):
    gen.eval()
    alpha = 1.0
    
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1,z_dim,1,1).to(device)
            generated_img = gen(noise,alpha=alpha,steps=steps)
            save_image(generated_img*0.5+0.5,f"generated_images/step{steps}_epoch{current_epoch}_{i}.png")
#             save_image(generated_img*0.5+0.5,f"step:{steps}_epoch{current_epoch}_{i}.png")
    
    gen.train()