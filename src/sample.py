import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler):
    
    # Sample stepwise by going backward one timestep at a time.
    # We save the x0 predictions, at the last timestep. 
    
    xt = torch.randn((100,
                      1,
                      28,
                      28)).to(device)
    #The tensor above has shape 100x1x28x28 --> 100 imgs each with one channel and spatial dimension of 28.
    for i in tqdm(reversed(range(1000))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists('./results/train_1'):
            os.mkdir('./results/train_1')
        img.save(os.path.join('./results/train_1', 'x0_{}.png'.format(i)))
        img.close()


