import numpy as np
import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample3(model, scheduler,run):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((100,
                      1,
                      28,
                      28)).to(device)
    for i in tqdm(reversed(range(1000))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        npy_dir = 'dir/Diffusion/results/train_1_3/npy'
        png_dir = 'dir/Diffusion/results/train_1_3/png'
        if(i == 0):
            for j in range(100):
                img = ims[j,:,:,:]
                #print(img.shape)
                img = F.interpolate(img.unsqueeze(0), size=(150), mode='bilinear', align_corners=False)
                #print(img.shape)
                img = img.squeeze(0)
                #print(img.shape)
                img_pil = torchvision.transforms.ToPILImage()(img)
                img_pil.save(os.path.join(png_dir, f'{j+100*run}.png'))
                np.save(os.path.join(npy_dir, f'{j+100*run}.npy'), img.numpy())

    print("Successfully generated 100 images. This was the {}th run.".format(run+1))






