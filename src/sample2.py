import torch
import torchvision
import numpy as np
import os
from torchvision.utils import make_grid
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample2(model, scheduler):
    
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
        #Save current time step img of all 10 samples as .npy and .png in respective folders 
        for j in range(100):
            img = ims[j]

            npy_dir = './results/train_1_2/sample_{}/npy'.format(j)
            png_dir = './results/train_1_2/sample_{}/png'.format(j)

            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            #NPY saving for all every timestep:
            np.save(os.path.join(npy_dir, f'{j}_{i}.npy'), img.numpy())

            # img2 = torch.Tensor.numpy(img,force=True)
            # if not os.path.exists('./results/train_1_2/sample_{}/npy'.format(j)):
            #     os.mkdir('./results/train_1/sample_{}/npy'.format(j))
            # img2.save(os.path.join('./results/train_1_2/sample{}/npy'.format(j), '{}_{}.npy'.format(j,i)))
            # if (i%50 == 0):
            #     img3 = torchvision.transforms.ToPILImage()(img)
            #     if not os.path.exists('./results/train_1_2/sample_{}/png'.format(j)):
            #         os.mkdir('./results/train_1_2/sample_{}/png'.format(j))
            #     img3 = torchvision.transforms.ToPILImage()(img)
            #     img3.save(os.path.join('./results/train_1_2/sample{}/png'.format(j), '{}_{}.png'.format(j,i)))

            if i % 50 == 0:  # Save PNG every 50 steps
                img_pil = torchvision.transforms.ToPILImage()(img)
                img_pil.save(os.path.join(png_dir, f'{j}_{i}.png'))
            


