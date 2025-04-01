import torch
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_noise_addition(images, noisy_images, step):
    #Visualizes original vs. noisy images for a few cases: 
    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()

    fig, axes = plt.subplots(2, len(images), figsize=(12, 4))

    for i in range(len(images)):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Noisy (Step {step})")

    plt.show()


def train(dataloader, model, epochs, scheduler, lr=1e-4, ckpt_name="model_checkpoint.pth", save_dir="results"):
    """
    Train function for diffusion model.

    Args:
    - dataloader (DataLoader): PyTorch dataloader for training
    - model (torch.nn.Module): Diffusion Unet model
    - epochs (int): Number of training epochs
    - scheduler (LinearNoiseScheduler): Linear Noise scheduler as in the DDPM paper
    - lr (float): Learning rate (default: 1e-4)
    - ckpt_name (str): Name of the checkpoint file (default: "model_checkpoint.pth")
    - save_dir (str): Directory to save model checkpoints (default: "results")
    """

    model.to(device)
    model.train()

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint if available
    ckpt_path = os.path.join(save_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        losses = []
        for im in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            im = im.float().to(device)
            im = im.to(scheduler.sqrt_alpha_cum_prod.device)
            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, scheduler.num_of_timesteps, (im.shape[0],)).to(device)

            # Add noise to images according to the timestep
            noisy_im = scheduler.add_noise(im.to(device), noise, t) # Move images to the selected device
            noise_pred = model(noisy_im, t)
            if epoch % 5 == 0 and len(losses) % 10 == 0:
              visualize_noise_addition(im[:5].squeeze(1), noisy_im[:5].squeeze(1), epoch)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Print loss for the epoch
        print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_path)

    print("Training complete.")
