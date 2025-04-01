
# GravLensDiffusion

This Project aims to generate high quality images of Strong Gravitational Lensing using Deep Generative Modelling , specifically Diffusion Models. I implemented a DDPM following the seminal paper on DDPMs closely, and trained it on a dataset of 10,000 strong Lensing images.
Then I sampled from the trained model to generate strong lensing images.

FID Evaluation was then carried out on a small sample of 100 generated images.

I used extensively used free GPU compute offered by Google Colab, hence my entire project directory is hosted on Google Drive(whose link I had mailed to the team while submitting evaluation tests.)
## Project Structure
The project directory has 4 sub directories: 

This is the structure as in Google Drive Project Directory. Due to Github file upload limits, I haven't uploaded all the files as below on Github. I will provide a projet struture description for Github proj directory just below the one that follows.

```bash
├── notebooks
│   ├── FID_with_downscaled_imgs.ipynb          --->Carries out FID analysis by downscaling Real/GT Images with Generated Image(which are inherently having lower spatial dimension)
│   ├── Frechet_Inception_Distance.ipynb --->Carries out FID analysis on Real Images and Generated Images(upscaled to match spatial dimension of Real Images)
│   ├── Sample_1.ipynb ---> Contains the Implementation of Sampling to generate 100 sample images, we save a grid of 100 images (10 x 10) at all 1000 timesteps (find it at ./results/train_1)
│   ├── Sample_2.ipynb ---> Contains the Implementation of Sampling to generate 100 sample images, we save image state at all 1000 timesteps as .npy for each of the 100 samples and 20 pngs (every 50th timestep) for each of the 100 samples 
│   ├── Sample_3.ipynb ---> We carry out sampling and generate 10000 new images (in a loop run 100 times, each generating 100 images), save the last timestep(generated image) as both .npy and .png.
│   ├── Train1.ipynb ---> Contains the training of Unet model , over 40 epochs using all the 10000 training images.Weights obtained here are stored at "./results/model_checkpoint.pth"
│   └── data_download.ipynb ---> Features code to download the huge dataset from google drive link, a nit of preprocessing and augmnetation, model check and visualization of training images.
├── src
│   ├── data_reshape.py
│   ├── dataset.py
│   ├── dataset2.py
│   ├── model_unet.py ---> This module contains the U-Net model (moderately deep) that is trained to learn the denoising process (reverse diffusion) in DDPM.
│   ├── noise_scheduler.py ---> This module contains the implementation of a linear noise scheduler, which progressively adds noise to input images in the forward diffusion process and gradually destroys information at each time step , and removes noise step by step in the reverse diffusion process to reconstruct the image
│   ├── npy_npz.py
│   ├── npy_npz2.py
│   ├── sample.py ---> This module containss the code for the sampling procedure to generate new images.
│   ├── sample2.py ---> Minor modifications made to the previous module, to support what we do in Sample_2.ipynb
│   ├── sample3.py ---> Minor modifications made to the previous module, to support what we do in Sample_3.ipynb
│   └── train.py ---> Contains the training loop to carry out the training process and learn model weights.
└──  results
│   ├── train_1 ---> Contains 1000 png images which are 10x10 grids , each representing all the 100 images generated in Sample_1.ipynb , at each of the 1000 timesteps. The grid gif you see below iss made using a few images from here.
│   ├── train_1_2 ---> Stores .npy and .png files corresponding to the generation in notebook Sample_2.ipynb.
│   ├── train_1_3 ---> Stores .npy and .png files corresponding to the generation in notebook Sample_3.ipynb
│   └── model_checkpoint.pth ---> Final weights that I am submitting
├── data
│   ├── Samples ---> Conatains all 10000 Training images as .npy files.
│   ├── Samples_png ---> Contains all 10000 Training images as .png files.
│   ├── downscaled_100_pngs ---> This contains 100 randomly chosen training images downscaled (to match the spatial dim of generated images) from 10 K images as png files.
│   └── samples.npz ---> Asingle .npz file containing all 10K training images as numpy arrays.
```

Below is the project structure as in Github, descriptions are just as above:
```bash
├── notebooks
│   ├── FID_with_downscaled_imgs.ipynb 
│   ├── Frechet_Inception_Distance.ipynb -
│   ├── Sample_2.ipynb 
│   ├── Sample_3.ipynb 
│   ├── Train1.ipynb 
│   └── data_download.ipynb 
├── src
│   ├── data_reshape.py
│   ├── dataset.py
│   ├── dataset2.py
│   ├── model_unet.py 
│   ├── noise_scheduler.py
│   ├── npy_npz.py
│   ├── npy_npz2.py
│   ├── sample.py 
│   ├── sample2.py 
│   ├── sample3.py 
│   └── train.py
└──  results
│   ├── merged.png ---> The grid gif you see below iss made using a few images from here. The process of obtaining this is explained in teh above directory structure.
│   ├── 100_pngs ---> This contains 100 sample generated images as png files.
│   └── model_checkpoint.pth ---> Final weights that I am submitting
```

Notice that I haven't uploaded the data directory and many sub directories of the results directory due to upload constraints, obtain them on the Google Drive Link, I shared.


## Results
The gif below shows the progression of 100 random noise samples to 100 generated images of Strong Gravitational Lensing over 1000 timesteps.




![merged](https://github.com/user-attachments/assets/07499f57-e701-4fa4-8bad-03aa131b826c)

The FID score obtained was 27.93, any score below 20 is generally acceptable. Note that I carried out the FID analysis on a meagre 100 generated samples(100 train samples slected at random and downscaled), I will soon carry out the same on a greater sample size and expect this score to fall futher. Also due to GPU restrictions the model was trained for only 40 epochs , and the input(train) iamges had to be downscaled from a dimesnion of 150 (spatial dimension) to 28, due to memory issues. Fixing these will impore the FID further.


A few more generated images are featured below.

![1981](https://github.com/user-attachments/assets/b04c6348-0271-4910-9292-2d36001a651a)


![2004](https://github.com/user-attachments/assets/0cd69849-c01a-4026-a383-ca38c758c830)


![2025](https://github.com/user-attachments/assets/8e1c1970-375d-4866-8962-ef6b4e5f506e)


![2184](https://github.com/user-attachments/assets/fe108bb2-96ac-4881-b0e2-89b2e33ae7bf)


## Acknowledgements:
1. The Unet model implementation takes inspiration from the following repo : https://github.com/explainingai-code/DDPM-Pytorch
2. The FID score calculation was done by a custom method , but the accurate score was evaluated using : https://github.com/mseitzer/pytorch-fid
