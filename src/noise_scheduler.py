import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#We will implement a Linear Noise Scheduler lie the original DDPM Paper
class LinNoiseScheduler:
    def __init__(self,num_of_timesteps,beta_start,beta_end):
        self.num_of_timesteps = num_of_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start,beta_end,num_of_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas,dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_1_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod)
    
    def add_noise(self,original,noise,t):
        original_shape = original.shape
        batch_size = original_shape[0]

        t = t.to(self.sqrt_alpha_cum_prod.device) 

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_1_minus_alpha_cum_prod = self.sqrt_1_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_1_minus_alpha_cum_prod = sqrt_1_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(device) * original
                + sqrt_1_minus_alpha_cum_prod.to(device) * noise)
    
    def sample_prev_timestep(self,xt,noise_pred,t):
        x0 = (xt-(self.sqrt_1_minus_alpha_cum_prod[t]*noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0,min=-1., max= 1.)

        mean = xt - ((self.betas[t]*noise_pred)/(self.sqrt_1_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])
        if t==0:
            return mean,x0
        else:
            variance = (1-self.alpha_cum_prod[t-1]) / (1-self.alpha_cum_prod[t])
            variance = variance*self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma*z,x0