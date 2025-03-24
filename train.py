import os
import torch
from tqdm import tqdm
from torch.nn.modules.loss import _Loss, MSELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from dataset import CIFAR10, to_pil_image
from model import UNet
from torchvision.utils import make_grid


class NoiseScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: int = 1e-4, beta_end: int = 2e-2, device: torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.betas = torch.linspace(beta_start, beta_end, steps=timesteps, device=device, dtype=dtype)
        self.alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    def add_noise(self, image: torch.Tensor, timestep: int):
        noise = torch.randn_like(image)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        noisy_image = sqrt_alphas_cumprod_t * image + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image, noise
    
    def remove_noise(self, noisy_image: torch.Tensor, pred_noise: torch.Tensor, timestep: int):
        noise_removed = noisy_image - self.sqrt_one_minus_alphas_cumprod[timestep] * pred_noise
        noise_removed = noise_removed / self.sqrt_alphas_cumprod[timestep]
        return noise_removed


def train(model: UNet, optimizer: Optimizer, criterion: _Loss, dataloader: DataLoader, num_steps: int, device: torch.device, dtype: torch.dtype):
    model = model.to(device, dtype=dtype)
    model.train()

    step = 0
    pbar = tqdm(total=num_steps, desc="Training", unit="step")

    T = 1000
    noise_scheduler = NoiseScheduler(timesteps=T, device=device, dtype=dtype)

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            image = batch.to(device, dtype=dtype)

            timestep = torch.randint(0, T, (image.shape[0],), device=device)
            noisy_image, noise = noise_scheduler.add_noise(image, timestep)

            pred_noise = model(noisy_image, timestep)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
        
            if step % 50_000 == 0:
                sample_out = sample(model, 8, device, dtype)
                save_samples(sample_out, step)
                model.train()

    
    save_samples(sample_out, step)
    pbar.close()

@torch.inference_mode
def sample(model, num_samples: int, device: torch.device, dtype: torch.dtype):
    model.eval()

    T = 1000
    noise_scheduler = NoiseScheduler(timesteps=T, device=device)

    x = torch.randn((num_samples, 3, 32, 32), device=device, dtype=dtype)
    for t in range(T, 0, -1):
        t_tensor = torch.full((num_samples,), t-1, dtype=torch.long, device=device)
        pred_noise = model(x, t_tensor)
        beta_t = noise_scheduler.betas[t - 1]
        alpha_t = noise_scheduler.alphas[t - 1]
        sqrt_one_minus_alpha_cumprod_t = noise_scheduler.sqrt_one_minus_alphas_cumprod[t-1]

        if t > 1:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)
        x = x + torch.sqrt(beta_t) * z
    
    return x

def save_samples(image: torch.Tensor, step: int):
    output_path = f"artifacts/train_outputs/sample_out_{step}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid = make_grid(image * 0.5 + 0.5).to(torch.float32)
    grid_pil = to_pil_image(grid)
    grid_pil.save(output_path)

if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float32

    model = UNet()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
    dataset = CIFAR10()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    train(model, optimizer, criterion, dataloader, 800_000, device, dtype)
