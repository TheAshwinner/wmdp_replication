import torch
import numpy as np
from torch.utils.data import DataLoader

class RMU:
  def __init__(self, model, datasets, device, alpha, lr, c, hidden_dimension_size, ctx_window, seed = 42):
    self.model = model
    self.datasets = datasets
    self.device = device
    self.alpha = alpha
    self.lr = lr
    self.c = c
    self.seed = seed
    self.hidden_dimension_size = hidden_dimension_size
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize u
    u = torch.randn(self.hidden_dimension_size)
    u = u / torch.norm(u)
    self.u = u

  def rmu_step(self, d_forget, d_retain, layer_idx):
    d_forget_dataloader = DataLoader(d_forget, batch_size=1, shuffle=True)
    d_retain_dataloader = DataLoader(d_retain, batch_size=1, shuffle=True)

    for batch in d_forget_dataloader:
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      
