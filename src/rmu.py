import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import JsonlDataset
from model import Model

class RMU:
  def __init__(self, model, tokenizer, datasets, device, alpha, lr, c, hidden_dimension_size, ctx_window, min_len, seed = 42):
    self.unlearned_model = Model(model, tokenizer, device, seed)
    self.frozen_model = Model(model, tokenizer, device, seed)
    self.tokenizer = tokenizer
    self.datasets = datasets
    self.device = device
    self.alpha = alpha
    self.lr = lr
    self.c = c
    self.ctx_window = ctx_window
    self.min_len = min_len
    self.seed = seed
    self.hidden_dimension_size = hidden_dimension_size
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize u
    u = torch.randn(self.hidden_dimension_size)
    u = u / torch.norm(u)
    self.u = u

  def rmu_step(self, d_forget, d_retain, layer_idx):
    print("Beginning RMU step...")
    cyber_forget = JsonlDataset(
      tokenizer=self.tokenizer, tokenizer_max_length=self.ctx_window, batch_size=1,
      min_len=self.min_len, dataset_name="cyber-forget-corpus.jsonl", dataset_folder="data/", device=self.device
    )
    cyber_forget._load_dataset()
    cyber_retain = JsonlDataset(
      tokenizer=self.tokenizer, tokenizer_max_length=self.ctx_window, batch_size=1,
      min_len=self.min_len, dataset_name="cyber-retain-corpus.jsonl", dataset_folder="data/", device=self.device
      )
    cyber_retain._load_dataset()

    # TODO: Freeze the model parameters at a given layer
    for i in range(len(cyber_forget.data)):
      print(len(cyber_forget.data[i]["text"]))
      act_forget = self.model.forward(cyber_forget.data[i]["text"], layer_idx)
      act_retain = self.model.forward(cyber_retain.data[i]["text"], layer_idx)
      break
    
    print(act_forget.shape)
    print(act_retain.shape)

    print("Finished RMU step...")