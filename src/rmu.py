import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import JsonlDataset
from model import Model
import copy

class RMU:
  def __init__(self, model, tokenizer, device, alpha, lr, c, hidden_dimension_size, ctx_window, min_len, seed = 42):
    self.unlearned_model = Model(model, tokenizer, device, seed)
    self.frozen_model = copy.deepcopy(self.unlearned_model)
    self.tokenizer = tokenizer
    self.retain_datasets = []
    self.forget_datasets = []
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
    
  
  def setup(self):
    # Initialize u
    u = torch.randn(self.hidden_dimension_size)
    u = u / torch.norm(u)
    self.u = u
    
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

    # TODO: make sure these datasets are the same size?
    self.retain_datasets.append(cyber_retain)
    self.forget_datasets.append(cyber_forget)



  def rmu_step(self, layer_idx):
    print("Beginning RMU step...")
    


    # TODO: Freeze the model parameters at a given layer
    for i in range(len(self.forget_datasets[0].data)):
      print(self.forget_datasets[0].data[i]["text"])
      print("length: ", len(self.forget_datasets))
      print("length of data: ", len(self.forget_datasets[0].data))
      print(self.forget_datasets[0][i])
      break
    

    print("Finished RMU step...")