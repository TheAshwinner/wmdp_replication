import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import JsonlDataset
from model import Model
import tqdm
import copy
import pdb

class RMU:
  def __init__(self, model, tokenizer, datasets, device, alpha, lr, c, hidden_dimension_size, tokenizer_max_length, min_len, layer_idx, seed = 42):
    self.unlearned_model = Model(model, tokenizer, device, seed)
    self.frozen_model = copy.deepcopy(self.unlearned_model)
    self.tokenizer = tokenizer
    self.retain_datasets = []
    self.forget_datasets = []
    self.device = device
    self.alpha = alpha
    self.lr = lr
    self.c = c
    self.tokenizer_max_length = tokenizer_max_length
    self.min_len = min_len
    self.seed = seed
    self.hidden_dimension_size = hidden_dimension_size
    self.layer_idx = layer_idx
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    self.freeze_layers_in_unlearned_model([self.layer_idx-2, self.layer_idx-1, self.layer_idx])

    # Initialize random unit vector u
    some_big_number = 15000
    u = torch.randn(some_big_number, self.hidden_dimension_size).to(self.device)
    u = u / torch.linalg.norm(u, dim=-1, keepdim=True).to(self.device)
    self.u = u

  def freeze_layers_in_unlearned_model(self, unfreeze_layers: list[int]):
    # Validation
    for layer in unfreeze_layers:
      if layer < 0:
        raise ValueError(f"Layer index cannot be negative. layer: {layer}")
      if layer >= len(self.unlearned_model.model.model.layers):
        raise ValueError(f"Layer index cannot be greater than the number of layers in the model. layer: {layer}, num layers:  {len(self.unlearned_model.model.model.layers)}")

    # Freeze all layers first
    for param in self.unlearned_model.model.parameters():
      param.requires_grad = False

    # Unfreeze the specified layers
    for layer in unfreeze_layers:
      for param in self.unlearned_model.model.model.layers[layer].parameters():
        param.requires_grad = True

    print("printing params and grads")
    for param in self.unlearned_model.model.parameters():
      if param.requires_grad:
        print(param.requires_grad)
    print("done printing params and grads")
    

  def retain_loss(self, act_retain, act_forget):
    l2_squared = torch.sum((act_retain - act_forget) ** 2, dim=-1)
    final = torch.mean(l2_squared)
    return final

  def forget_loss(self, act_updated):
    l2_squared = torch.sum((act_updated - self.c * self.u[:len(act_updated[0]), :]) ** 2, dim=-1)
    final = torch.mean(l2_squared)
    return final

  def rmu_step(self):
    print("Beginning RMU step...")
    cyber_forget = JsonlDataset(
      tokenizer=self.tokenizer, tokenizer_max_length=self.tokenizer_max_length, batch_size=1,
      min_len=self.min_len, dataset_name="cyber-forget-corpus.jsonl", dataset_folder="data/", device=self.device
    )
    cyber_forget._load_dataset()
    cyber_retain = JsonlDataset(
      tokenizer=self.tokenizer, tokenizer_max_length=self.tokenizer_max_length, batch_size=1,
      min_len=self.min_len, dataset_name="cyber-retain-corpus.jsonl", dataset_folder="data/", device=self.device
      )
    cyber_retain._load_dataset()
    print("after loading datasets")

    # Retain loss
    for i in tqdm.tqdm(range(len(cyber_retain.data))):
      act_updated_retain = self.unlearned_model.forward(cyber_retain[i]["input_ids"], self.layer_idx, with_grad=True)
      act_frozen_retain = self.frozen_model.forward(cyber_retain[i]["input_ids"], self.layer_idx, with_grad=False)
      retain_loss = self.retain_loss(act_updated_retain, act_frozen_retain)
      print(retain_loss)
      break

    # Forget loss
    for i in tqdm.tqdm(range(len(cyber_forget.data))):
      act_updated_forget = self.unlearned_model.forward(cyber_forget[i]["input_ids"], self.layer_idx, with_grad=True)
      forget_loss = self.forget_loss(act_updated_forget)
      print(forget_loss)
      break


    print("act_updated_retain.requires_grad:", act_updated_retain.requires_grad)
    print("act_updated_forget.requires_grad:", act_updated_forget.requires_grad)

    full_loss = forget_loss + self.alpha * retain_loss
    optimizer = torch.optim.AdamW(self.unlearned_model.model.parameters(), lr=self.lr)
    optimizer.zero_grad()
    full_loss.backward()
    optimizer.step()

    print("Finished RMU step...")
    return