import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import JsonlDataset
from model import Model
import tqdm
import copy
import pdb
import os
from pathlib import Path

class RMU:
  def __init__(self, model_name, datasets, device, alpha, lr, c, hidden_dimension_size, tokenizer_max_length, min_len, layer_idx, num_epochs, num_batches, seed = 42):
    self.device = device
    self.alpha = alpha
    self.lr = lr
    self.c = c
    self.tokenizer_max_length = tokenizer_max_length
    self.min_len = min_len
    self.seed = seed
    self.hidden_dimension_size = hidden_dimension_size
    self.layer_idx = layer_idx
    self.num_epochs = num_epochs
    self.num_batches = num_batches
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.unlearned_model = self.load_model(model_name)
    self.frozen_model = copy.deepcopy(self.unlearned_model)
    self.retain_datasets, self.forget_datasets = self.load_datasets()
    self.model_name = model_name
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)

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
      if layer >= len(self.unlearned_model.get_layers()):
        raise ValueError(f"Layer index cannot be greater than the number of layers in the model. layer: {layer}, num layers:  {len(self.unlearned_model.get_layers())}")

    # Freeze all layers first
    for param in self.unlearned_model.get_parameters():
      param.requires_grad = False

    # Unfreeze the specified layers
    for layer in unfreeze_layers:
      for param in self.unlearned_model.get_layers()[layer].parameters():
        param.requires_grad = True
    

  def retain_loss(self, act_retain, act_forget):
    l2_squared = torch.sum((act_retain - act_forget) ** 2, dim=-1)
    final = torch.mean(l2_squared)
    return final

  def forget_loss(self, act_updated):
    l2_squared = torch.sum((act_updated - self.c * self.u[:len(act_updated[0]), :]) ** 2, dim=-1)
    final = torch.mean(l2_squared)
    return final

  def load_datasets(self):
    # TODO: extend this to other datasets beyond cyber
    # TODO: one edge case to handle is what happens if the two datasets have different lengths
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
    return [cyber_forget], [cyber_retain]

  def load_model(self, model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    my_model = Model(model, self.device, self.seed)
    return my_model
    
  def save_checkpoint(self, epoch, batch_id):
    """Save the current state of the unlearned model."""
    checkpoint_path = f"models/unlearned_epoch{epoch}_batch{batch_id}"
    self.unlearned_model.save_model(checkpoint_path)
    self.tokenizer.save_pretrained(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
  def rmu_step(self, save_frequency=100):
    """
    Run RMU training with periodic checkpoint saving.
    Args:
        save_frequency: Save checkpoint every N batches
    """
    print("Beginning RMU step...")

    for epoch in tqdm.tqdm(range(self.num_epochs)):
      for batch_id in tqdm.tqdm(range(self.num_batches)):
        # interleaving
        dataset_id = batch_id % len(self.forget_datasets)
        element_id = batch_id // len(self.forget_datasets)

        forget_input = self.forget_datasets[dataset_id][element_id]["input_ids"]
        retain_input = self.retain_datasets[dataset_id][element_id]["input_ids"]

        # Forget loss
        act_unlearned_forget = self.unlearned_model.forward(forget_input, self.layer_idx, with_grad=True)
        forget_loss = self.forget_loss(act_unlearned_forget)

        # Retain loss
        act_unlearned_retain = self.unlearned_model.forward(retain_input, self.layer_idx, with_grad=True)
        act_frozen_retain = self.frozen_model.forward(retain_input, self.layer_idx, with_grad=False)
        retain_loss = self.retain_loss(act_unlearned_retain, act_frozen_retain)

        full_loss = forget_loss + self.alpha * retain_loss
        optimizer = torch.optim.AdamW(self.unlearned_model.get_parameters(), lr=self.lr)
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        # Save checkpoint periodically
        if batch_id > 0 and batch_id % save_frequency == 0:
          self.save_checkpoint(epoch, batch_id)

      # Save checkpoint at the end of each epoch
      self.save_checkpoint(epoch, "final")

    print("Finished RMU step...")
    return