import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import JsonlDataset, WikitextDataset, BaseDataset
from model import Model
import tqdm
import copy
from pathlib import Path
import pdb

class RMU:
  def __init__(self, model_name, forget_datasets, retain_datasets, device, alpha, lr, c, hidden_dimension_size, tokenizer_max_length, min_len, layer_idx, num_epochs, num_batches, seed = 42):
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
    self.batch_size = 1
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.unlearned_model = self.load_model(model_name)
    self.frozen_model = copy.deepcopy(self.unlearned_model)
    self.forget_datasets, self.retain_datasets = self.load_datasets(forget_datasets, retain_datasets)
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

  def load_datasets(self, forget_datasets, retain_datasets) -> tuple[list[BaseDataset], list[BaseDataset]]:
    """
    Load and prepare the datasets for model unlearning.
    
    This method processes the dataset names provided for both forgetting and retaining,
    converting them into actual dataset objects that can be used during training.
    
    Args:
        forget_datasets (list[str]): List of dataset names to be used for unlearning/forgetting.
        retain_datasets (list[str]): List of dataset names to be used for retaining knowledge.
    
    Returns:
        tuple[list[BaseDataset], list[BaseDataset]]: A tuple containing:
            - List of processed forget datasets
            - List of processed retain datasets
    
    Note:
        - Each dataset is loaded and processed using the parse_dataset method
        - Current implementation has TODOs for extending beyond cyber datasets
        - There's a potential edge case when datasets have different lengths
    """

    # TODO: extend this to other datasets beyond cyber
    # TODO: one edge case to handle is what happens if the two datasets have different lengths

    processed_forget_datasets = []
    for dataset_name in forget_datasets:
      dataset = self.parse_dataset(dataset_name)
      dataset._load_dataset()
      processed_forget_datasets.append(dataset)

    processed_retain_datasets = []
    for dataset_name in retain_datasets:
      dataset = self.parse_dataset(dataset_name)
      dataset._load_dataset()
      processed_retain_datasets.append(dataset)

    return processed_forget_datasets, processed_retain_datasets
  
  def parse_dataset(self, dataset_name) -> BaseDataset:
    valid_dataset_names = ["wikitext", "cyber-forget-corpus.jsonl", "bio-forget-corpus.jsonl", "cyber-retain-corpus.jsonl", "bio-retain-corpus.jsonl"]
    if dataset_name not in valid_dataset_names:
      raise ValueError(f"Invalid dataset name. dataset_name: {dataset_name}")
    
    if dataset_name == "wikitext":
      return WikitextDataset(
        tokenizer=self.tokenizer, tokenizer_max_length=self.tokenizer_max_length, batch_size=self.batch_size,
        min_len=self.min_len, device=self.device
      )
    else:
      return JsonlDataset(
        tokenizer=self.tokenizer, tokenizer_max_length=self.tokenizer_max_length, batch_size=self.batch_size,
        min_len=self.min_len, dataset_name=dataset_name, dataset_folder="data/", device=self.device
      )

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
    Perform one step of the RMU algorithm.
    
    This method implements the core RMU algorithm, which aims to selectively unlearn 
    specific information from the model while retaining general capabilities.
    The process iterates through epochs and batches, computing both forget and retain losses,
    and updating the model parameters accordingly.
    
    Args:
        save_frequency (int, optional): Number of batches after which to save a checkpoint.
            Defaults to 100.
    
    Returns:
        None: The method updates the model in-place and saves checkpoints to disk.
    
    Note:
        - Forget loss pushes activations toward a random unit vector
        - Retain loss ensures activations on retain data remain similar to the original model
        - Checkpoints are saved periodically and at the end of each epoch
    """
    print("Beginning RMU step...")

    for epoch in tqdm.tqdm(range(self.num_epochs)):
      for batch_id in tqdm.tqdm(range(self.num_batches)):
        # interleaving
        forget_dataset_id = batch_id % len(self.forget_datasets)
        forget_element_id = batch_id // len(self.forget_datasets)
        retain_dataset_id = batch_id % len(self.retain_datasets)
        retain_element_id = batch_id // len(self.retain_datasets)

        forget_input = self.forget_datasets[forget_dataset_id][forget_element_id]["input_ids"]
        retain_input = self.retain_datasets[retain_dataset_id][retain_element_id]["input_ids"]

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