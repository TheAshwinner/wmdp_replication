# We want a Dataset class that loads in a jsonl file, tokenizes the dataset as expected and returns the input ids and attention mask.
import os
import json
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from datasets import load_dataset

WIKITEXT_DATASET_NAME = "Salesforce/wikitext"

class BaseDataset(ABC, Dataset):
    def __init__(self, tokenizer, tokenizer_max_length, batch_size, min_len, device):
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.tokenizer_max_length = tokenizer_max_length
        self.batch_size = batch_size
        self.min_len = min_len
        self.device = device
        self.data = []
        
    @abstractmethod
    def _load_dataset(self):
        pass
        
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.tokenizer(item["text"], return_tensors="pt", padding=True, truncation=True, max_length=self.tokenizer_max_length)
        inputs = {key: value.to(self.device) for key, value in input_ids.items()}
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        
    def __len__(self):
        return len(self.data)


class JsonlDataset(BaseDataset):
  def __init__(self, tokenizer, tokenizer_max_length, batch_size, min_len, device, dataset_name, dataset_folder):
    super().__init__(tokenizer, tokenizer_max_length, batch_size, min_len, device)
    self.dataset_name = dataset_name
    self.dataset_folder = dataset_folder

  def _load_dataset(self):
    dataset_path = os.path.join(self.dataset_folder, self.dataset_name)
    if not os.path.exists(dataset_path):
      raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    data_list = []
    with open(dataset_path, "r") as f:
      for line in f:
        data = json.loads(line)
        if len(data["text"]) > self.min_len:
          data_list.append(data)

    self.data = data_list

class WikitextDataset(BaseDataset):
  def __init__(self, tokenizer, tokenizer_max_length, batch_size, min_len, device, wikitext_config = "wikitext-2-raw-v1", wikitext_split="train"):
    super().__init__(tokenizer, tokenizer_max_length, batch_size, min_len, device)
    self.wikitext_config = wikitext_config
    self.wikitext_split = wikitext_split

  def _load_dataset(self):
    dataset = load_dataset(WIKITEXT_DATASET_NAME, self.wikitext_config)
    self.data = dataset[self.wikitext_split]
