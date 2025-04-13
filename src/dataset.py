# We want a Dataset class that loads in a jsonl file, tokenizes the dataset as expected and returns the input ids and attention mask.
import os
import json

class JsonlDataset():
  def __init__(self, tokenizer, tokenizer_max_length, batch_size, min_len, dataset_name, dataset_folder, device):
    self.tokenizer = tokenizer
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    self.tokenizer_max_length = tokenizer_max_length
    self.batch_size = batch_size
    self.min_len = min_len
    self.dataset_name = dataset_name
    self.dataset_folder = dataset_folder
    self.data = []
    self.device = device

  def __getitem__(self, idx):
    item = self.data[idx]
    input_ids = self.tokenizer(item["text"], return_tensors="pt", padding=True, truncation=True, max_length=self.tokenizer_max_length)
    inputs = {key: value.to(self.device) for key, value in input_ids.items()}
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

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
    
  def __len__(self):
    return len(self.data)

