import torch

from typing import List
import pdb

class Model():
  def __init__(self, model, device, seed = 42):
    self.model = model.to(device)
    self.device = device
    self.seed = seed
    torch.manual_seed(seed)

  
  def forward(self, inputs, layer_idx: int, with_grad=False):
    if layer_idx >= len(self.model.model.layers):
      raise ValueError(f"Layer index {layer_idx} is out of bounds for the model. The model has {len(self.model.transformer.h)} layers.")
    
    inputs = inputs.to(self.device)
    
    activations = []
    def hook_fn(module, input, output):
      act = output[0] if isinstance(output, tuple) else output
      activations.append(act.to(self.device))


    try:
      hook = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
      

      if with_grad:
        _ = self.model(inputs)
      else:
        with torch.no_grad():
          _ = self.model(inputs)

    finally:
      hook.remove()
    if len(activations) != 1:
      raise ValueError("Activations length is not 1. Surprising.")

    return activations[0]

  def save_model(self, path):
    self.model.save_pretrained(path)
    print(f"Saved model to {path}")
    return
  
  def get_parameters(self):
    return self.model.parameters()

  def get_layers(self):
    # TODO: this works for Zephyr model, but likely needs to be updated for other models
    return self.model.model.layers