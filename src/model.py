import torch

from typing import List
import pdb

class Model():
  def __init__(self, model, tokenizer, device, seed = 42):
    self.model = model.to(device)
    self.tokenizer = tokenizer
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

  # def forward(self, 
  #             input_ids, 
  #             layer_id: int,
  #             with_grad=False
  # ) -> torch.Tensor:
  #     """Forward pass and returns the activations of the specified layer."""
  #     # Ensure input_ids is on the correct device
  #     # input_ids = input_ids.to(self.device)

  #     tokenized_inputs = self.tokenizer(input_ids, return_tensors="pt").to(self.device)
      
  #     activations = []
  #     def hook_function(module, input, output):
  #         # Ensure output is on the correct device
  #         act = output[0] if isinstance(output, tuple) else output
  #         activations.append(act.to(self.device))

  #     hook_handle = self.get_layer(layer_id).register_forward_hook(hook_function)
      
  #     if with_grad:
  #         _ = self.model(**tokenized_inputs)
  #     else:
  #         with torch.no_grad():
  #             _ = self.model(**tokenized_inputs)
          

  #     hook_handle.remove()
  #     activations_tensor = torch.stack(activations)
  #     return activations_tensor.to(self.device)  # Ensure final output is on the correct device
  
  # def get_layer(self, layer_id: int) -> torch.nn.Module:
  #         """
  #         Get a specific layer from the model.

  #         Args:
  #             layer_id: Index of the desired layer

  #         Returns:
  #             The specified model layer

  #         Raises:
  #             IndexError: If layer_id is out of range
  #         """
  #         layers = self.get_all_layers()
  #         if 0 <= layer_id < len(layers):
  #             return layers[layer_id]
  #         else:
  #             raise IndexError(f"Layer ID {layer_id} is out of range. The model has {len(layers)} layers.")
          
  # def get_all_layers(self) -> List[torch.nn.Module]:
  #       if hasattr(self.model, 'transformer'):  # For models like GPT-2
  #           layers = self.model.transformer.h
  #       elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # For other models (e.g., zephyr, Yi, Mixtral)
  #           layers = self.model.model.layers
  #       else:
  #           raise ValueError(f"Unknown architecture for model {self.model_name}. Unable to find layers.")
  #       return layers