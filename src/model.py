import torch

class Model():
  def __init__(self, model, tokenizer, device, seed = 42):
    self.model = model.to(device)
    self.tokenizer = tokenizer
    self.device = device
    self.seed = seed
    torch.manual_seed(seed)
    self.activations = {}

  def hook_fn(self, module, input, output):
    self.activations["transformer_block_output"] = output[0].detach()
  
  def forward(self, inputs, layer_idx: int, with_grad: bool):
    if layer_idx >= len(self.model.transformer.h):
      raise ValueError(f"Layer index {layer_idx} is out of bounds for the model. The model has {len(self.model.transformer.h)} layers.")
    try:
      hook = self.model.transformer.h[layer_idx].register_forward_hook(self.hook_fn)
      tokenized_inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)

      if with_grad:
        _ = self.model(tokenized_inputs)
      else:
        with torch.no_grad():
          _ = self.model(tokenized_inputs)
    finally:
      hook.remove()
    return self.activations["transformer_block_output"]