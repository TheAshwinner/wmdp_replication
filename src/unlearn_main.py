import rmu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import yaml
import pdb

ALLOWED_MODELS = ["openai-community/gpt2", "HuggingFaceH4/zephyr-7b-beta", "google/gemma-2-2b-it"]

def main():
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser(description="Run model unlearning with specified config.")
  parser.add_argument(
      "--config_path",
      type=str,
      required=True,
      help="Path to the YAML configuration file."
  )
  args = parser.parse_args()

  # --- Load Configuration ---
  try:
      with open(args.config_path, 'r') as f:
          config = yaml.safe_load(f)['args'] # Load the 'args' section
      print("Configuration loaded successfully:")
      print(config)
  except FileNotFoundError:
      print(f"Error: Configuration file not found at {args.config_path}")
      return
  except Exception as e:
      print(f"Error loading or parsing YAML file: {e}")
      return

  # --- Load Model ---
  # Use the model_name from the loaded config
  model_name = config.get('model_name', 'openai-community/gpt2') # Default to gpt2 if not specified
  if model_name not in ALLOWED_MODELS:
    raise ValueError(f"Model {model_name} is not supported. Please use one of the following models: {ALLOWED_MODELS}")
  print(f"\nLoading model: {model_name}...")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

  print("\nModel loaded:")
  print(model)

  # --- TODO: Implement the rest of the unlearning logic using config ---
  # Example: Accessing other parameters
  num_epochs = config.get('num_epochs')
  num_batches = config.get('num_batches')
  batch_size = config.get('batch_size')
  learning_rate = config.get('learning_rate')
  forget_datasets = config.get('forget_dataset_list', [])
  retain_datasets = config.get('retain_dataset_list', [])
  steering_coefficient = config.get('steering_coefficient')
  alpha = config.get('alpha')
  forget_layer_id = config.get('forget_layer_id')
  optimizer_param_layer_id = config.get('optimizer_param_layer_id', [])
  update_layer_ids = config.get('update_layer_ids', [])
  updated_model_path = config.get('updated_model_path', "")
  seed = config.get('seed')

  print(f"\nNumber of epochs: {num_epochs}")
  print(f"Number of batches: {num_batches}")
  print(f"Batch size: {batch_size}")
  print(f"Learning rate: {learning_rate}")
  print(f"Forget datasets: {forget_datasets}")
  print(f"Retain datasets: {retain_datasets}")
  print(f"Steering coefficient: {steering_coefficient}")
  print(f"Alpha: {alpha}")
  print(f"Forget layer ID: {forget_layer_id}")
  print(f"Optimizer param layer ID: {optimizer_param_layer_id}")
  print(f"Update layer IDs: {update_layer_ids}")
  print(f"Updated model path: {updated_model_path}")
  print(f"Seed: {seed}")

  my_rmu = rmu.RMU(model=model,
                   tokenizer=tokenizer,
                   datasets=None,
                   device="cuda",
                   alpha=alpha,
                   lr=float(learning_rate),
                   c=steering_coefficient,
                   hidden_dimension_size=4096,
                   tokenizer_max_length=1024,
                   min_len=30,
                   layer_idx=7, # todo: maybe change this later
                   seed=seed)
  my_rmu.rmu_step()

if __name__ == "__main__":
  main()
