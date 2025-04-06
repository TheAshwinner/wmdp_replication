import rmu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
  print("Begin main.")
  # Load model directly
  tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
  model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")

  
  
  rmu_class = rmu.RMU(model, tokenizer, device, 1200, 5e-5, 6.5, 768, 50, 42)
  rmu_class.setup()
  rmu_class.rmu_step(7)
  print(rmu_class)

if __name__ == "__main__":
  main()
