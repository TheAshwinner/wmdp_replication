import rmu
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
  # Load model directly
  tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
  model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

  print(model)

if __name__ == "__main__":
  main()
