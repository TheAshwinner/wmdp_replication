from datasets import load_dataset


def main():
  # Import dataset and convert to jsonl

  ds_cyber_retain = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus")
  ds_cyber_retain["train"].to_json("data/cyber-retain-corpus.jsonl", lines=True)

  ds_cyber_forget = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus")
  ds_cyber_forget["train"].to_json("data/cyber-forget-corpus.jsonl", lines=True)

  ds_bio_retain = load_dataset("cais/wmdp-corpora", "bio-retain-corpus")
  ds_bio_retain["train"].to_json("data/bio-retain-corpus.jsonl", lines=True)

  # Special permission is needed to use the bio_forget_corpus


if __name__ == "__main__":
  main()
