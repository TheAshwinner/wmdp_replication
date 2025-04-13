import lm_eval
import os
import json
import argparse
from utils import load_yaml_config, CustomJSONEncoder
from typing import Dict, Any
import torch

class Benchmark:
    def __init__(self, args, results_path):
        self.args = args
        self.results_path = results_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def run_benchmark(self) -> Dict[str, Any]:
        results = {}

        if not (hasattr(self.args, 'model_name') and self.args.model_name) or not (hasattr(self.args, 'unlearned_model') and self.args.unlearned_model):
            raise ValueError("model_name and unlearned_model must be provided")
        
        # Evaluate base model
        base_model_name = self.args.model_name
        print(f"Evaluating base model: {base_model_name}")
        base_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={base_model_name},trust_remote_code={self.args.trust_remote_code}",
            tasks=self.args.benchmarks,
            device=self.device,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
        )
        base_results['results']['model_name'] = base_model_name
        results['base_model'] = base_results
        
        # Evaluate unlearned model if provided
        unlearned_model = self.args.unlearned_model
        print(f"Evaluating unlearned model: {unlearned_model}")
        
        # Check if the unlearned model is a local path
        if os.path.exists(unlearned_model):
            # For locally saved models, we need to use the local path directly
            unlearned_model_args = f"pretrained={unlearned_model},trust_remote_code={self.args.trust_remote_code}"
        else:
            # For HF models, use as before
            unlearned_model_args = f"pretrained={unlearned_model},trust_remote_code={self.args.trust_remote_code}"
            
        unlearned_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=unlearned_model_args,
            tasks=self.args.benchmarks,
            device=self.device,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
        )
        # Use the basename of the model path for the results
        unlearned_results['results']['model_name'] = os.path.basename(unlearned_model) if os.path.exists(unlearned_model) else unlearned_model
        results['unlearned_model'] = unlearned_results
        
        return results

    def save_results(self, results) -> None:
        # Create a timestamp-based directory for this evaluation run
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(self.results_path, f"eval_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save base model results
        if 'base_model' in results:
            base_model_name = results['base_model']['results']['model_name']
            base_dir = os.path.join(eval_dir, f"base_{base_model_name}")
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, "results.json"), "w") as f:
                json.dump(results['base_model']['results'], f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        # Save unlearned model results
        if 'unlearned_model' in results:
            unlearned_model_name = results['unlearned_model']['results']['model_name']
            unlearned_dir = os.path.join(eval_dir, f"unlearned_{unlearned_model_name}")
            os.makedirs(unlearned_dir, exist_ok=True)
            with open(os.path.join(unlearned_dir, "results.json"), "w") as f:
                json.dump(results['unlearned_model']['results'], f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        

def main():
    parser = argparse.ArgumentParser("Run unlearning benchmarks")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    print(config)
    # Access the results_path attribute directly.
    # This will raise an AttributeError if it's not defined in the config.
    results_path = config.results_path
    benchmark = Benchmark(config, results_path)
    results = benchmark.run_benchmark()
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
