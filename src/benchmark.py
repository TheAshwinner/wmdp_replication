import lm_eval
import os
import json
import argparse
from utils import load_yaml_config, CustomJSONEncoder
from typing import Dict, Any

class Benchmark:
    def __init__(self, args, results_path):
        self.args = args
        self.results_path = results_path
        
    def run_benchmark(self) -> Dict[str, Any]:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.args.model_name},trust_remote_code={self.args.trust_remote_code}",
            tasks=self.args.benchmarks,
            device=self.args.device,
            log_samples=self.args.log_samples,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
            cache_requests=self.args.cache_requests
        )
        results['results']['model_name'] = self.args.model_name
        return results

    def save_results(self, results) -> None:
        full_dir_name = os.path.join(self.results_path, self.args.model_name)
        if not os.path.exists(full_dir_name):
            os.makedirs(full_dir_name)
        with open(os.path.join(full_dir_name, "results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

def main():
    parser = argparse.ArgumentParser("Run unlearning benchmarks")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--results_path", type=str, help="Path to save the results", default="results")
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    print(config)
    benchmark = Benchmark(config, args.results_path)
    results = benchmark.run_benchmark()
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
