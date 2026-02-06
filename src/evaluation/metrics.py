# src/evaluation/metrics.py

def print_metrics(results):
    for model_name, metrics in results.items():
        print(f"=== {model_name} ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("\n")
