import time
import torch
from transformers import AutoModelForSequenceClassification

def run_latency_benchmark():
    """Genuine latency benchmark using time.perf_counter."""
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.eval()
    
    inputs = {
        "input_ids": torch.randint(0, 30522, (1, 128)),
        "attention_mask": torch.ones(1, 128)
    }
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs)
            
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    return avg_latency

if __name__ == "__main__":
    run_latency_benchmark()
