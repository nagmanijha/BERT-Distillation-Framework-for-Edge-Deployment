import subprocess

def run_memory_benchmark():
    """Run memory benchmark tracking VRAM allocated via nvidia-smi."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memories = [int(x.strip()) for x in output.strip().split('\n')]
        print(f"Memory allocated across GPUs: {memories} MB")
        return memories
    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}")
        return []

if __name__ == "__main__":
    run_memory_benchmark()
