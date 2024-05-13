import torch
import subprocess as sp

def get_gpu_memory():
    """Retrieve the free memory of all GPUs using the nvidia-smi tool."""
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[1:-1]
        memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    except Exception as e:
        print(f"Error executing nvidia-smi: {str(e)}")
        memory_free_values = [0]  # Default to 0 if there's an issue fetching memory values
    return memory_free_values

def select_best_gpu():
    """Select the best GPU device based on the maximum free memory available."""
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU...")
        return torch.device('cpu')
    
    free_memory_list = get_gpu_memory()
    max_memory = 0
    best_gpu = 0
    
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}, Free memory: {free_memory_list[i]} MB')
        if free_memory_list[i] > max_memory:
            max_memory = free_memory_list[i]
            best_gpu = i

    best_device = torch.device(f'cuda:{best_gpu}')
    print(f'Selecting GPU {best_gpu} with {max_memory} MB free memory, Device = {best_device}')
    return best_device


if __name__ == "__main__":
    # Use the function to select the best GPU
    selected_device = select_best_gpu()
    print(selected_device)