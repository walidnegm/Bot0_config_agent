from llama_cpp import Llama

# Replace with your actual GGUF model path
model_path = "./tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

try:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=1,  # Try offloading 1 layer to GPU
        verbose=True     # Enable detailed logs (default is True)
    )
    print("Model loaded successfully. Check console logs for CUDA details.")
except Exception as e:
    print(f"Error loading model with GPU offload: {e}")
