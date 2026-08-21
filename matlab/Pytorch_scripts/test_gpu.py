#!/usr/bin/env python3
"""
Quick test to verify GPU (MPS) acceleration is working
"""
import torch
import time

print("="*60)
print("PyTorch GPU Test")
print("="*60)

# Check device availability
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"✓ Using device: {device} (Apple Metal)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ Using device: {device}")
else:
    device = torch.device('cpu')
    print(f"⚠ Using device: {device} (No GPU detected)")

print(f"PyTorch version: {torch.__version__}")
print("="*60)

# Simple performance test
size = 5000
print(f"\nPerformance test: {size}x{size} matrix multiplication")

# CPU test
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)
start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# GPU test
if device.type != 'cpu':
    x_gpu = torch.randn(size, size, device=device)
    y_gpu = torch.randn(size, size, device=device)
    
    # Warmup
    _ = torch.matmul(x_gpu, y_gpu)
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    if device.type == 'mps':
        torch.mps.synchronize()  # Wait for MPS operations to complete
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
else:
    print("No GPU available for testing")

print("="*60)
print("✓ Test complete!")
