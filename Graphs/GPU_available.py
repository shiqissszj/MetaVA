import torch

# 检查 MPS（Metal Performance Shaders）是否可用
mps_available = torch.backends.mps.is_available()
print(f"MPS 是否可用: {mps_available}")

# 检查 MPS 是否已构建到 PyTorch 中
mps_built = torch.backends.mps.is_built()
print(f"MPS 是否已构建: {mps_built}")

# 如果 MPS 可用，展示设备信息
if mps_available and mps_built:
    # 设置 MPS 设备
    device = torch.device("mps")
    print(f"当前设备: {device}")

    # 创建一个简单的张量并移动到 MPS 设备
    x = torch.ones(1, device=device)
    print(f"张量在 MPS 设备上: {x}")
else:
    # 如果 MPS 不可用，回退到 CPU
    device = torch.device("cpu")
    print(f"MPS 不可用，使用设备: {device}")

# 检查 CPU 信息（可选）
print(f"CPU 核心数: {torch.multiprocessing.cpu_count()}")