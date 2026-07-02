import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.rand(1000, 1000, device="cuda")
y = x @ x
print(y.sum())  # ~2.5e8 esperado