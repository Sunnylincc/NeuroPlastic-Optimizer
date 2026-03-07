import torch
from torch import nn

from neuroplastic_optimizer.optimizer import NeuroPlasticOptimizer

model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
optimizer = NeuroPlasticOptimizer(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 4)
y = torch.randint(0, 3, (32,))

for _ in range(10):
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
