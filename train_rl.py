import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from imblearn.over_sampling import ADASYN
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


if torch.cuda.is_available():
  device = "cuda"
else:
  device="cpu"
torch.random.manual_seed(1234)
batch_size = 536
dataset = torch.load("drive/MyDrive/PEVdata/dataset.pt")
dataset = dataset.reshape(dataset.size(0) * dataset.size(1), -1)
train_loader = DataLoader(SoCDataset(dataset[:, :-1], dataset[:, -1][:, None]), batch_size=batch_size, shuffle=True)

rate = 20.

num_timesteps = 48
num_cars = 30
total_power = 1500.
epsilon = 0.6
battery_capacity = 200.
num_epochs = 200
lr_decay = 0.98
# env = charging_ev(num_cars, num_timesteps, total_power, epsilon, battery_capacity, device, batch_size)
hidden_size = 200
beta = 0.7
baseline = ExponentialBaseline(beta)
agent = mal_agent(hidden_size, num_cars).to(device)
optimizer = Adam(agent.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
  optimizer, lambda epoch: lr_decay ** epoch
)
agent.train()
average_reward = []
loss_log = []
for epoch in range(num_epochs):
  for i, (x, y) in enumerate(train_loader):
    x = x.to(device)
    log = torch.zeros(batch_size, 1, device=device)
    env = charging_ev(num_cars, num_timesteps, total_power, epsilon, battery_capacity, device, batch_size)
    while not env.finished():
      num_charging = torch.poisson(torch.ones(batch_size, device=device) * rate)
      num_charging[num_charging > num_cars - 1] = num_cars - 1
      requests = torch.rand(batch_size, num_cars - 1, device=device)
      requests = (torch.arange(requests.size(1), device=device) >= num_charging.unsqueeze(1)).float() + requests

      requests = torch.cat((x[:, env.timestep].unsqueeze(1), requests), dim=-1)
      requests[requests < 0.] = 0.
      requests[requests > 1.] = 1.
      a, log_p = agent.sample(requests.float())
      requests[:, 0] += a
      requests[requests < 0.] = 0.
      requests[requests > 1.] = 1.
      r = -env.step(requests)[:, 0] / float(env.time)
      log += log_p.unsqueeze(1)
    optimizer.zero_grad()
    loss = ((r.unsqueeze(1) - baseline.eval(r)) * log).mean()
    loss_log.append(loss.item())
    average_reward.append(-r.mean().item())
    loss.backward()
    optimizer.step()
  lr_scheduler.step()
  print(f"Epoch {epoch}: Average Reward {-r.mean()}")

plt.figure(2)
line1, = plt.plot(np.arange(len(average_reward)), average_reward)
plt.xlabel("Batch")
plt.ylabel("Average Reward")
plt.figure(3)
line2, = plt.plot(np.arange(len(loss_log)), loss_log)
plt.xlabel("Batch")
plt.ylabel("Policy Loss")
