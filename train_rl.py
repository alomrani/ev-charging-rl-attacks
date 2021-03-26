import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from imblearn.over_sampling import ADASYN
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from options import get_options
from soc_dataset import SoCDataset
from reinforce_baseline import ExponentialBaseline

from DNNAgent import mal_agent
from charging_env import charging_ev

def train(opts):

  torch.random.manual_seed(opts.seed)
  dataset = torch.load("drive/MyDrive/PEVdata/dataset.pt")
  dataset = dataset.reshape(dataset.size(0) * dataset.size(1), -1)
  train_loader = DataLoader(SoCDataset(dataset[:, :-1], dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)

  # env = charging_ev(num_cars, num_timesteps, total_power, epsilon, battery_capacity, opts.device, batch_size)
  baseline = ExponentialBaseline(opts.beta)
  agent = mal_agent(opts.hidden_size, opts.num_cars).to(opts.device)
  optimizer = Adam(agent.parameters(), lr=0.001)
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: opts.lr_decay ** epoch
  )
  agent.train()
  average_reward = []
  loss_log = []
  for epoch in range(opts.num_epochs):
    r = train_batch(agent, train_loader, opts.batch_size, optimizer, baseline, loss_log, average_reward, opts)
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



def train_batch(agent, train_loader, batch_size, optimizer, baseline, loss_log, average_reward, opts):
  
  for i, (x, y) in enumerate(train_loader):
    x = x.to(opts.device)
    log = torch.zeros(batch_size, 1, device=opts.device)
    env = charging_ev(opts.num_cars, opts.num_timesteps, opts.total_power, opts.epsilon, opts.battery_capacity, opts.device, batch_size)
    while not env.finished():
      num_charging = torch.poisson(torch.ones(batch_size, device=opts.device) * opts.lamb)
      num_charging[num_charging > opts.num_cars - 1] = opts.num_cars - 1
      requests = torch.rand(batch_size, opts.num_cars - 1, device=opts.device)
      requests = (torch.arange(requests.size(1), device=opts.device) >= num_charging.unsqueeze(1)).float() + requests

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
  return r



def main():
  train(get_options())
