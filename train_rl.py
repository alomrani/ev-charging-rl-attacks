import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
# from imblearn.over_sampling import ADASYN
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.optim import Adam
from options import get_options
from soc_dataset import SoCDataset
from reinforce_baseline import ExponentialBaseline

from DNNAgent import mal_agent
from charging_env import charging_ev
import os
from itertools import product
import json
import seaborn

def train(opts):

  torch.random.manual_seed(opts.seed)
  if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
          json.dump(vars(opts), f, indent=True)
  train_dataset = torch.load(opts.train_dataset)
#  train_dataset = train_dataset.reshape(train_dataset.size(0) * train_dataset.size(1), -1)
  val_dataset = torch.load(opts.val_dataset)
#  val_dataset = val_dataset.reshape(val_dataset.size(0) * val_dataset.size(1), -1)
  

  # env = charging_ev(num_cars, num_timesteps, total_power, epsilon, battery_capacity, opts.device, batch_size)
  if opts.train_seed:
    seeds = [1234, 4321, 1098, 7890]
    agents = []
    rewards = []
    for i in range(len(seeds)):
      torch.random.manual_seed(seeds[i])
      agent, avg_reward = train_epoch(train_dataset, train_dataset, opts)
      agents.append(agent)
      rewards.append(avg_reward)
    
    rewards = np.array(rewards)
    plt.figure(3)
    # plt.title(f"Num Cars {opts.num_cars} arrival rate : {opts.lamb}")
    # plt.xlabel("Batch")
    # plt.ylabel("Average Episode Reward")
    seaborn.set(style="darkgrid", font_scale=1)
    seaborn.tsplot(data=rewards)
    plt.savefig(opts.save_dir + "/avg_rewards_seed.png")
    
  elif opts.tune:
    PARAM_GRID = list(product(
            [0.01, 0.001, 0.0001, 0.00001, 0.02, 0.002, 0.0002, 0.004, 0.0004, 0.00004],  # learning_rate
            [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # baseline exponential decay
            [0.99, 0.98, 0.97, 0.96, 0.95]  # lr decay
        ))
    # total number of slurm workers detected
    # defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # this worker's array index. Assumes slurm array job is zero-indexed
    # defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SCOREFILE = os.path.expanduser(f"./val_rewards_{opts.gamma}_{opts.num_cars}.csv")
    max_val = 0.
    best_params = []
    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
      torch.manual_seed(opts.seed)
      params = PARAM_GRID[param_ix]
      opts.exp_beta = params[1]
      opts.lr_model = params[0]
      opts.lr_decay = params[2]

      agent, _ = train_epoch(train_dataset, val_dataset, opts)
      val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
      avg_r = eval(agent, val_loader, opts)
     # if avg_r > max_val:
     #   best_params = params
     #   max_val = avg_r

      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (avg_r,)))}\n')

    #with open(SCOREFILE, "a") as f:
    #  f.write(f'{"Best params: " + ",".join(map(str, best_params + (avg_r,)))}\n')
  elif opts.eval_only:
    val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
    agent = mal_agent(opts.hidden_size, opts.num_cars).to(torch.device(opts.device))
    load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
    agent.load_state_dict(load_data)
    r, purtubed = eval(agent, val_loader, opts)
    plt.figure(4)
    ax1, = plt.plot(np.arange(opts.num_timesteps), np.array(purtubed[0, :]))
    ax2, = plt.plot(np.arange(opts.num_timesteps), np.array(purtubed[1, :]))
    plt.xlabel("Timestep")
    plt.ylabel("SoC value")
    plt.title(f"Benign vs Malicious reported SoC Sequence gamma={opts.gamma}")
    plt.legend([ax1, ax2], ["benign", "malicious"])
    plt.savefig(opts.save_dir + "/spoof_vs_normal.png")
    print(f"Mean reward: {r}")

  else:
    train_epoch(train_dataset, train_dataset, opts)





def run_env(agent, batch, opts):
  env = charging_ev(opts)
  log = torch.zeros(opts.batch_size, 1, device=opts.device)
  total_purturbation = torch.zeros(opts.batch_size, 1, device=opts.device)
  random_sequence = []
  while not env.finished():
    num_charging = torch.poisson(torch.ones(opts.batch_size, device=opts.device) * opts.lamb)
    num_charging[num_charging > opts.num_cars - 1] = opts.num_cars - 1
    requests = torch.rand(opts.batch_size, opts.num_cars - 1, device=opts.device)
    requests = (torch.arange(requests.size(1), device=opts.device) >= num_charging.unsqueeze(1)).float() + requests
    requests = torch.cat((batch[:, env.timestep].unsqueeze(1), requests), dim=-1)
    requests[requests < 0.] = 0.
    requests[requests > 1.] = 1.
    a, log_p = agent.sample(requests.float())
    total_purturbation += torch.abs(torch.clamp(a, -1., 1.))[:, None]
    requests[:, 0] += a
    requests[requests < 0.] = 0.
    requests[requests > 1.] = 1.
    random_sequence.append(requests[0, 0])
    r = -env.step(requests)[:, 0] / float(env.time)
    log += log_p.unsqueeze(1)
  if opts.eval_only:
    return r, log, total_purturbation / float(env.time), np.array(random_sequence)
  return r, log, total_purturbation / float(env.time)

def train_batch(agent, train_loader, optimizer, baseline, loss_log, average_reward, opts):

  rewards = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(opts.device))
    log = torch.zeros(opts.batch_size, 1, device=opts.device)
    r, log, total_purturbs = run_env(agent, x, opts)
    r = r + total_purturbs.squeeze(1) * opts.gamma * opts.battery_capacity
    rewards.append(r.mean().item())
    optimizer.zero_grad()
    loss = ((r.unsqueeze(1) - baseline.eval(r)) * log).mean()

    loss_log.append(loss.item())
    average_reward.append(-r.mean().item())
    loss.backward()
    optimizer.step()
  return np.array(rewards).mean()


def train_epoch(train_dataset, val_dataset, opts):

  train_loader = DataLoader(SoCDataset(train_dataset[:, :-1], train_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True, num_workers=1)
  val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True, num_workers=1)
  baseline = ExponentialBaseline(opts.exp_beta)
  agent = mal_agent(opts.hidden_size, opts.num_cars).to(torch.device(opts.device))
  optimizer = Adam(agent.parameters(), lr=opts.lr_model)
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: opts.lr_decay ** epoch
  )
  average_reward = []
  loss_log = []
  agent.train()
  for epoch in range(opts.num_epochs):
    r = train_batch(agent, train_loader, optimizer, baseline, loss_log, average_reward, opts)

    lr_scheduler.step()
    if not opts.tune:
      r_val = eval(agent, val_loader, opts)
      print(f"Epoch {epoch}: Average Reward {r_val.mean()}")
  if not opts.tune:
    plt.figure(1)
    line1, = plt.plot(np.arange(len(average_reward)), average_reward)
    plt.xlabel("Batch")
    plt.ylabel("Average Reward")
    plt.savefig(opts.save_dir + "/avg_reward.png")
    plt.figure(2)
    line2, = plt.plot(np.arange(len(loss_log)), loss_log)
    plt.xlabel("Batch")
    plt.ylabel("Policy Loss")
    plt.savefig(opts.save_dir + "/train_loss.png")
    torch.save(agent.state_dict(), opts.save_dir + "/trained_agent.pt")
  return agent, average_reward



def eval(agent, dataloader, opts):
  agent.eval()
  average_reward = []
  for i, (x, y) in enumerate(dataloader):
    x = x.to(opts.device)
    if opts.eval_only:
      r, log, total_purturbs, purturbed_sequence = run_env(agent, x, opts)
    else:
      r, log, total_purturbs = run_env(agent, x, opts)
    r = r + total_purturbs.squeeze(1) * opts.gamma * opts.battery_capacity
    average_reward.append(-r.mean().item())
  if opts.eval_only:
    return np.array(average_reward).mean(), torch.cat((x[0, :].unsqueeze(0), torch.tensor(purturbed_sequence).unsqueeze(0)), dim=0)
  return np.array(average_reward).mean()



if __name__ == "__main__":
  train(get_options())


