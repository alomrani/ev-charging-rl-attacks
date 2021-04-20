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

from DNNAgent import mal_rl_agent
from spoof_agent1 import mal_agent1
from spoof_agent2 import mal_agent2
from spoof_agent3 import mal_agent3
from spoof_agent4 import mal_agent4
from charging_env import charging_ev
import os
from itertools import product
import json
import seaborn
from imblearn.over_sampling import ADASYN
from DetectionModelDNN import DetectionModelDNN

def train(opts):

  torch.random.manual_seed(opts.seed)
  np.random.seed(opts.seed)
  if not os.path.exists(opts.save_dir) and not opts.eval_detect:
        os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
          json.dump(vars(opts), f, indent=True)
  train_dataset = torch.load(opts.train_dataset)
#  train_dataset = train_dataset.reshape(train_dataset.size(0) * train_dataset.size(1), -1)
  val_dataset = torch.load(opts.val_dataset)
#  val_dataset = val_dataset.reshape(val_dataset.size(0) * val_dataset.size(1), -1)
  

  # val_dataset = val_dataset.reshape(val_dataset.size(0) * val_dataset.size(1), -1)
  test_dataset = torch.load(opts.test_dataset)
  mal_agent = {
        "rl-agent": mal_rl_agent,
        "attack1": mal_agent1,
        "attack2": mal_agent2,
        "attack3": mal_agent3,
        "attack4": mal_agent4,
  }.get(opts.attack_model, None)
  assert mal_agent is not None, "Unknown model: {}".format(mal_agent)
  # env = charging_ev(num_cars, num_timesteps, total_power, epsilon, battery_capacity, opts.device, batch_size)
  if opts.train_seed:
    seeds = [1234, 4321, 1098, 7890]
    agents = []
    rewards = []
    for i in range(len(seeds)):
      torch.random.manual_seed(seeds[i])
      agent, avg_reward = train_epoch(mal_agent, train_dataset, train_dataset, opts)
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

      agent, _ = train_epoch(mal_agent, train_dataset, val_dataset, opts)
      val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
      avg_r, *_ = eval(agent, val_loader, opts)
      # if avg_r > max_val:
      #   best_params = params
      #   max_val = avg_r

      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (avg_r,)))}\n')

  elif opts.eval_only:
    test_loader = DataLoader(SoCDataset(test_dataset[:, :-1], test_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
    agent = mal_agent(opts.hidden_size, opts.num_cars, opts).to(torch.device(opts.device))
    if opts.load_path is not None:
      load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
      agent.load_state_dict(load_data)
    r, purtubed, power_mal, power_ben = eval(agent, test_loader, opts)
    plt.figure(4)
    ax1, = plt.plot(np.arange(opts.num_timesteps), np.array(purtubed[0, :]))
    ax2, = plt.plot(np.arange(opts.num_timesteps), np.array(purtubed[-2, :]))
    plt.xlabel("Timestep")
    plt.ylabel("SoC value")
    if opts.gamma == -1:
      plt.title(f"Benign vs Malicious reported SoC Sequence {opts.attack_model}")
      opts.gamma = opts.attack_model
    else:
      plt.title(f"Benign vs Malicious reported SoC Sequence gamma={opts.gamma}")
    plt.legend([ax1, ax2], ["benign", "malicious"])

    plt.savefig(opts.save_dir + f"/spoof_vs_normal_{opts.gamma}.png")
    print(f"Mean reward: {r}")
    print(f"Mean Power for Malicious EV: {power_mal}")
    print(f"Mean Power for benign EV: {power_ben}")
  elif opts.eval_detect:
    agent = mal_agent(opts.hidden_size, opts.num_cars, opts).to(torch.device(opts.device))
    if opts.load_path is not None:
      load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
      agent.load_state_dict(load_data)
    eval_detect(agent, test_dataset, opts)
  elif opts.eval_detect_range:
    accuracies = []
    for i in range(len(opts.load_paths)):
      data = opts.load_paths[i]
      agent = mal_agent(opts.hidden_size, opts.num_cars, opts).to(torch.device(opts.device))
      load_data = torch.load(data, map_location=torch.device(torch.device(opts.device)))
      agent.load_state_dict(load_data)
      accuracies.append(eval_detect(agent, test_dataset, opts))
    plt.figure(5)
    plt.plot(np.array(["0.3", "0.4", "0.5", "0.6"]), np.array(accuracies))
    plt.title("Detection Accuracy Against RL Attacks")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.savefig("attacks_detect.png")
  elif opts.create_mal_dataset:
    agent = mal_agent(opts.hidden_size, opts.num_cars, opts).to(torch.device(opts.device))
    # if opts.load_path is not None:
    #   load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
    #   agent.load_state_dict(load_data)
    
    benign_dataset = torch.cat((train_dataset, val_dataset, test_dataset), dim=0)
    train_dataset_b = benign_dataset[:-4000, :]
    val_dataset_b = benign_dataset[-4000:-2000, :]
    test_dataset_b = benign_dataset[-2000:, :]

    loader_benign = DataLoader(SoCDataset(benign_dataset[:, :-1], benign_dataset[:, -1].unsqueeze(1)), batch_size=opts.batch_size, shuffle=True)
    loader_val = DataLoader(SoCDataset(val_dataset_b[:, :-1], val_dataset_b[:, -1].unsqueeze(1)), batch_size=opts.batch_size, shuffle=True)
    loader_test = DataLoader(SoCDataset(test_dataset_b[:, :-1], test_dataset_b[:, -1].unsqueeze(1)), batch_size=opts.batch_size, shuffle=True)
    mal_dataset = generate_mal_samples(agent, loader_benign, opts)
    idx = torch.randperm(mal_dataset.shape[0])
    mal_dataset = mal_dataset[idx].view(mal_dataset.size())
    # mal_dataset_val = generate_mal_samples(agent, loader_val, opts)
    # mal_dataset_test = generate_mal_samples(agent, loader_test, opts)
    train_dataset_imb = torch.cat((train_dataset_b, mal_dataset[:-4000, :]), dim=0)
    ada = ADASYN(random_state=42, sampling_strategy=1., n_neighbors=5)
    x = train_dataset_imb[:, :-1]
    y = train_dataset_imb[:, -1]
    soc_data, label = ada.fit_resample(x, y)

    train_dataset_balanced = torch.cat((torch.tensor(soc_data), torch.tensor(label)[:, None]), dim=1)
    print(train_dataset_balanced.shape)
    idx = torch.randperm(train_dataset_balanced.shape[0])
    train_dataset_balanced = train_dataset_balanced[idx].view(train_dataset_balanced.size())
    validation_rl_whole = torch.cat((val_dataset_b, mal_dataset[-4000:-2000, :]), dim=0)
    test_rl_whole = torch.cat((test_dataset_b, mal_dataset[-2000:, :]), dim=0)
    torch.save(train_dataset_balanced, "detection_train.pt")
    torch.save(validation_rl_whole, "detection_val.pt")
    torch.save(test_rl_whole, "detection_test.pt")

  else:
    train_epoch(mal_agent, train_dataset, val_dataset, opts)

def generate_mal_samples(agent, loader, opts):
  num_samples = len(opts.load_paths)
  mal_samples = []
  for i in range(num_samples):
    torch.random.manual_seed(opts.seed + i*10)
    np.random.seed(opts.seed + i*10)
    load_data = torch.load(opts.load_paths[i], map_location=torch.device(torch.device(opts.device)))
    agent.load_state_dict(load_data)
    r, purturbed, *_ = eval(agent, loader, opts)
    mal_samples.append(purturbed[1:, :])
  mal_dataset = torch.cat(mal_samples, dim=0)
  mal_dataset = torch.cat((mal_dataset, torch.ones(mal_dataset.size(0), 1)), dim=1)
  return mal_dataset

def eval_detect(agent, val_dataset, opts):
  val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  r, purturbed, *_ = eval(agent, val_loader, opts)
  model = DetectionModelDNN(768, opts.num_timesteps, opts.p).to(opts.device)
  model.eval()
  load_data = torch.load(opts.load_path2, map_location=torch.device(torch.device(opts.device)))
  model.load_state_dict(load_data)
  out = model(purturbed[1:, :].float()).detach()
  acc = (out.argmax(1) == 1).float().sum() / (purturbed.size(0) - 1)

  print(f"Detection Accuracy against RL attacks: {acc}")
  return acc


def run_env(agent, batch, opts):
  env = charging_ev(opts)
  batch_size = batch.size(0)
  log = torch.zeros(batch_size, 1, device=opts.device)
  total_purturbation = torch.zeros(batch_size, 1, device=opts.device)
  purturbed_sequence = []
  while not env.finished():
    num_charging = torch.poisson(torch.ones(batch_size, device=opts.device) * opts.lamb)
    num_charging[num_charging > opts.num_cars - 1] = opts.num_cars - 1
    requests = torch.rand(batch_size, opts.num_cars - 1, device=opts.device)
    requests = (torch.arange(requests.size(1), device=opts.device) >= num_charging.unsqueeze(1)).float() + requests
    requests = torch.cat((batch[:, env.timestep].unsqueeze(1), requests), dim=-1)
    requests[requests < 0.] = 0.
    requests[requests > 1.] = 1.
    a, log_p = agent.sample(requests.float())
    total_purturbation += torch.abs(torch.clamp(a, -1., 1.))[:, None]
    requests[:, 0] += a
    requests[requests < 0.] = 0.
    requests[requests > 1.] = 1.
    purturbed_sequence.append(list(requests[:, 0]))
    power = env.step(requests)
    r = -power[:, 0] / float(env.time)
    if log_p is not None:
      log += log_p.unsqueeze(1)
  mean_power_benign = -power[:, 1:].mean() / float(env.time)
  return r, log, total_purturbation / float(env.time), np.array(purturbed_sequence).T, mean_power_benign

def train_batch(agent, train_loader, optimizer, baseline, loss_log, average_reward, opts):

  rewards = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(opts.device))
    log = torch.zeros(opts.batch_size, 1, device=opts.device)
    r, log, total_purturbs, _ = run_env(agent, x, opts)
    r = r + total_purturbs.squeeze(1) * opts.gamma * opts.battery_capacity
    rewards.append(r.mean().item())
    optimizer.zero_grad()
    loss = ((r.unsqueeze(1) - baseline.eval(r)) * log).mean()

    loss_log.append(loss.item())
    average_reward.append(-r.mean().item())
    loss.backward()
    optimizer.step()
  return np.array(rewards).mean()


def train_epoch(mal_agent, train_dataset, val_dataset, opts):

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
      r_val, *_ = eval(agent, val_loader, opts)
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
  average_power_mal = []
  purturbed_sequences = []
  average_power_ben = []
  for i, (x, y) in enumerate(dataloader):
    x = x.to(opts.device)
    r, log, total_purturbs, purturbed_sequence, mean_power_benign = run_env(agent, x, opts)
    purturbed_sequences.append(torch.tensor(purturbed_sequence))
    average_power_mal.append(-r.mean().item())
    average_power_ben.append(-mean_power_benign.item())
    r = r + total_purturbs.squeeze(1) * opts.gamma * opts.battery_capacity
    average_reward.append(-r.mean().item())
  concat = torch.stack(purturbed_sequences, dim=0)
  return (
    np.array(average_reward).mean(), 
    torch.cat((x[-2, :].unsqueeze(0), concat.reshape(concat.size(0)*concat.size(1), -1)), dim=0), 
    np.array(average_power_mal).mean(),
    np.array(average_power_ben).mean()
  )



if __name__ == "__main__":
  train(get_options())


