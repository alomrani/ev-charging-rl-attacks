import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
# from imblearn.over_sampling import ADASYN
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.optim import Adam
from options1 import get_options
from soc_dataset import SoCDataset
from reinforce_baseline import ExponentialBaseline
from attack_policy.DNNAgent import mal_rl_agent
from attack_policy.spoof_agent1 import mal_agent1
from attack_policy.spoof_agent2 import mal_agent2
from attack_policy.spoof_agent3 import mal_agent3
from attack_policy.spoof_agent4 import mal_agent4
from charging_env import charging_ev
from DetectionModelDNN import DetectionModelDNN
import os
from itertools import product
import json
import seaborn as sns
import matplotlib.patches as mpatches

def train_dnn(opts):
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
  test_dataset = torch.load(opts.test_dataset)

#  val_dataset = val_dataset.reshape(val_dataset.size(0) * val_dataset.size(1), -1)


  train_loader = DataLoader(SoCDataset(train_dataset[:, :-1], train_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  test_loader = DataLoader(SoCDataset(test_dataset[:, :-1], test_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  print(train_dataset.shape)
  
  if opts.eval_only:
    model = DetectionModelDNN(opts.hidden_size, opts.num_timesteps, opts.p).to(opts.device)

    if opts.load_path is not None:
      load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
      model.load_state_dict(load_data)
    loss = nn.CrossEntropyLoss()
    val_acc, val_loss = eval(model, test_loader, loss, opts)
    print(val_acc)
  elif opts.train_plots:
    colors = sns.color_palette()
    models = ['Model 1', 'Model 2', 'Model 3']
    plt.figure(1)
    sns.set(style="darkgrid")
    plots = []
    legend_patches = []
    for i, model in enumerate(models):
      model_train_loss = np.load(f"train_loss_{model[-1]}.npy")
    # plt.title(f"Num Cars {opts.num_cars} arrival rate : {opts.lamb}")
      line = sns.tsplot(data=np.array(model_train_loss), color=colors[i])
      patch = mpatches.Patch(color=colors[i], label=model)
      legend_patches.append(patch)
    plt.legend(handles=legend_patches, title="Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.savefig(opts.save_dir + "/training_loss.pdf", dpi=400)

    plt.figure(2)

    legend_patches = []
    for i, model in enumerate(models):
      model_train_loss = np.load(f"val_acc_{model[-1]}.npy")
    # plt.title(f"Num Cars {opts.num_cars} arrival rate : {opts.lamb}")
      line = sns.tsplot(data=np.array(model_train_loss), color=colors[i])
      patch = mpatches.Patch(color=colors[i], label=model)
      legend_patches.append(patch)
    plt.legend(handles=legend_patches, title="Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Accuracy")

    plt.savefig(opts.save_dir + "/validation_accuracy.pdf", dpi=400)

  elif opts.tune:
    PARAM_GRID = list(product(
            [0.01, 0.001, 0.0001, 0.00001, 0.02, 0.002, 0.0002, 0.00002, 0.03, 0.003, 0.0003, 0.00003, 0.004, 0.0004, 0.00004],  # learning_rate
            [0., 0.5, 0.6, 0.7, 0.75, 0.8, 0.85],  # dropout rate
            [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]  # lr decay
        ))
    # total number of slurm workers detected
    # defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # this worker's array index. Assumes slurm array job is zero-indexed
    # defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SCOREFILE = os.path.expanduser(f"./val_acc_with_syn.csv")
    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
      torch.manual_seed(opts.seed)
      np.random.seed(opts.seed)
      params = PARAM_GRID[param_ix]
      opts.p = params[1]
      opts.lr_model = params[0]
      opts.lr_decay = params[2]

      model, *_ = train_epoch(train_loader, val_loader, opts)
      val_acc, val_loss = eval(model, val_loader, nn.CrossEntropyLoss(), opts)
      # if avg_r > max_val:
      #   best_params = params
      #   max_val = avg_r

      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (val_acc,)))}\n')
  else:
    seeds = [200, 890, 786, 4872]
    best_val_acc = 0
    all_train_l, all_val_l, all_val_ac = [], [], []
    for seed in seeds:
      curr_model, curr_train_l, curr_val_l, curr_val_ac, max_val_acc = train_epoch(train_loader, val_loader, opts, seed=seed)

      all_train_l.append(curr_train_l)
      all_val_l.append(curr_val_l)
      all_val_ac.append(curr_val_ac)
      if max_val_acc > best_val_acc:
        best_model = curr_model
        best_val_acc = max_val_acc

        best_train_l, best_val_l, best_val_ac = curr_train_l, curr_val_l, curr_val_ac
        torch.save(best_model.state_dict(), opts.save_dir + "/best_model_overall.pt")
        
        print("Cur Best Val Acc: ", best_val_acc)

    sns.set_style("darkgrid")
    plt.figure(1)
    line1, *_ = plt.plot(np.arange(opts.n_epochs), np.array(best_train_l))
    line2, *_ = plt.plot(np.arange(opts.n_epochs), np.array(best_val_l))
    plt.title("Loss During Training")
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.legend((line1, line2), ("Training Loss", "Validation Loss"))
    plt.savefig(opts.save_dir + "/train_loss.pdf", dpi=1200)
    plt.figure(2)
    line2, *_ = plt.plot(np.arange(opts.n_epochs), best_val_ac)
    plt.title("Accuracy During Training")
    plt.xlabel("Batch Step")
    plt.ylabel("Accuracy")
    plt.savefig(opts.save_dir + "/val_acc.pdf", dpi=1200)
    print(np.array(all_val_ac).shape)
    np.save(opts.save_dir + "/val_acc.npy", np.array(all_val_ac))
    np.save(opts.save_dir + "/val_loss.npy", np.array(all_val_l))
    np.save(opts.save_dir + "/train_loss.npy", np.array(all_train_l))



def train_epoch(train_loader, val_loader, opts, seed=None):
  if seed is not None:
    torch.random.manual_seed(seed)
    np.random.seed(seed)
  model = DetectionModelDNN(opts.hidden_size, opts.num_timesteps, opts.p).to(opts.device)

  optimizer = Adam(model.parameters(), lr=opts.lr_model)
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: opts.lr_decay ** epoch
  )
  l = nn.CrossEntropyLoss()
  train_l = []
  val_l = []
  val_ac = []
  max_acc = 0.
  val_acc = 0
  val_loss = 0
  for epoch in range(opts.n_epochs):
    val_acc = 0.
    for x, y in tqdm(train_loader):
      model.train()
      x = x.to(device=opts.device)
      y = y.to(device=opts.device)
      optimizer.zero_grad()
      output = model(x.float())
      train_loss = l(output, y.flatten().long())
      train_loss.backward()
      optimizer.step()
    if not opts.tune:
      val_acc, val_loss = eval(model, val_loader, l, opts)
    if val_acc > max_acc and not opts.tune:
      max_acc = val_acc
      torch.save(model.state_dict(), opts.save_dir + "/best_model.pt".format(epoch))
    train_l.append(train_loss.detach().item())
    val_l.append(torch.tensor(val_loss).mean().item())
    val_ac.append(val_acc)
    print("\nEpoch {}: Train Loss : {} Val Accuracy : {}".format(epoch, train_loss, val_acc))
    lr_scheduler.step()
  return model, train_l, val_l, val_ac, max_acc


def eval(model, val_loader, loss, opts):
  total = 0
  val_acc = 0
  val_loss = []
  for x, y in tqdm(val_loader):
    model.eval()
    x = x.to(device=opts.device)
    y = y.to(device=opts.device)
    output = model(x.float()).detach()
    val_l = loss(output, y.flatten().long())
    val_acc += torch.sum((output.argmax(1) == y.squeeze(1)).float()).item()
    total += x.size(0)
    val_loss.append(val_l)
  print(total)
  val_acc = val_acc / opts.val_size
  return val_acc, val_loss




if __name__ == "__main__":
    train_dnn(get_options())
