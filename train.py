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
num_epochs = 400
train_loader = DataLoader(SoCDataset(train[:, :-1], train[:, -1][:, None]), batch_size=5000, shuffle=True)
val_loader = DataLoader(SoCDataset(valid[:, :-1], valid[:, -1][:, None]), batch_size=2000, shuffle=True)

model = DetectionModelDNN(768, 48, 0.6).to(device)

optimizer = Adam(model.parameters(), lr=0.0001)
l = nn.CrossEntropyLoss()
train_l = []
val_l = []
val_ac = []
max_acc = 0.
for epoch in range(num_epochs):
  val_acc = 0.
  for x, y in train_loader:
    model.train()
    x = x.to(device=device)
    y = y.to(device=device)
    optimizer.zero_grad()
    output = model(x.float())
    train_loss = l(output, y.flatten().long())
    train_loss.backward()
    optimizer.step()
  for x, y in val_loader:
    model.eval()
    x = x.to(device=device)
    y = y.to(device=device)
    output = model(x.float()).detach()
    val_loss = l(output, y.flatten().long())
    val_acc += torch.sum((output.argmax(1) == y.squeeze(1)).float())
  val_acc = val_acc / 4000.
  if val_acc > max_acc:
    max_acc = val_acc
    torch.save(model.state_dict(), "drive/MyDrive/PEVdata/best_model.pt".format(epoch))
  train_l.append(train_loss)
  val_l.append(val_loss)
  val_ac.append(val_acc)
  print("\nEpoch {}: Train Loss : {} Val Accuracy : {}".format(epoch, train_loss, val_acc))
plt.figure(1)
line1, = plt.plot(np.arange(num_epochs), train_l)
line2, = plt.plot(np.arange(num_epochs), val_l)
plt.legend((line1, line2), ("Training Loss", "Validation Loss"))
plt.figure(2)
line2, = plt.plot(np.arange(num_epochs), val_ac)
