class SoCDataset(Dataset):
  def __init__(self, soc, labels):
    self.soc = soc
    self.labels = labels
    self.len = len(soc)
  def __len__(self):
    return self.len
  def __getitem__(self, idx):
    return self.soc[idx, :], self.labels[idx, :]
