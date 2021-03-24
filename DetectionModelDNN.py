import torch.nn as nn

class DetectionModelDNN(nn.Module):
  def __init__(self, hidden_size, input_size, p):
    super(DetectionModelDNN, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
  def forward(self, input):
    return self.network(input)
