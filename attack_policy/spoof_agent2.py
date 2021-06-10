import torch.nn as nn
import math
import torch
import numpy as np

class mal_agent2(nn.Module):
  def __init__(self, hidden_size, input_size, opts):
    super(mal_agent2, self).__init__()
    self.opts = opts
  def forward(self, requests):
    return self.sample(requests)
  
  def sample(self, requests):
    alpha = np.random.uniform(low=0.1, high=0.601, size=(self.opts.batch_size))
    purturbed = requests[:, 0] * alpha
    a = purturbed - requests[:, 0]
    return a, None