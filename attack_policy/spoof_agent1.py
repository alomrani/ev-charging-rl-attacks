import torch.nn as nn
import math
import torch
import numpy as np

class mal_agent1(nn.Module):
  def __init__(self, hidden_size, input_size, opts):
    super(mal_agent1, self).__init__()
    self.cur = 0
    self.alpha = None
    self.opts = opts
  def forward(self, requests):
    return self.net(requests)
  
  def sample(self, requests):
    if self.cur == 0:
        self.alpha = np.random.uniform(low=0.1, high=0.601, size=(self.opts.batch_size))
    purturbed = requests[:, 0] * self.alpha
    a = purturbed - requests[:, 0]
    self.cur = (self.cur + 1) % self.opts.num_timesteps
    return a, None