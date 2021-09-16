import torch.nn as nn
import math
import torch
import numpy as np

class mal_agent4(nn.Module):
  def __init__(self, hidden_size, input_size, opts):
    super(mal_agent4, self).__init__()
    self.cur = 0
    self.alpha = None
    self.range = None
    self.opts = opts
  def forward(self, requests):
    return self.net(requests)
  
  def sample(self, requests):
    if self.cur == 0:
        self.start =  torch.tensor(np.random.randint(low=0, high=43, size=(self.opts.batch_size)))
        self.length = torch.tensor(np.random.randint(low=8, high=49, size=(self.opts.batch_size)))
    self.alpha = np.random.uniform(low=0.1, high=0.601, size=(self.opts.batch_size))
    self.mul = ((self.cur >= self.start) & (self.cur < (self.start + self.length))).float() * self.alpha
    purturbed = requests[:, 0] * self.mul
    a = purturbed - requests[:, 0]
    self.cur = (self.cur + 1) % self.opts.num_timesteps
    return a, None