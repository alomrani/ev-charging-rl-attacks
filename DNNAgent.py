import torch.nn as nn
import math

class mal_agent(nn.Module):
  def __init__(self, hidden_size, input_size):
    super(mal_agent, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
    self.init_parameters()
  def init_parameters(self):
    for name, param in self.named_parameters():
      stdv = 1. / math.sqrt(param.size(-1))
      param.data.uniform_(-stdv, stdv)
  def forward(self, requests):
    return self.net(requests)
  
  def sample(self, requests):
    out = self.net(requests)
    mu, sigma = out[:, 0].float(), torch.abs(out[:, 1]) + 1e-5
    n = torch.distributions.Normal(mu.detach(), sigma.detach())
    a = n.rsample().detach()
    p = torch.exp(-0.5 *((a - mu) / (sigma))**2) * 1 / (sigma * np.sqrt(2 * np.pi))
    log_p = torch.log(p + 1e-5)
    return a, log_p
