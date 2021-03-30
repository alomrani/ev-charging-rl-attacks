import torch
import numpy as np


class charging_ev():
  def __init__(self, opts):
    self.num_cars = opts.num_cars
    self.time = opts.num_timesteps
    self.cur = torch.ones(opts.batch_size, device=opts.device) * opts.total_power
    self.allocated_power = torch.zeros(opts.batch_size, opts.num_cars, device=opts.device)
    self.history = torch.zeros(opts.batch_size, opts.num_cars, device=opts.device)
    self.battery_capacity = opts.battery_capacity
    self.total_power = opts.total_power
    self.timestep = 0
    self.opts = opts
    self.batch_size = opts.batch_size
    self.device = opts.device
    self.epsilon = opts.epsilon
  def finished(self):
    return self.timestep == self.time

  def step(self, requests):

    self.cur = torch.ones(self.batch_size, device=self.device) * self.total_power
    pi = (1. - requests) * self.epsilon + (1. - self.epsilon) * 0.2

    p = (1. - requests) * self.battery_capacity

    A = pi / p
    A, I = torch.sort(A, descending=True)
    for i in range(len(A[0])):
      power = p.gather(1, I[:, i].unsqueeze(1))
      added_power = torch.nn.functional.one_hot(I[:, i], num_classes=self.num_cars) * power
      check = (added_power <= self.cur.unsqueeze(1)).float()
      added_power = check * added_power
      self.allocated_power += added_power
      self.cur -= power.squeeze(1)

    power = p.gather(1, I[:, 0].unsqueeze(1))
    added_power = torch.nn.functional.one_hot(I[:, 0], num_classes=self.num_cars) * power
    self.allocated_power += added_power
    self.timestep += 1
    return self.allocated_power
