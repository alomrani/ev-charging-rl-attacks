import torch
import numpy as np


class charging_ev():
  def __init__(self, num_cars, time, total_power, epsilon, battery_capacity, device, batch_size):
    self.num_cars = num_cars
    self.time = time
    self.cur = torch.ones(batch_size, device=device) * total_power
    self.allocated_power = torch.zeros(batch_size, num_cars, device=device)
    self.history = torch.zeros(batch_size, num_cars, device=device)
    self.battery_capacity = battery_capacity
    self.total_power = total_power
    self.timestep = 0

  def finished(self):
    return self.timestep == self.time

  def step(self, requests):
    self.cur = torch.ones(batch_size, device=device) * total_power
    pi = (1. - requests) * epsilon + (1. - epsilon) * 0.2

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
