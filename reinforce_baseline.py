class ExponentialBaseline():
    def __init__(self, beta):

        self.beta = beta
        self.v = None

    def eval(self, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v
