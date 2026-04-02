from Tensor.tensor.tensor import Tensor
class Optimizer:
    def __init__(self,params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = Tensor.zeros_like(p)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self,params,lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data -= self.lr * p.grad


class SGD_momentum(Optimizer):
    def __init__(self,params,lr=0.01,beta=0.9):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [p.zeros_like() for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue

            # classical
            # vt= βvt−1 + gt
            # θ = θ−ηvt

            # smoothed SGD(EMA - based)
            # vt=βvt−1 + (1−β) gt
            # θ = θ−ηvt
            self.velocity[i] = self.beta * self.velocity[i] + (1 - self.beta) * p.grad
            p.data -= self.lr * p.grad

