from Tensor.tensor.tensor import Tensor
from Tensor.optimizers.optimizer import Optimizer




class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data = Tensor.elementwise_add(p.data, Tensor.elementwise_mul(p.grad.data, -self.lr))


class SGD_momentum(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [Tensor.data_like(p.shape, 0) for p in params]

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
            # self.velocity[i] = self.beta * self.velocity[i] + (1 - self.beta) * p.grad
            self.velocity[i] = Tensor.elementwise_add(Tensor.elementwise_mul(self.beta, self.velocity[i]),
                                                           Tensor.elementwise_mul((1 - self.beta), p.grad.data))
            # p.data -= self.lr * p.grad

            p.data = Tensor.elementwise_add(p.data, Tensor.elementwise_mul(p.grad.data, -self.lr))