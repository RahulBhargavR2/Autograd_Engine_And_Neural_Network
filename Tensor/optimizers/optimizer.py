from Tensor.tensor.tensor import Tensor
class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data = Tensor.data_like(p.shape, 0)

    def step(self):
        raise NotImplementedError