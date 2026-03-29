import random
from Tensor import Tensor

class Linear:
    # we consider columns as the neurons and rows as inputs
    def __init__(self,in_features, out_features):
        self.w = Tensor(
            [[random.uniform(-1,1) for _ in range(out_features)] for _ in range(in_features)],
            requires_grad=True
        )
        self.b = Tensor([random.uniform(-1,1) for _ in range(out_features)],requires_grad=True)
    def __call__(self, x):
        out = x @ self.w + self.b
        out.relu()
        return out

    def parameters(self):
        return [self.w, self.b]

# x = Tensor([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0]
# ])  # (2, 3)
#
# print(x.data)
# #
# layer = Linear(3, 2)
# print(layer.w)
# print(layer.b)
# out = layer(x)
#
# print(out.shape)  # (2, 2)