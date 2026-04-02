import random
from Tensor.tensor.tensor import Tensor

from Tensor.optimizers.sgd import SGD

class Linear:
    # we consider columns as the neurons and rows as inputs
    def __init__(self, in_features, out_features):
        self.w = Tensor(
            [[random.uniform(-0.5, 0.5) for _ in range(out_features)] for _ in range(in_features)],
            requires_grad=True
        )
        self.b = Tensor([[random.uniform(-0.5, 0.5) for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x):
        out = x @ self.w + self.b
        return out

    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self):
        self.l1 = Linear(2, 4)
        self.l2 = Linear(4, 1)

    def __call__(self, x):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        x = x.sigmoid()
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return (diff * diff).mean()

if __name__ == "__main__":

    model = MLP()
    lr = 0.01

    X = Tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    Y = Tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    # try:
    for epoch in range(10000):

        # Forward
        y_pred = model(X)
        # print(type(y_pred),type(Y))
        loss = mse_loss(y_pred, Y)

        # Zero gradients
        # print(len(model.parameters()))
        # do not re assign a new tensor, just mutate the data
        for p in model.parameters():
            p.grad.data = Tensor.data_like(p.shape,0)

        # Backward
        # print("loss:",loss)
        loss.backward()

        # Update
        for p in model.parameters():
            # print(type(p))
            # p -= lr * p.grad
            p.data = Tensor.elementwise_add(p.data, Tensor.elementwise_mul(p.grad.data, -lr))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")
    # except Exception as e:
    #     print(e)

    print("Predictions:")
    print(model(X).data)
