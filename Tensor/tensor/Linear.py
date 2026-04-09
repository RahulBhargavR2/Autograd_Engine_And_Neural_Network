import random
from operator import index

from Tensor.optimizers.Adam import Adam
from Tensor.optimizers.RMSprop import RMSProp
from Tensor.tensor.tensor import Tensor

from Tensor.optimizers.sgd import SGD, SGDMomentum
from scalar.Viizualizer import draw_dot


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
    def __init__(self,nin,nouts):
        eles = [nin] + nouts
        self.layers = []

        for i in range (len(nouts)):
            self.layers.append(Linear(eles[i],eles[i+1]))

    def __call__(self, x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = x.relu()
            else:
                x = x.sigmoid()
        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters


def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return (diff * diff).mean()


def normal():
    model = MLP(2,[4,4,1])
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
            p.grad.data = Tensor.data_like(p.shape, 0)

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


def using_sgd():
    model = MLP(2,[4,4,1])
    lr = 0.01

    X = Tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ],requires_grad=False)

    Y = Tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ],requires_grad=False)
    optimizer = Adam(model.parameters(), lr)
    batch_size = 2

    for epoch in range(10000):
        index = list(range(4))
        random.shuffle(index)
        for i in range(0,4,batch_size):
            batch_idx = index[i:i+batch_size]

            x_batch = X[batch_idx]
            y_batch = Y[batch_idx]

            y_pred = model(x_batch)
            optimizer.zero_grad()
            loss = mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        print("Predictions:", y_pred)


if __name__ == "__main__":
    # normal()
    using_sgd()
