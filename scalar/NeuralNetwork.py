from Value import Value
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='bias')

    def __call__(self, x,nonlin=True):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu() if nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout,nonlin=True):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.nonlin = nonlin

    def __call__(self, x):
        out = [n(x,self.nonlin) for n in self.neurons]
        # for n in self.neurons:
        #     out.append(n(x,self.nonlin))
        return out[0] if len(out) == 1 else out

    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters


class MLP:
    def __init__(self, nin, nouts):
        eles = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            # linear only for hidden layers and not last layer, last layer should be of nay value including -ve
            # if relu applied then the op will no be proper
            nonlin = ( i != len(nouts) - 1)
            self.layers.append(Layer(eles[i], nouts[i],nonlin))
        # self.layers = [Layer(eles[i], eles[i + 1],) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())

        return parameters

#
# xs = [
#     [2.0, 3.0, -1.0],
#     [3.0, -1.0, 0.5],
#     [0.5, 1.0, 0.1],
#     [1.0, -1.0, -1.0]
# ]
#
# ys = [1.0, -1.0, -1.0, 1.0]

xs = [[i/10.0] for i in range(1,11)]
ys = [(i/10.0)**2 for i in range(1,11)]


n = MLP(1,[4,4,1])
ypred = [n(x) for x in xs]




loss = Value(0)
for k in range(1000):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((x-y)**2 for x,y in zip(ypred,ys)))

#     backward pass
#     never forger to reset the grad, else it will addup to previous passes and messes up the calculation
    for p in n.parameters():
        p.grad = 0
    loss.backward()

    #update the weight in the direction of the descent
    for p in n.parameters():
        p.data += -0.02*p.grad
    if k%1000 == 0:
        print(k,loss)
# print(loss)
# draw_dot(loss).render(view=True)

while True:
    val = int(input('Enter a number: '))
    op = n([val/10.0])
    print(op.data*100)