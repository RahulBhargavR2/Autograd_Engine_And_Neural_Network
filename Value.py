import math

from Viizualizer import draw_dot
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backword = lambda :None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backword():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backword = _backword
        return out

    # def __sub__(self, other):
    #     out = Value(self.data - other.data, (self, other), '-')
    #     def _backword():


    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backword():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backword = _backword
        return out

    def tanh(self):
        x = self.data
        exp = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(exp,(self,),'tanh')
        def _backword():
            self.grad = (1-exp**2 )* out.grad
        out._backword = _backword

        return out

    def backword(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backword()

    def __repr__(self):
        return f"Value(data = {self.data})"






b = Value(6.881373587,label='b')
x1 = Value(2.0,label='x1')
x2 = Value(0.0,label='x2')

w1 = Value(-3.0,label='w1')
w2 = Value(1.0,label='w2')

x1w1 = x1 * w1; x1w1.label='x1w1'
x2w2 = x2 * w2; x2w2.label='x2w2'

x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1x2w2'
n = x1w1x2w2 +b; n.label='n'

o = n.tanh(); o.label='o'


o.backword()
draw_dot(o).render(view=True)