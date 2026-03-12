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

    #method to add 2 values
    def __add__(self, other):
        # to make sure we can even add with normal data eg a + 1
        # 1 will be converted to Value type and then added
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        #method that populates grad to its previous nodes
        def _backword():
            self.grad += 1.0 * out.grad # derivative of sum is evenly populated, multiplied with previous grad(chained)
            other.grad += 1.0 * out.grad # adding grad in case a single var is used multiple times
        out._backword = _backword
        return out

    # provides  negated value
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)



    def __mul__(self, other):
        other =  other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backword():
            self.grad += other.data * out.grad # for mul derivative is value of other locally
            other.grad += self.data * out.grad # globally need to multiple with out's grad (chain rule)
        out._backword = _backword
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def tanh(self):
        x = self.data
        exp = (math.exp(2*x)-1)/(math.exp(2*x)+1) # tanh representation in e
        out = Value(exp,(self,),'tanh')
        def _backword():
            self.grad += (1-exp**2 )* out.grad # derivative of tanh is (1-tanh^2)
        out._backword = _backword

        return out



    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')

        def _backword():
            self.grad += out.data * out.grad # derivative of e^x is e^x only, multiply to get grad globally
        out._backword = _backword
        return out

    def __pow__(self, other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),f'**{other}')

        def _backword():
            self.grad += out.data * (self.data**(other-1)) * out.grad # derivative of a^n is n*a^(n-1)

        out._backword = _backword
        return out

    # method to backpropogate
    def backword(self):
        topo = []
        visited = set()

        #using topological ordering of the di-graph
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        #backpropogate to all nodes from right to left
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