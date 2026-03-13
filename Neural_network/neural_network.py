#Micrograd https://youtu.be/VMj-3S1tku0?si=xEnZ7RpiVfkU1apg
#Neural networks and Backpropagation implementation

'''AI-copied node visualizer'''
def print_tree(node, prefix="", is_last=True, is_root=True):

    # 1. Format the text for the current node
    op_str = f" ({node._op})" if node._op else ""
    label_str = f"{node.label}: " if node.label else ""
    node_text = f"[Value {node.data:.4f}| grad:{node.grad:.4f}]{op_str}"

    # 2. Print the node with the correct tree branches
    if is_root:
        print(node_text)
    else:
        connector = "└── " if is_last else "├── "
        print(prefix + connector + node_text)

    # 3. Prepare the prefix for the next level down
    if not is_root:
        prefix += "    " if is_last else "│   "

    # 4. Recursively call this function on all children
    children = list(node._prev)
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_tree(child, prefix, is_last_child, is_root=False)
''''''''''''''''''''''''''''''

from math import exp, log
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        def backward():
            self.grad += (other.data * (self.data**(other.data-1))) * out.grad
            other.grad += (out.data * log(abs(self.data))) * out.grad
        out._backward = backward
        return out
    def __rpow__(self, other):
        return Value(other) ** self

    def __truediv__(self, other):
        return self * other ** -1
    def __rtruediv__(self, other):
        return Value(other)/self

    def exp(self):
        out = Value(exp(self.data), (self,) , 'exp')
        def backward():
            self.grad += out.data * out.grad
        out._backward = backward
        return out

    def sigm(self):
        out = Value(1/(1+exp(-self.data)), (self,) , 'sigm')
        def backward():
            self.grad += out.data*(1-out.data) * out.grad
        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



    def __repr__(self):
        return f'{self.label}: data={self.data}'

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((w1*x1 for w1, x1 in zip(self.w,x)), self.b)
        out = act.sigm()
        return out
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]
    def __call__(self, x):
        for layer in self.layers:
            x  = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def forward(self, data, results):
        outs = []
        for d in data:
            outs.extend(self(d))
        return sum([(out-result)**2 for out, result in zip(outs, results)])

    def backward(self, l:Value, step=0.05):
        l.backward()
        for p in self.parameters():
            p.data += -step*p.grad



dataset =  [[ 1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [-1.0,  1.0,  1.0],
            [-1.0, -1.0, -1.0]]
results = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
x = [[Value(n) for n in sample] for sample in dataset]


n = MLP(3, [4,4,1])
epochs = 10000
learning_rate = 0.05

print(f'Training... ({epochs} epochs)')
for i in range(epochs):
    L = n.forward(x, results)
    n.zero_grad()
    n.backward(L, step=0.05)
    if i % 100 == 0:
        print(f'- Epoch {i}: Loss {L.data:.4f}')

print(f'Final Loss: {L.data:.8f}')
print()

for sample, target in zip(x, results):
    prediction = n(sample)[0].data
    print(f"Target: {target}; Prediction: {prediction:.4f}")