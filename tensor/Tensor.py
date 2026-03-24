from scalar.Viizualizer import draw_dot


class Tensor:
    def __init__(self, x, _children=(), _op='', label='', requires_grad=True):
        self.data = x
        self.shape = Tensor.get_shape(self.data)
        # use normal list for grad to avoid infinite recursion while creating tensor
        self.grad = Tensor.data_like(self.shape, 0) if requires_grad else None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.requires_grad = requires_grad
        self._backward = lambda: None

    def __repr__(self):
        return f"{self.data}"

    # helper methods

    @staticmethod
    def get_shape(data):
        if isinstance(data, list):
            # if isinstance(data[0], list):
            #     currL = len(data[0])
            #     print(currL)
            #     for arr in data:
            #         if len(arr) != currL:
            #             raise Exception('Invalid shape')

            return (len(data),) + Tensor.get_shape(data[0])
        else:
            return ()

    # returns the 1D representation of the tensor
    @staticmethod
    def flatten(data):
        if not isinstance(data, list):
            return [data]

        result = []
        for item in data:
            result.extend(Tensor.flatten(item))
        return result

    @staticmethod
    def reshape(data, shape):
        if len(shape) == 0:
            return data[0]

        size = shape[0]
        step = len(data) // size

        return [Tensor.reshape(data[i * step:(i + 1 )* step], shape[1:]) for i in range(size)]

    # main reshape method
    def reshape_tensor(self, new_shape):
        flat = Tensor.flatten(self.data)
        total = 1
        for ele in new_shape:
            total *= ele
        if total != len(flat):
            raise Exception("Invalid shape")
        return Tensor.reshape(flat, new_shape)

    @staticmethod
    def data_like(shape, fill):
        if len(shape) == 0:
            return fill
        return [Tensor.data_like(shape[1:], fill) for i in range(shape[0])]

    @staticmethod
    def zeros_like(data):
        shape = data.shape
        return Tensor(Tensor.data_like(shape, 0))

    @staticmethod
    def ones_like(data):
        shape = data.shape
        return Tensor(Tensor.data_like(shape, 1))

    @staticmethod
    def ones(data):
        shape = data.shape
        return Tensor.data_like(shape, 1)

    @staticmethod
    def broadcast_shape(shape1, shape2):
        result = []

        i, j = len(shape1) - 1, len(shape2) - 1

        while i >= 0 or j >= 0:
            dim1 = shape1[i] if i >= 0 else 1
            dim2 = shape2[j] if j >= 0 else 1

            if dim1 == dim2:
                result.append(dim1)
            elif dim1 == 1:
                result.append(dim2)
            elif dim2 == 1:
                result.append(dim1)
            else:
                raise Exception("Shape not broadcastable")

            i -= 1
            j -= 1

        return tuple(reversed(result))

    @staticmethod
    def broadcast_to(data, target_shape):
        current_shape = Tensor.get_shape(data)

        # prepend 1s to match dims
        while len(current_shape) < len(target_shape):
            data = [data]
            current_shape = (1,) + current_shape

        out = Tensor._broadcast_recursive(data, current_shape, target_shape)

        return out

    @staticmethod
    def _broadcast_recursive(data, curr_shape, target_shape):
        if curr_shape == target_shape:
            return data

        if len(curr_shape) == 1:
            if curr_shape[0] == 1:
                return [Tensor._broadcast_recursive(data[0], (), target_shape[1:]) for _ in range(target_shape[0])]

        result = []
        for i in range(target_shape[0]):
            idx = 0 if curr_shape[0] == 1 else i
            result.append(
                Tensor._broadcast_recursive(
                    data[idx],
                    curr_shape[1:],
                    target_shape[1:]
                )
            )

        return result

    @staticmethod
    def unbroadcast(grad, original_shape):
        grad_shape = Tensor.get_shape(grad)

        # remove extra dimensions
        while len(grad_shape) > len(original_shape):
            grad = Tensor.sum_axis(grad, axis=0)
            grad_shape = Tensor.get_shape(grad)

        # collapse broadcasted dims
        for i, dim in enumerate(original_shape):
            if dim == 1:
                grad = Tensor.sum_axis(grad, axis=i)

        return grad

    @staticmethod
    def sum_axis(data, axis):
        # Base case: axis = 0 → sum current level
        if axis == 0:
            if not isinstance(data[0], list):
                return sum(data)

            result = data[0]
            for i in range(1, len(data)):
                result = Tensor.elementwise_add(result, data[i])

            return result

        # Recursive case: go deeper
        return [Tensor.sum_axis(sub, axis - 1) for sub in data]

    @staticmethod
    def elementwise_add(a, b):
        if not isinstance(a, list):
            return a + b

        return [Tensor.elementwise_add(x, y) for x, y in zip(a, b)]

    # instance methods

    def backward(self):
        if self.grad is  None:
            self.grad = Tensor.ones(self)

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # 1. compute broadcast shape
        target_shape = Tensor.broadcast_shape(self.shape, other.shape)

        # 2. broadcast data
        a = Tensor.broadcast_to(self.data, target_shape)
        b = Tensor.broadcast_to(other.data, target_shape)

        # 3. forward computation
        out_data = Tensor.elementwise_add(a, b)
        out = Tensor(out_data, (self, other), '+')

        # 4. backward function
        def _backward():
            if self.grad is not None:
                grad_self = Tensor.unbroadcast(out.grad, self.shape)
                self.grad = Tensor.elementwise_add(self.grad, grad_self)

            if other.grad is not None:
                grad_other = Tensor.unbroadcast(out.grad, other.shape)
                other.grad = Tensor.elementwise_add(other.grad, grad_other)

        out._backward = _backward
        return out

    @staticmethod
    def elementwise_mul(a, b):
        if not isinstance(a, list):
            return a * b
        return [Tensor.elementwise_mul(x, y) for x, y in zip(a, b)]

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # 1 get broadcast shape
        target_shape = Tensor.broadcast_shape(self.shape, other.shape)

        self_a = Tensor.broadcast_to(self.data, target_shape)
        other_b = Tensor.broadcast_to(other.data, target_shape)

        out = Tensor.elementwise_mul(self_a, other_b)
        out = Tensor(out, (self, other), '*')

        def _backward():
            if self.grad is not None:
                grad = Tensor.elementwise_mul(other_b, out.grad)
                grad_self = Tensor.unbroadcast(grad, self.shape)
                self.grad = Tensor.elementwise_add(self.grad, grad_self)

            if other.grad is not None:
                grad = Tensor.elementwise_mul(self_a, out.grad)
                grad_other = Tensor.unbroadcast(grad, other.shape)
                other.grad = Tensor.elementwise_add(other.grad, grad_other)

        out._backward = _backward
        return out

    # matmul
    @staticmethod
    def transpose(matrix):
        return list(map(list, zip(*matrix)))

    @staticmethod
    def matmul(matrix_a, matrix_b):
        result = []
        for i in range(len(matrix_a)):
            row = []
            for j in range(len(matrix_b[0])):
                total = 0
                for k in range(len(matrix_a[0])):
                    total += matrix_a[i][k] * matrix_b[k][j]
                row.append(total)
            result.append(row)
        return result

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape[1] != other.shape[0]:
            raise Exception("Shapes not compatible for matmul")

        a = self.data
        b = other.data
        out = Tensor.matmul(a, b)
        out = Tensor(out, (self, other), '@')

        def _backward():
            if self.grad is not None:
                bt = Tensor.transpose(b)
                grad_self = Tensor.matmul(out.grad, bt)
                self.grad = Tensor.elementwise_add(self.grad, grad_self)

            if other.grad is not None:
                at = Tensor.transpose(a)
                grad_other = Tensor.matmul(at, out.grad)
                other.grad = Tensor.elementwise_add(other.grad, grad_other)

        out._backward = _backward
        return out
