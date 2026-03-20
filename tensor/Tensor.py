import numpy as np


class Tensor:
    # def __init__(self, x, _children=(), _op='', label='', requires_grad=True):
    #     self.data = np.array(x, dtype=float)
    #     self.grad = np.zeros_like(self.data) if requires_grad else None
    #     self._prev = set(_children)
    #     self._backword = lambda: None
    #     self.shape = self.data.shape
    #
    #     self._op = _op
    #     self.label = label

    def __init__(self, x, _children=(), _op='', label='', requires_grad=True):
        self.data = [x] if not isinstance(x, list) else x
        self.shape = Tensor.get_shape(self.data)
        self.grad = self.zeros_like() if requires_grad else None
        self.children = set(_children)
        self.op = _op
        self.label = label
        self.requires_grad = requires_grad


    # def __add__(self, other):
    #     out = Tensor(self.data + other.data)

    @staticmethod
    def get_shape( data):
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
    def reshape(data,shape):
        if len(shape) == 0:
            return data[0]

        size = shape[0]
        step = len(data) // size

        return [ Tensor.reshape(data[i*step:(i+1*step)],shape[1:]) for i in range(size)]

    def reshape_tensor(self,new_shape):
        flat = Tensor.flatten(self.data)
        total = 0
        for ele in new_shape:
            total *= ele
        if ele != len(flat):
            raise Exception("Invalid shape")
        return Tensor.reshape(flat,new_shape)





    def zeros_like(self):
        shape = self.shape
        data = 0
        for ele in reversed(shape):
            data = [data] * ele
        return data





# t = Tensor([
#     [[[1, 2, 3], [2, 5, 6], [4, 3, 3]], [[1, 2, 3], [2, 5, 6], [4, 3, 3]], [[1, 2, 3], [2, 5, 6], [4, 3,1 ]]],
#             [[[1, 2, 3], [2, 5, 6], [4, 3,1]], [[1, 2, 3], [2, 5, 6], [4, 3,2 ]], [[1, 2, 3], [2, 5, 6], [4, 3,4 ]]]])
# print(t.data[0][0][0][0])
t = Tensor([[1,2,1],[1,2,1],[1,1,2]])

# print(t.shape)
# t.zeros_like()
# print(t.grad)
print(t.flatten(t.data))
