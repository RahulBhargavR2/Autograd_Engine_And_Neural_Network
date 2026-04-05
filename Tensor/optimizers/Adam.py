from Tensor.optimizers.optimizer import Optimizer
from Tensor.tensor.tensor import Tensor


class Adam(Optimizer):
    def __init__(self,params, lr = 0.01, beta1=0.9, beta2=0.99):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.m = [Tensor.data_like(p.shape,0) for p in params]
        self.v = [Tensor.data_like(p.shape,0) for p in params]
        self.t = 0

    def step(self):
        self.t += 1

        for i,p in enumerate(self.params):
            if not p.requires_grad:
                continue

            # m_t = beat_1 * m_t-1 + (1-beat_1) * grad

            self.m[i] = Tensor.elementwise_add(
                Tensor.elementwise_mul(self.beta1, self.m[i]),
                Tensor.elementwise_mul(1-self.beta1,p.grad.data)
            )

            # v_t = beat_2 * v_t-1 + (1-beat_2) * (grad)^2
            g_2 = Tensor.elementwise_pow(p.grad.data, 2)
            self.v[i] = Tensor.elementwise_add(
                Tensor.elementwise_mul(self.beta2, self.v[i]),
                Tensor.elementwise_mul(1-self.beta2,g_2)
            )

            #bias correction
            m_hat = Tensor.elementwise_mul(self.m[i],1/(1-self.beta1 ** self.t))
            v_hat = Tensor.elementwise_mul(self.v[i],1/(1-self.beta2 ** self.t))

            # theta_t = theta_t-1 - lr * (m_t/(sqrt(v_t)+eps))
            # m already contains data about grad so we don't multiply it with g

            #(sqrt(v_t)+eps)
            sqrt_v_hat = Tensor.elementwise_add(Tensor.elementwise_pow(v_hat,0.5),self.eps)

            #(m_t/(sqrt(v_t)+eps))
            mt_sqrt = Tensor.elementwise_mul(
                m_hat,
                Tensor.elementwise_pow(sqrt_v_hat,-1)
            )

            grad_lr = Tensor.elementwise_mul(mt_sqrt,-self.lr)

            p.data = Tensor.elementwise_add(p.data,grad_lr)


