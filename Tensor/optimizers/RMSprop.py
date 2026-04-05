from Tensor.optimizers.optimizer import Optimizer
from Tensor.tensor.tensor import Tensor


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params)
        self.beta = beta
        self.lr = lr
        self.eps = 1e-8
        self.velocity = [Tensor.data_like(p.shape, 0) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue

            # velocity_t = beta * velocity_t-1 + (1-beta) * (grad_t)^2
            mul_vel_beta = Tensor.elementwise_mul(self.beta,self.velocity[i])  # beta * velocity_t-1

            grad_t_square =  Tensor.elementwise_pow(p.grad.data, 2) # (grad_t)^2

            mul_beta = Tensor.elementwise_mul(1 - self.beta, grad_t_square) # (1-beta) * (grad_t)^2

            self.velocity[i] = Tensor.elementwise_add( mul_vel_beta,mul_beta) # beta * velocity_t-1 + (1-beta) * (grad_t)^2

            # parameter theta
            # theta = theta - lr * 1/(sqrt(velocity_t) + e) * grad_t

            sqrt_vel = Tensor.elementwise_pow(self.velocity[i], 0.5)  #sqrt(velocity_t)

            sqrt_vel_e = Tensor.elementwise_add(sqrt_vel,self.eps) #(sqrt(velocity_t) + e)

            sqrt_vel_e_div = Tensor.elementwise_pow(sqrt_vel_e,-1)#1/(sqrt(velocity_t) + e)

            mul_with_grad = Tensor.elementwise_mul(sqrt_vel_e_div,p.grad.data) #1/(sqrt(velocity_t) + e) * grad_t

            mul_lr = Tensor.elementwise_mul(-self.lr,mul_with_grad) # - lr * 1/(sqrt(velocity_t) + e) * grad_t

            p.data = Tensor.elementwise_add(p.data,mul_lr)   # theta = theta - lr * 1/(sqrt(velocity_t) + e) * grad_t

