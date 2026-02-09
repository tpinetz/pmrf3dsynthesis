import torch
import numpy as np

def PatchWiseWasserSteinSinkhorn(img1: torch.Tensor,
                                 img2: torch.Tensor,
                                 window_size: int,
                                 stride: int,
                                 num_iterations: int = 100,
                                 eps: float = 1.,
                                 shift: bool = False) -> torch.Tensor:
    if shift:
        roll1 = np.random.choice((window_size // 2) + 1)
        roll2 = np.random.choice((window_size // 2) + 1)
        roll3 = np.random.choice((window_size // 2) + 1)

        img1 = torch.roll(img1, shifts=(roll1, roll2, roll3), dims=(0,1,2))
        # roll1 = np.random.choice((window_size // 2) + 1)
        # roll2 = np.random.choice((window_size // 2) + 1)
        # roll3 = np.random.choice((window_size // 2) + 1)
        img2 = torch.roll(img2, shifts=(roll1, roll2, roll3), dims=(0,1,2))

    x = img1.unfold(0, window_size, stride)
    x = x.unfold(1, window_size, stride)
    x = x.unfold(2, window_size, stride)

    y = img2.unfold(0, window_size, stride)
    y = y.unfold(1, window_size, stride)
    y = y.unfold(2, window_size, stride)

    x = x.reshape(-1, window_size**3)
    y = y.reshape(-1, window_size**3)

    # C_res = torch.nn.functional.pairwise_distance(x[None, ...], y[:,None], p=2)
    C_res = (x ** 2).sum(1).view(-1, 1) + (y ** 2).sum(1).view(-1, 1) - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    with torch.no_grad():
        Q = C_res.detach()
        eps = torch.max(torch.max(Q, -1, keepdim=True)[0], -2, keepdim=True)[0] * 10
        Q = torch.exp(-Q / eps)
        b = torch.ones((Q.shape[0], 1)).type_as(Q)
        T = 1
        for _ in range(num_iterations):
            K = T * Q
            a = 1 / window_size**3 / torch.matmul(K, b)
            b = 1 / window_size**3 / torch.matmul(K.transpose(1,0), a)
            T = a * K * b.transpose(1,0)

    return torch.sum((T * C_res).sum())