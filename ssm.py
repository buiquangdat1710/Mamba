import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from p_scan import pscan


def selective_scan(x, delta, A, B, C, D):
    """
    Perform selective scan operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape # L = 64

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (1,64,256,256)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (1,64,256,256)

    BX = deltaB * x.unsqueeze(-1)  # (1,64,256,256)


    hs = pscan(deltaA, BX) #(1,64,256,256)

    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    """
    Perform selective scan sequence operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).
        dim_inner (int): Inner dimension size.
        d_state (int): State dimension size.

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank # 8
        self.dim_inner = dim_inner # 256
        self.d_state = d_state # 256

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        ) # (256, 8 + 2*256) = (256, 520)
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True) # (8, 256)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float()) # [256,256]
        D = self.D.float() # [256]
        deltaBC = self.deltaBC_layer(x) # [1, 64, 520]
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        ) # delta: [1, 64, 8], B: [1, 64, 256], C: [1, 64, 256]
        delta = F.softplus(self.dt_proj_layer(delta)) # [1, 64, 256]

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            # A: [256,256], B: [1, 64, 256], C: [1, 64, 256], D: [256], delta: [1, 64, 256]
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y
    

# x = torch.randn(1, 64, 256)
# model = SSM(in_features=256, dt_rank=8, dim_inner=256, d_state=256)
# out = model(x)

