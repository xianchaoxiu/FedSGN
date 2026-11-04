import torch
from typing import List, Tuple




def build_beta_summaries(
    Z_list: List[torch.Tensor],
    beta_list: List
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = Z_list[0].device
    dtype = Z_list[0].dtype

    beta_tensor = torch.stack([
        torch.as_tensor(b, device=device, dtype=dtype)
        for b in beta_list
    ])

    beta_sum = beta_tensor.sum()

    Z_agg = torch.zeros_like(Z_list[0])

    for Z_i, beta_i in zip(Z_list, beta_tensor):
        Z_agg += beta_i * Z_i

    return beta_sum, Z_agg


def objective(
    Y: torch.Tensor,
    Z_list: List[torch.Tensor],
    beta_list: List,
    mu: float
) -> torch.Tensor:

    p = Y.shape[1]

    I = torch.eye(
        p,
        device=Y.device,
        dtype=Y.dtype
    )

    # orthogonality penalty
    obj = (
        mu / 4.0
        * torch.norm(
            Y.T @ Y - I,
            p='fro'
        ) ** 2
    )

    # consensus term
    for Z_i, beta_i in zip(Z_list, beta_list):

        beta_i = torch.as_tensor(
            beta_i,
            device=Y.device,
            dtype=Y.dtype
        )

        obj += (
            beta_i / 2.0
            * torch.norm(
                Y - Z_i,
                p='fro'
            ) ** 2
        )

    return obj


def grad_Y(
    Y: torch.Tensor,
    beta_sum: torch.Tensor,
    Z_agg: torch.Tensor,
    mu: float
) -> torch.Tensor:

    p = Y.shape[1]

    I = torch.eye(
        p,
        device=Y.device,
        dtype=Y.dtype
    )

    # ∇h(Y)
    #
    # = beta_sum Y - Z_agg
    # + mu Y(Y^T Y - I)

    grad = (
        beta_sum * Y
        - Z_agg
        + mu * Y @ (Y.T @ Y - I)
    )

    return grad


@torch.no_grad()
def select_row_support(
    Y: torch.Tensor,
    grad: torch.Tensor,
    alpha: float,
    s: int
) -> torch.Tensor:

    # support is selected from
    #
    # Y - alpha * grad h(Y)

    V = Y - alpha * grad

    # row-wise l2 norm
    row_norms = torch.norm(
        V,
        p=2,
        dim=1
    )

    s = min(
        int(s),
        Y.shape[0]
    )

    support = torch.topk(
        row_norms,
        k=s,
        largest=True,
        sorted=False
    ).indices

    return support


def sparse_newton_step(
    Y: torch.Tensor,
    Z_list: List[torch.Tensor],
    beta_list: List,
    k: int,
    mu: float,
    alpha_support: float = 1.0,
    armijo_c: float = 1e-4,
    armijo_tau: float = 0.5,
    max_backtrack: int = 20,
    hessian_reg: float = 1e-6
):

    """
    FedSGN sparse Newton update for Y.

    k:
        number of selected original features,
        i.e. ||Y||_{2,0} <= k
    """

    device = Y.device
    dtype = Y.dtype

    # ========================================================
    # 1. Aggregate beta_i and beta_i Z_i
    # ========================================================

    beta_sum, Z_agg = build_beta_summaries(
        Z_list,
        beta_list
    )


    # ========================================================
    # 2. Gradient at current Y
    # ========================================================

    G = grad_Y(
        Y,
        beta_sum,
        Z_agg,
        mu
    )


    # ========================================================
    # 3. Select row support T
    #
    # T = top-s rows of
    #     || (Y - alpha * grad h(Y))_{j,:} ||_2
    # ========================================================

    support = select_row_support(
        Y,
        G,
        alpha_support,
        k
    )

    s = support.numel()
    p = Y.shape[1]


    # ========================================================
    # 4. Work at the current iterate.  For the active Newton block, keep
    # inactive rows fixed while differentiating; the full direction below
    # sets D[T^c] = -Y[T^c], exactly as Algorithm 1 in the manuscript.
    # ========================================================

    Y_base = Y.detach().clone()


    # ========================================================
    # 5. Reduced Newton problem
    #
    # Only optimize Y_T.
    # Hessian is calculated on all active variables,
    # so cross-Hessian terms are preserved.
    # ========================================================

    x0 = (
        Y_base[support]
        .clone()
        .detach()
        .reshape(-1)
        .requires_grad_(True)
    )


    def reduced_objective(x):

        active_rows = x.reshape(s, p)

        Y_tmp = Y_base.clone()
        Y_tmp = Y_tmp.index_copy(
            0,
            support,
            active_rows
        )

        return objective(
            Y_tmp,
            Z_list,
            beta_list,
            mu
        )


    # reduced gradient
    f0 = reduced_objective(x0)

    g = torch.autograd.grad(
        f0,
        x0,
        create_graph=True
    )[0]


    # exact reduced Hessian
    H = torch.autograd.functional.hessian(
        reduced_objective,
        x0
    )


    # numerical regularization
    H_reg = (
        H
        + hessian_reg
        * torch.eye(
            H.shape[0],
            device=device,
            dtype=dtype
        )
    )


    # ========================================================
    # 6. Newton direction
    # ========================================================

    use_newton = True

    try:

        direction = torch.linalg.solve(
            H_reg,
            -g
        )

        # Must be a descent direction
        gTd = torch.dot(
            g,
            direction
        )

        if (
            not torch.isfinite(direction).all()
            or gTd >= 0
        ):
            use_newton = False

    except RuntimeError:

        use_newton = False


    # ========================================================
    # 7. Fallback to gradient descent if Newton direction
    #    is not a descent direction
    # ========================================================

    if not use_newton:

        direction = -g

        gTd = torch.dot(
            g,
            direction
        )


    # ========================================================
    # 8. Form the full manuscript direction and use the actual current
    # iterate as the Armijo reference point.
    # ========================================================

    full_direction = -Y.detach().clone()
    full_direction[support] = direction.detach().reshape(s, p)
    full_gradient = grad_Y(Y, beta_sum, Z_agg, mu).detach()
    gTd = torch.sum(full_gradient * full_direction)
    if (not torch.isfinite(full_direction).all()) or gTd >= 0:
        full_direction = -full_gradient
        full_direction[torch.ones(Y.shape[0], dtype=torch.bool, device=device).scatter_(0, support, False)] = -Y.detach()[torch.ones(Y.shape[0], dtype=torch.bool, device=device).scatter_(0, support, False)]
        gTd = torch.sum(full_gradient * full_direction)

    eta = 1.0
    current_obj = objective(Y, Z_list, beta_list, mu).detach()

    accepted = False

    for _ in range(max_backtrack):

        Y_try = Y.detach() + eta * full_direction
        f_try = objective(Y_try, Z_list, beta_list, mu).detach()

        if (
            f_try
            <=
            current_obj
            + armijo_c
            * eta
            * gTd.detach()
        ):

            accepted = True
            break

        eta *= armijo_tau


    # ========================================================
    # 9. Construct Y^{k+1}
    # ========================================================

    if accepted:
        Y_new = Y_try
    else:
        Y_new = Y.detach().clone()


    # ========================================================
    # 10. Information
    # ========================================================

    with torch.no_grad():

        obj_after = objective(
            Y_new,
            Z_list,
            beta_list,
            mu
        ).item()

        row_norms = torch.norm(
            Y_new,
            dim=1
        )

        nnz_rows = int(
            (row_norms > 1e-10)
            .sum()
            .item()
        )


    info = {
        "obj_after": obj_after,
        "step_size": float(eta),
        "nnz_rows": nnz_rows,
        "newton_used": use_newton,
        "support": support.detach().cpu()
    }


    return Y_new.detach(), info
