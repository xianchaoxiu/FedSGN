import torch

class UserFedSGN:


    def __init__(
            self,
            device,
            id,
            train_data,
            common_Y,
            beta,
            local_epochs):

        self.device = device
        self.id = id

        self.train_data = (
            train_data.T
            .to(device)
        )

        self.train_samples = (
            train_data.shape[0]
        )

        self.local_epochs = local_epochs

        self.A_i = self.train_data

        self.Q_i = (
            self.A_i
            @
            self.A_i.T
        )

        self.C_i = (self.Q_i/self.train_samples)

        eigvals = torch.linalg.eigvalsh(
            self.C_i
        )

        self.lam_min_C = eigvals[0].item()
        self.lam_max_C = eigvals[-1].item()

        self.gamma_i = (
            1.1
            *
            self.lam_max_C
        )

        # Algorithm 1 treats beta_i as an input to the Y-subproblem, so it
        # must exist before the first local X update.
        I = torch.eye(self.C_i.shape[0], device=self.device)
        self.r_i = torch.linalg.norm(self.gamma_i * I - self.C_i, 2)
        self.beta = torch.sqrt(torch.tensor(2.0, device=self.device)) * self.r_i + 1e-3

        self.X_i = torch.zeros_like(common_Y).to(device)

        self.Lambda_i = torch.zeros_like(
            common_Y
        ).to(device)

        self.Y_i = (
            common_Y
            .clone()
            .to(device)
        )

    def set_global_Y(self,Y_server):
        self.Y_i = (Y_server.clone().to(self.device))

    def update_primal(self):

        I = torch.eye(self.C_i.shape[0],device=self.device)
        C = self.C_i
        d = C.shape[0]
        I = torch.eye(d,device=self.device)
        eps = 1e-6

        lhs = ((self.gamma_i+self.beta)*I-C+eps*I)
        rhs = (self.beta*self.Y_i+self.Lambda_i)
        self.X_i = torch.linalg.solve(lhs,rhs)


        if not torch.isfinite(self.X_i).all():
            self.X_i.zero_()

        with torch.no_grad():
            obj = (0.5*torch.trace(self.X_i.T@(self.gamma_i*I-C)@self.X_i)-torch.sum(self.Lambda_i * (self.X_i-self.Y_i))+0.5*self.beta*torch.norm(self.X_i-self.Y_i,p='fro')**2).item()
        print(
            f"[User {self.id}] "
            f"X update Obj={obj:.6f}, "
            f"beta={self.beta:.4f}, "
            f"gamma={self.gamma_i:.4f}"
        )

    def update_dual(
            self,
            Y_server,
            tau=1.0):

        self.Y_i = (
            Y_server
            .clone()
            .to(self.device)
        )

        self.Lambda_i = (

            self.Lambda_i

            -

            tau
            *
            self.beta
            *
            (
                self.X_i
                -
                self.Y_i
            )

        )

        if not torch.isfinite(
            self.Lambda_i
        ).all():

            self.Lambda_i.zero_()

    def get_model(self):

        return (self.X_i.detach(),self.Lambda_i.detach(),self.beta)

    def get_sample_number(self):

        return self.A_i.shape[1]
