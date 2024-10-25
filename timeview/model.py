import torch
from timeview.basis import BSplineBasis
from timeview.config import Config


def is_dynamic_bias_enabled(config):
    if hasattr(config, 'dynamic_bias'):
        return config.dynamic_bias
    return False


def expit_m1(t):
    """expit function with mapping R -> (-1, 1)"""
    return 2 * torch.special.expit(t) - 1


def filter_by_n(y, n):
    return (
        torch.arange(y.shape[1])
        .repeat(y.shape[0])
        .reshape(y.shape)
    ) < n[:, None]


class MahalanobisLoss2D(torch.nn.Module):
    def __init__(self, cov_type="iid", param=None):
        super().__init__()

        self.cov_type = cov_type
        if self.cov_type not in ("iid", "ar1", "block"):
            raise ValueError
        self.param = param

    def precision(self, param, dim):
        if self.cov_type == "iid":
            return torch.eye(dim)

        if self.cov_type == "ar1":
            param = expit_m1(param)
            coef = -param / (1 + param ** 2)
            return torch.eye(dim) + coef * (
                torch.diag(torch.ones(dim - 1), 1)
                + torch.diag(torch.ones(dim - 1), -1)
            )

        if self.cov_type == "block":
            param = torch.special.expit(param)
            coef = param / (1 + (dim - 1) * param)
            out = torch.eye(dim) - coef * torch.ones((dim, dim))
            return out / (1 - coef)

    def forward(self, y_true, y_pred, param=None, n=None):
        if param is None:
            param = self.param
        elif self.param is not None:
            raise TypeError("cannot pass param argument if param attribute is not none")

        diff = y_true - y_pred

        if diff.ndim == 1:
            diff = diff.unsqueeze(0)

        if n is None:
            n = torch.full((diff.shape[0],), diff.shape[-1])
        else:
            diff = diff * filter_by_n(diff, n)

        if self.cov_type == "iid":
            out = torch.sum(diff ** 2, dim=-1)
        else:
            # out = torch.stack([
            #     d[:n_] @ self.precision(param, n_) @ d[:n_].T
            #     for d, n_ in zip(diff, n)
            # ])
            out = torch.diag(diff @ self.precision(param, diff.shape[-1]) @ diff.T)
        return torch.mean(out / n)


class Encoder(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.n_basis = config.n_basis
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0
        latent_size = self.n_basis + is_dynamic_bias_enabled(config)

        self.layers = []
        for i in range(len(self.hidden_sizes)):
            self.layers.extend((
                torch.nn.Linear(self.hidden_sizes[i - 1] if i else self.n_features, self.hidden_sizes[i]),
                torch.nn.BatchNorm1d(self.hidden_sizes[i]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_p),
            ))
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1], latent_size))
        self.nn = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)


class AR(torch.nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p

        if self.p < 0:
            raise ValueError
        if self.p > 1:
            raise NotImplementedError
            # when p > 1, the polynomial 1 - phi_1 * t - phi_2 * t ^ 2 + ...
            # must have roots lower than 1 in absolute value to be stationary.
            # with p == 1 this easy, as we limit -1 < phi < 1,
            # but with higher rank it is not straight-forward. so we limit p <= 1 for now.

        if self.p:
            self.phi = torch.nn.Parameter(torch.zeros(self.p))
        else:
            self.phi = None

    def forward(self, y, y_pred):
        out = torch.zeros_like(y)
        if self.p:
            resid = y - y_pred
            resid[:, 1:] = resid[:, :-1]  # torch.roll(resid, 1, -1)
            resid[:, 0] = 0.0
            out = out + expit_m1(self.phi) * resid
        return out


class NeuralAR(torch.nn.Module):
    def __init__(self, hidden_sizes, dropout_p=0.0):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_p = dropout_p

        self.layers = []
        for i in range(len(self.hidden_sizes)):
            self.layers.extend((
                torch.nn.Linear(self.hidden_sizes[i - 1] if i else 2, self.hidden_sizes[i]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_p),
            ))
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1], 1))
        self.nn = torch.nn.Sequential(*self.layers)

    def forward(self, y, y_pred):
        x = torch.stack((y, y_pred), 2)
        x[:, 1:, :] = x[:, :-1, :]  # torch.roll(x, 1, 1)
        x[:, 0, :] = 0.0
        return self.nn(x).squeeze(-1)


class TTS(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        torch.manual_seed(config.seed)

        self.config = config
        self.encoder = Encoder(self.config)

        if self.config.ar.type == 'none':
            self.ar = AR(0)
        elif self.config.ar.type == 'parametric':
            self.ar = AR(self.config.ar.p)
        elif self.config.ar.type == 'neural':
            self.ar = NeuralAR(self.config.ar.hidden_sizes, self.config.ar.dropout_p)

        if self.config.cov_type == "iid":
            self.cov_param = None
        elif (
            self.config.cov_type == "ar1"
            and self.config.ar.type == "parametric"
            and self.ar.p == 1
        ):
            self.cov_param = self.ar.phi
        else:
            self.cov_param = torch.nn.Parameter(torch.zeros(1))

        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X, Phis, y=None):
        """
        Args:
            X: a tensor of shape (D,M) where D is the number of sample and M is the number of static features
            Phi:
                if dataloader_type = 'tensor': a tensor of shape (D,N_max,B) where D is the number of sample, N_max is the maximum number of time steps and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d,B) where N_d is the number of time steps and B is the number of basis functions
        """
        h = self.encoder(X)

        if is_dynamic_bias_enabled(self.config):
            self.bias = h[:, -1]
            h = h[:, :-1]
        
        if self.config.dataloader_type == "iterative":
            if self.ar.p:
                raise NotImplementedError
                # todo: implement ar for iterative

            return [
                torch.matmul(Phi, h[d, :]) + (
                    self.bias[d]
                    if is_dynamic_bias_enabled(self.config) else
                    self.bias
                ) for d, Phi in enumerate(Phis)
            ]

        elif self.config.dataloader_type == "tensor":
            preds = torch.matmul(Phis, torch.unsqueeze(h, -1)).squeeze(-1) + (
                torch.unsqueeze(self.bias, -1)
                if is_dynamic_bias_enabled(self.config) else
                self.bias
            )
            return preds + self.ar(y, preds)

    def predict_latent_variables(self, X):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
        Returns:
            a numpy array of shape (D,B) where D is the number of sample and B is the number of basis functions
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        self.encoder.eval()
        if is_dynamic_bias_enabled(self.config):
            with torch.no_grad():
                return self.encoder(X)[:, :-1].cpu().numpy()
        else:
            with torch.no_grad():
                return self.encoder(X).cpu().numpy()        

    def forecast_trajectory(self, x, t):
        """
        Args:
            x: a numpy array of shape (M,) where M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (N,) where N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        x = torch.unsqueeze(torch.from_numpy(x), 0).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[0, -1]
                h = h[:, :-1]
            return (torch.matmul(Phi, h[0, :]) + self.bias).cpu().numpy()

    def forecast_trajectories(self, X, t):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (D,N) where D is the number of sample and N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)  # shape (N,B)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X)  # shape (D,B)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[:, -1]
                h = h[:, :-1]
            return (torch.matmul(h, Phi.T) + self.bias).cpu().numpy()  # shape (D,N), broadcasting will take care of the bias
