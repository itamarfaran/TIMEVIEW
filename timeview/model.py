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
    def __init__(self, cov_type: str = "iid", param=None):
        super().__init__()

        self.cov_type = cov_type
        if self.cov_type not in ("iid", "ar1", "block"):
            raise ValueError

        if self.cov_type == "iid":
            self.param = None
        elif param is None:
            self.param = torch.nn.Parameter(torch.zeros(1))
        else:
            self.param = param

    def precision(self, dim):
        if self.cov_type == "iid":
            return torch.eye(dim)

        if self.cov_type == "ar1":
            param = expit_m1(self.param)
            coef = -param / (1 + param ** 2)
            return torch.eye(dim) + coef * (
                torch.diag(torch.ones(dim - 1), 1)
                + torch.diag(torch.ones(dim - 1), -1)
            )

        if self.cov_type == "block":
            param = torch.special.expit(self.param)
            coef = param / (1 + (dim - 1) * param)
            out = torch.eye(dim) - coef * torch.ones((dim, dim))
            return out / (1 - coef)

    def forward(self, y_true, y_pred, n=None):
        diff = y_true - y_pred

        if n is None:
            n = diff.shape[-1]
        else:
            diff = diff * filter_by_n(y_true, n)

        if self.cov_type == "iid":
            out = torch.sum(diff ** 2, dim=-1)

        else:
            out = diff @ self.precision(diff.shape[-1]) @ diff.T
            if diff.ndim > 1:
                out = torch.diag(out)

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
        latent_size = self.n_basis + 1 if is_dynamic_bias_enabled(config) else self.n_basis

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


class ARMA(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.p = config.arma.p
        self.q = config.arma.q

        if min(self.p, self.q) < 0:
            raise ValueError
        if max(self.p, self.q) > 1:
            raise NotImplementedError

        self.phi, self.theta = None, None

        if self.p:
            self.phi = torch.nn.Parameter(torch.zeros(self.p))
        if self.q:
            self.theta = torch.nn.Parameter(torch.zeros(self.p))

    def forward(self, y=None, y_pred=None):
        out = torch.zeros_like(y)
        if self.p:
            y_lag = torch.roll(y, 1, -1)
            y_lag[:, 0] = 0.0
            out = out + expit_m1(self.phi) * y_lag
        if self.q:
            resid = y - y_pred
            resid_lag = torch.roll(resid, 1, -1)
            resid_lag[:, 0] = 0.0
            out = out + expit_m1(self.theta) * resid_lag
        return out


class TTS(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        torch.manual_seed(config.seed)

        self.config = config
        self.encoder = Encoder(self.config)
        self.arma = ARMA(self.config)

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
            if self.arma.p + self.arma.q:
                raise NotImplementedError
                # todo: implement arma for iterative

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
            return preds + self.arma(y, preds)

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
