
import torch
from torch.distributions.normal import Normal
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import UpperConfidenceBound


class BUCB_Acquisition(AcquisitionFunction):
    def __init__(self, base_model, fantasy_model, beta):
        super().__init__(fantasy_model)
        self.base_model = base_model      
        self.fantasy_model = fantasy_model  
        self.beta = torch.tensor(beta)

    def forward(self, X):
        mu = self.base_model.posterior(X).mean.squeeze(-1)
        sigma = self.fantasy_model.posterior(X).variance.sqrt().squeeze(-1)
        return mu + torch.sqrt(self.beta) * sigma


class PosteriorStdAcquisition(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, X):
        return self.model.posterior(X).variance.sqrt().squeeze(-1)


def bucb(model,
         bounds,
        beta, 
        q,
        num_restarts=1,
        raw_samples=16,):
    base_model = model
    fantasy_model = model
    new_points = []

    for _ in range(q):
        acq = BUCB_Acquisition(
            base_model=base_model,
            fantasy_model=fantasy_model,
            beta=beta,
        )

        x_next, _ = optimize_acqf(acq,
         bounds, 
         q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
         )

        y_fake = fantasy_model.posterior(x_next).mean

        x_next_ = x_next.view(1, -1)
        y_fake_ = y_fake.view(1, -1)

        Y_raw = fantasy_model.outcome_transform.untransform(
                fantasy_model.train_targets.unsqueeze(-1)
                )[0]

        Y_fantasy = torch.cat(
        [Y_raw, y_fake_],
        dim=0,
    )
        noise = torch.full_like(Y_fantasy, 1e-4)

        fantasy_model = SingleTaskGP(
            torch.cat([fantasy_model._original_train_inputs, x_next_]),
            Y_fantasy,
            train_Yvar=noise,
            input_transform=base_model.input_transform,
            outcome_transform=base_model.outcome_transform,

        )
        fantasy_model.load_state_dict(base_model.state_dict())  

        x_next = x_next.view(1, -1)

        new_points.append(x_next)


    return torch.cat(new_points)


def gp_ucb_pe(
    model,
    bounds,
    beta,
    q,
    num_restarts=1,
    raw_samples=16,
):
    """
    GP-UCB-PE batch Bayesian optimization.

    Parameters
    ----------
    base_model : SingleTaskGP
        GP model fitted on true observations.
        The posterior mean of this model remains fixed throughout the batch.
    bounds : Tensor
        Optimization bounds of shape [2, d].
    beta : float
        Exploration parameter for GP-UCB.
    q : int
        Batch size.
    """

    new_points = []

    # ======================================================
    # 1. UCB phase: select the first point via GP-UCB
    # ======================================================
    ucb = UpperConfidenceBound(
        model=model,
        beta=beta,
    )

    x0, _ = optimize_acqf(
        acq_function=ucb,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    new_points.append(x0)

    # Use posterior mean as a hallucinated observation
    # (this keeps the posterior mean unchanged in expectation)
    with torch.no_grad():
        y_fake = model.posterior(x0).mean

    # Initialize fantasy model:
    # - conditioning set is augmented
    # - hyperparameters are copied from the base model

    Y_raw = model.outcome_transform.untransform(
            model.train_targets.unsqueeze(-1)
            )[0]

    Y_fantasy = torch.cat(
    [Y_raw, y_fake],
    dim=0,
    )
    noise = torch.full_like(Y_fantasy, 1e-4)

    fantasy_model = SingleTaskGP(
        torch.cat([model._original_train_inputs, x0]),
        Y_fantasy,
        train_Yvar=noise,
        input_transform=model.input_transform,
        outcome_transform=model.outcome_transform,
    )
    fantasy_model.load_state_dict(model.state_dict())

    # ======================================================
    # 2. PE phase: select remaining points by maximizing ¦Ò(x)
    # ======================================================
    for _ in range(q - 1):
        pe_acq = PosteriorStdAcquisition(fantasy_model)

        x_next, _ = optimize_acqf(
            acq_function=pe_acq,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        x_next = x_next.view(1, -1)
        new_points.append(x_next)

        # Hallucinated observation:
        # mean is taken from the base model (not the fantasy model)
        with torch.no_grad():
            y_fake = model.posterior(x_next).mean

        # Update fantasy model:
        # - posterior variance is updated
        # - posterior mean remains that of the base GP
        Y_raw = fantasy_model.outcome_transform.untransform(
                fantasy_model.train_targets.unsqueeze(-1)
            )[0]
        Y_fantasy = torch.cat(
            [Y_raw, y_fake],
            dim=0,
        )
        noise = torch.full_like(Y_fantasy, 1e-4)
        fantasy_model = SingleTaskGP(
            torch.cat([fantasy_model._original_train_inputs, x_next]),
            Y_fantasy,
            train_Yvar=noise,
            input_transform=model.input_transform,
            outcome_transform=model.outcome_transform,

        )
        fantasy_model.load_state_dict(model.state_dict())

    return torch.cat(new_points)

