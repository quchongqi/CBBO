import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

def ei_kriging_believer(
    model,
    train_x,
    train_y,
    bounds,
    q,
):
    new_points = []

    for i in range(q):
        ei = ExpectedImprovement(
            model=model,
            best_f=train_y.max(),
        )

        x_next, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=64,
        )

        with torch.no_grad():
            y_fake = model.posterior(x_next).mean

        train_x = torch.cat([train_x, x_next])
        train_y = torch.cat([train_y, y_fake])

        model = SingleTaskGP(train_x, train_y)
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_model(mll)

        new_points.append(x_next)

    return torch.cat(new_points, dim=0)


def ei_constant_liar(
    model,
    train_x,
    train_y,
    bounds,
    q,
    style,
):
    new_points = []

    if style == 'max':
        L = train_y.max()
    elif style == 'min':
        L = train_y.min()
    
    elif style == 'mean':
        L = train_y.mean()
    else: 
        raise ValueError(f"Unknown style: {style}")

    for i in range(q):
        ei = ExpectedImprovement(
            model=model,
            best_f=train_y.max(),
        )

        x_next, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=64,
        )

        y_fake = L.view(1, 1)

        train_x = torch.cat([train_x, x_next])
        train_y = torch.cat([train_y, y_fake])

        model = SingleTaskGP(train_x, train_y)
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_model(mll)

        new_points.append(x_next)

    return torch.cat(new_points, dim=0)
