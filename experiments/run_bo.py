import torch
import numpy as np
import time
import pandas as pd
import os
import sys
import inspect

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf

pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)

def run_bo(
    result_path,
    method,
    method_style,
    dataset,
    seed,
    T=20,
    q=5,
    n_init=8,
    device=None,
    **kwargs,
):
    if device is None:
        device = torch.device("cpu")
    dtype = torch.double

    torch.manual_seed(seed)
    np.random.seed(seed)

    f, bounds, dim = get_dataset(dataset, device, dtype)

    records = []

    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
        n_init, dim, device=device, dtype=dtype
    )
    train_Y = f(train_X).unsqueeze(-1)
    
    # ===== record =====
    for i in range(n_init):
        row = {
            "iter": 0,
            "y": train_Y[i].item(),
            "time": 0,
        }
        for d in range(dim):
            row[f"x_{d}"] = train_X[i, d].item()
        records.append(row)
        
    best_values = []
    best_values.append(train_Y.max().item())

    for t in range(T):

        start_time = time.time()

        #------------------method 1--------------
        noise = torch.full_like(train_Y, 1e-4)

        model = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=noise,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=1),
        ).to(device=device, dtype=dtype)
        # -------------method 2---------------------------------------------
        # model = SingleTaskGP(train_X, train_Y).to(device=device, dtype=dtype)
        #----------------------------------------------------------------------

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        #------------------Body of method ---------------------------------------------------
        #------------------------------------------------------------------------------------

        # LP, EI-KB, EI-CL, BUCB, UCB-PE, 
        if method_style == 1:
            kwargs.pop('N', None)
            kwargs.pop('D_t', None)
            # LP
            if 'LP' in method:
                acqf_name = method.split("-")[0] 
                kwargs['bounds'] = bounds                
                
                if acqf_name == 'LogEI' or  acqf_name == 'EI':
                    best_f = kwargs.get("best_f", None)
                    if best_f == None:
                        best_f = train_Y.max().item()
                        kwargs["best_f"] = best_f

                elif acqf_name == 'UCB':
                    beta = kwargs.get("beta", None)
                    if beta == None:
                        beta = 0.2
                        kwargs["beta"] = beta

                elif acqf_name == 'PES':
                    optimal_inputs = kwargs.get("optimal_inputs", None)
                    if optimal_inputs == None:
                        from botorch.utils.sampling import draw_sobol_samples
                        optimal_inputs = draw_sobol_samples(
                        bounds=bounds,
                        n=1024,
                        q=1,
                        ).squeeze(1)   # [1024, d]
                        kwargs["optimal_inputs"] = optimal_inputs
                elif acqf_name == 'MES':
                    candidate_set = kwargs.get("candidate_set", None)
                    if candidate_set == None:
                        candidate_set = torch.rand(1000, bounds.size(1)).to(device=device, dtype=dtype)
                        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
                        kwargs["candidate_set"] = candidate_set
                elif acqf_name == 'EE':
                    temperature = kwargs.get("temperature", None)
                    kernel_amplitude = kwargs.get("amplitude", None)
                    if temperature == None:
                        temperature = 5.0
                        kwargs['temperature'] = temperature
                    if kernel_amplitude == None:
                        kernel_amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude
                        kwargs['kernel_amplitude'] = kernel_amplitude

                acqf, filtered_kwargs = get_aqcf( acqf_name=acqf_name,
                model=model,
                **kwargs,
                )
                kwargs['base_acqf'] = acqf

                kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in filtered_kwargs
                }
                kwargs['q'] = q
            
            # EI-KB
            elif method == 'EI-KB':
                kwargs['train_x'] = train_X
                kwargs['train_y'] = train_Y
                kwargs['bounds'] = bounds
                kwargs['q'] = q

            elif method == 'EI-CL':
                kwargs['train_x'] = train_X
                kwargs['train_y'] = train_Y
                kwargs['bounds'] = bounds
                kwargs['q'] = q
                style = kwargs.get("style", None)
                if style == None:
                    style = 'mean'
                    kwargs['style'] = 'mean'
            
            elif method == 'BUCB' or method == 'UCB-PE': 
                kwargs['bounds'] = bounds
                beta = kwargs.get("beta", None)
                if beta == None:
                    beta = 0.2
                    kwargs['beta'] = beta
                kwargs['q'] = q
 
            candidates = get_methods(
            name=method, 
            model=model, 
            **kwargs
            )

        # qLogEI, qEI, qUCB, qKG, PPES, qMES, GIBBON, BEEBO 
        elif method_style == 2:
            kwargs.pop('N', None)
            kwargs.pop('D_t', None)
            if method == 'qLogEI' or  method == 'qEI':
                best_f = kwargs.get("best_f", None)
                if best_f == None:
                    best_f = train_Y.max().item()
                    kwargs["best_f"] = best_f
            elif  method == 'qUCB':
                beta = kwargs.get("beta", None)
                if beta == None:
                    beta = 0.2
                    kwargs["beta"] = beta
            elif method == 'PPES':
                optimal_inputs = kwargs.get("optimal_inputs", None)
                if optimal_inputs == None:
                    from botorch.utils.sampling import draw_sobol_samples
                    optimal_inputs = draw_sobol_samples(
                    bounds=bounds,
                    n=1024,
                    q=1,
                    ).squeeze(1)   # [1024, d]
                    kwargs["optimal_inputs"] = optimal_inputs
            elif method == 'qMES' or method == 'GIBBON':
                candidate_set = kwargs.get("candidate_set", None)
                if candidate_set == None:
                    candidate_set = torch.rand(1000, bounds.size(1)).to(device=device, dtype=dtype)
                    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
                    kwargs["candidate_set"] = candidate_set
            elif method == 'BEEBO':
                temperature = kwargs.get("temperature", None)
                amplitude = kwargs.get("amplitude", None)
                if temperature == None:
                    temperature = 1.0
                    kwargs['temperature'] = temperature
                if amplitude == None:
                    amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude
                    print('---------amplitude:', amplitude)
                    amplitude = float(min(amplitude, 5.0))  # avoid: RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
                    kwargs['amplitude'] = amplitude
            
            acqf =  get_methods(name=method, model=model, **kwargs).to(device=device, dtype=dtype)

            if method == 'qMES' or method == 'GIBBON' or method == 'BEEBO':
                candidates, _= optimize_acqf(
                    acq_function=acqf, 
                    q=q, 
                    num_restarts=20, 
                    bounds=bounds, # the bounds of the optimization problem
                    raw_samples=512, 
                    sequential=True, 
                    )
            else: 
                candidates, _= optimize_acqf(
                acq_function=acqf, 
                q=q, 
                num_restarts=20, 
                bounds=bounds, # the bounds of the optimization problem
                raw_samples=512, 
                )

        # CBBO 
        elif method_style == 3: 
            if 'CBBO' in method:
                kwargs['q'] = q
                kwargs['bounds'] = bounds
                discretizer_methods = kwargs.get("discretizer_methods", None)
                if discretizer_methods == None:
                    discretizer_methods = 'hypercube'
                    kwargs['discretizer_methods'] = discretizer_methods
                
                kwargs['device'] = device
                eta_l = kwargs.get("eta_l", None)
                if eta_l == None:
                    eta_l = 0.1
                    kwargs['eta_l'] = eta_l

                eta_tau = kwargs.get("eta_tau", None)
                if eta_tau == None:
                    eta_tau = 0.6
                    kwargs['eta_tau'] = eta_tau
                N = kwargs.get("N", None)
                if N == None:
                    N = 1000
                    kwargs['N'] = N
                D_t = kwargs.get("D_t", None)      
                
                acqf_name = method.split("-")[1]
                kwargs['acqf_name'] = acqf_name

                if acqf_name == 'LogEI' or acqf_name == 'EI':
                    best_f = kwargs.get("best_f", None)
                    if best_f == None:
                        best_f = train_Y.max().item()
                        kwargs["best_f"] = best_f
                elif acqf_name == 'UCB':
                    beta = kwargs.get("beta", None)
                    if beta== None:
                        beta = 0.2
                        kwargs["beta"] = beta   
                elif acqf_name == 'KG':
                    with torch.no_grad():
                        _ = model.posterior(model.train_inputs[0])   # Call in advance to prevent an error in 'num_fantasies'
                    kwargs['num_fantasies'] = 1
                elif acqf_name == 'PES':
                    optimal_inputs = kwargs.get("optimal_inputs", None)
                    if optimal_inputs == None:
                        from botorch.acquisition.analytic import PosteriorMean
                        pm = PosteriorMean(model).to(device=device, dtype=dtype)
                        X_star, _ = optimize_acqf(
                                    pm,
                                    bounds=bounds,
                                    q=1,
                                    num_restarts=20,
                                    raw_samples=512,
                                )
                        kwargs['optimal_inputs'] = X_star
                elif acqf_name == 'MES':
                    candidate_set = kwargs.get("candidate_set", None)
                    if candidate_set == None:
                        from cbbo.discretization import discretizer
                        candidate_set = discretizer(
                            discretizer_method='hypercube',
                            bounds=bounds.T,
                            N=kwargs['N'],
                        )
                        kwargs['candidate_set'] = candidate_set
                elif acqf_name == 'EE':
                    temperature = kwargs.get("temperature", None)
                    amplitude = kwargs.get("amplitude", None)
                    if temperature == None:
                        temperature = 1.0
                        kwargs['temperature'] = temperature
                    if amplitude == None:
                        amplitude = model.covar_module.outputscale.item() # get the GP's kernel amplitude
                        amplitude = float(min(amplitude, 5.0))  # avoid: RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
                        kwargs['amplitude'] = amplitude
                
                candidates, _, _, D_t_new = get_methods(
                                    name=method, 
                                    model=model, 
                                    **kwargs
                                    )
                # update D_t
                kwargs['D_t'] = D_t_new
     
            else: 
                raise ValueError(f"Unknow Method: {method}")

        else: 
            raise ValueError(f"Unknown Method Style: {method_style}")

        #------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------  

        elapsed = time.time() - start_time

        new_Y = f(candidates).unsqueeze(-1)

        # ===== record =====
        for i in range(q):
            row = {
                "iter": t + 1,
                "y": new_Y[i].item(),
                "time": elapsed,
            }
            for d in range(dim):
                row[f"x_{d}"] = candidates[i, d].item()
            records.append(row)

        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        best_values.append(train_Y.max().item())

    # ===== save excel =====
    result_path = result_path
    save_path_records = make_result_path(result_path, dataset, method, q, f'records_{seed}')
    save_path_best_values = make_result_path(result_path, dataset, method, q, f'best_values_{seed}')
    pd.DataFrame(records).to_excel(save_path_records, index=False)
    df_best = pd.DataFrame(
    {"best_y": best_values}
    )
    df_best.to_excel(save_path_best_values, index=False)


    print(
    f" Finished: dataset={dataset}, method={method}, "
    f"q={q}, seed={seed}\n"
    f" Saved to: {save_path_records}"
    )
    print(best_values)


def make_result_path(root, dataset, method, q, seed):
    path = os.path.join(root, dataset, method, f"q{q}")
    os.makedirs(path, exist_ok=True)

    filename = f"{seed}.xlsx"
    return os.path.join(path, filename)


from botorch.test_functions import (Branin, Ackley, Hartmann, Cosine8, Powell, Rastrigin, Rosenbrock, StyblinskiTang, Shekel, 
      Beale, Bukin, Griewank, Michalewicz, DropWave, DixonPrice, EggHolder, HolderTable, Levy, SixHumpCamel, ThreeHumpCamel)

def get_dataset(name, device, dtype):
    import re
    name = name.lower()
    dim = int(re.findall(r"\d+", name)[0])

    if name == "branin2":
        f = Branin(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2
    
    elif name == "beale2":
        f = Beale(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2

    elif name == "bukin2":
        f = Bukin(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2  

    elif name == "griewank2":
        f = Griewank(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2 

    elif name == "dropwave2":
        f = DropWave(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2 

    elif name == "dixonprice2":
        f = DropWave(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2   

    elif name == "eggholder2":
        f = EggHolder(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2   

    elif name == "holdertable2":
        f = HolderTable(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2  

    elif name == "sixhumpcamel2":
        f = SixHumpCamel(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2   

    elif name == "threehumpcamel2":
        f = ThreeHumpCamel(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 2        
 
    elif name == "cosine8":
        f = Cosine8(negate=False).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 8

    elif "ackley" in name: 
        assert dim is not None
        f = Ackley(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)   

    elif "levy" in name: 
        assert dim is not None
        f = Levy(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)        

    elif "hartmann" in name:
        assert dim is not None
        f = Hartmann(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
    
    elif "rastrigin" in name:
        assert dim is not None
        f = Rastrigin(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
    
    elif "rosenbrock" in name:
        assert dim is not None
        f = Rosenbrock(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)   
    
    elif "styblinski" in name or "stybtang" in name:
        assert dim is not None
        f = StyblinskiTang(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)    

    elif "powell" in name:
        assert dim is not None and dim % 4 == 0, \
            "Powell function requires dim % 4 == 0"
        f = Powell(dim=dim, negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)

    elif "shekel" in name:
        f = Shekel(negate=True).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)
        dim = 4

    elif "michalewicz" in name: 
        assert dim is not None
        f = Michalewicz(dim=dim, negate=False).to(device=device, dtype=dtype)
        bounds = f.bounds.to(device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return f, bounds, dim

from botorch.acquisition.analytic import (
    ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound,
    )
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from beebo import BatchedEnergyEntropyBO

def get_aqcf(acqf_name,
                model,
                **kwargs,):
    
    if acqf_name == 'LogEI':
        filtered_kwargs = get_filtered_kwargs(f=LogExpectedImprovement, **kwargs)
        acqf = LogExpectedImprovement(model=model,
        **filtered_kwargs,
        )
    elif acqf_name == 'EI':
        filtered_kwargs = get_filtered_kwargs(f=ExpectedImprovement, **kwargs)
        acqf = ExpectedImprovement(model=model,
        **filtered_kwargs,
        )
    elif acqf_name == 'UCB':
        filtered_kwargs = get_filtered_kwargs(f=UpperConfidenceBound, **kwargs)
        acqf = UpperConfidenceBound(model=model,
        **filtered_kwargs,
        ) 
    elif acqf_name == 'KG':
        filtered_kwargs = get_filtered_kwargs(f=qKnowledgeGradient, **kwargs)
        acqf = qKnowledgeGradient(model=model,
        **filtered_kwargs,
        )
    elif acqf_name == 'PES':
        filtered_kwargs = get_filtered_kwargs(f=qPredictiveEntropySearch, **kwargs)
        acqf = qPredictiveEntropySearch(model=model,
        **filtered_kwargs,
        )
    elif acqf_name == 'MES':
        filtered_kwargs = get_filtered_kwargs(f=qMaxValueEntropy, **kwargs)
        acqf = qMaxValueEntropy(model=model,
        **filtered_kwargs,
        )
    elif acqf_name == 'EE':
        filtered_kwargs = get_filtered_kwargs(f=BatchedEnergyEntropyBO, **kwargs)
        acqf = BatchedEnergyEntropyBO(model=model,
        **filtered_kwargs,
        )
    else:
        raise ValueError(f"Unknown Aquisition Function: {name}")
    
    return acqf, filtered_kwargs


def get_filtered_kwargs(f, **kwargs):
        sig = inspect.signature(f)
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }
        return filtered_kwargs


from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from beebo import BatchedEnergyEntropyBO

from methods_batch.ei_batch import ei_kriging_believer
from methods_batch.ei_batch import ei_constant_liar
from methods_batch.ucb_batch import bucb
from methods_batch.ucb_batch import gp_ucb_pe
from methods_batch.local_penalization import optimize_acqf_with_lp
from cbbo.cbbo import coverage_batch_bo_step


def get_methods(
    name,
    model,
    **kwargs,
):

    if name == "qEI":
        return qExpectedImprovement(model=model, **kwargs)   
    elif name == 'qLogEI':
        return qLogExpectedImprovement(model=model, **kwargs)
    elif name == 'qUCB':
        return qUpperConfidenceBound(model=model, **kwargs)
    elif name == 'qKG':
        return qKnowledgeGradient(model=model, **kwargs)
    elif name == 'PPES':
        return qPredictiveEntropySearch(model=model, **kwargs)
    elif name == 'qMES':
        return qMaxValueEntropy(model=model, **kwargs)
    elif name == 'GIBBON':
        return qLowerBoundMaxValueEntropy(model=model, **kwargs)
    elif name =='BEEBO':
        return BatchedEnergyEntropyBO(model=model, **kwargs)
    
    elif name == 'EI-KB':
        return ei_kriging_believer(model=model, **kwargs)
    elif name == 'EI-CL':
        return ei_constant_liar(model=model, **kwargs)
    elif name == 'BUCB':
        return bucb(model=model, **kwargs)
    elif name == 'UCB-PE':
        return gp_ucb_pe(model=model, **kwargs)
    elif 'LP' in name:
        return optimize_acqf_with_lp(model=model, **kwargs)
    
    elif 'CBBO' in name:
        return coverage_batch_bo_step(model=model, **kwargs)

    else:
        raise ValueError(f"Unknown Methods: {name}")



# if __name__ == "__main__":

#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     print('-----device:', device)
#     run_bo(
#         method="CBBO-EI",
#         method_style=3,       # optimize_acqf
#         dataset="powell4",
#         seed=0,
#         T=5,                     
#         q=20,
#         n_init=4,
#         device=device,
#     )
