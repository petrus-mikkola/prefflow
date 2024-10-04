import torch
import normflows as nf
from normflows.distributions.target import Target
from torch.distributions import MultivariateNormal
import numpy as np

import types

def set_up_problem(target_name,D):
    normalize = None
    if target_name == "onemoon":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = OneMoon()
    if target_name == "funnel":
        bounds = ((-5, 5),) * D #((-3,3),) + ((-5, 5),) * (D-1)
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = Funnel(D,bounds)
    if target_name == "onegaussian":
        bounds = ((-4, 4),) * D #needed for plotting
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = OneGaussian(D)
    if target_name == "twogaussians":
        bounds = ((-5, 5),) * D #needed for plotting
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = TwoGaussians(D)
    if target_name == "twomoons":
        D = 2
        bounds = ((-3, 3),) * D #needed for plotting
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = nf.distributions.TwoMoons()
    if target_name == "sinusoidal":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = nf.distributions.Sinusoidal_split(0.1,12)
    if target_name == "ringmixture":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = nf.distributions.RingMixture(n_rings=1)
    if target_name == "abalone_density":
        D=7
        original_bounds = ((0,0.82),(0,0.66),(0,1.14),(0,2.83),(0,1.49),(0,0.77),(0,1.01))
        ## Load the model using pickle
        import pickle
        model_load_path = 'ablone_nfm.pkl'
        with open(model_load_path, 'rb') as file:
            loaded_nf_model = pickle.load(file)      
        def normalize(X,reverse=False):
            if len(X.shape)==3: #assume shape (K,D,N)
                size = (1, D, 1)
            if len(X.shape)==2: #assume shape (N,D)
                size = (1, D)
            mins = torch.tensor([original_bounds[d_][0] for d_ in range(D)]).view(size)
            maxs = torch.tensor([original_bounds[d_][1] for d_ in range(D)]).view(size)
            if not reverse:
                return 2 * ((X - mins) / (maxs - mins)) - 1
            else:
                return ((X + 1) * (maxs - mins) / 2) + mins     
        #bounds in flow = ((-1,1),) * D
        bounds = original_bounds
        uniform = None
        target = loaded_nf_model
        # Store the original functions
        original_function_sample = target.sample
        original_function_log_prob = target.log_prob
        # Redefine sample and log_prob methods to match desired formats and dtypes
        def new_sample(self, x):
            return original_function_sample(x)[0].detach() #drop log-density output
        def new_log_prob(self, x):
            return original_function_log_prob(x.float()) #loaded model needs flot32 inputs
        # Create stable sample method for samling large number of samples
        def sample_stable(self, num_samples=1):
            max_batch_size = 1000
            #For complex flows sampling need to be split into batches to prevent run out of memeory issues
            num_batches = (num_samples + max_batch_size - 1) // max_batch_size
            samples = []
            logprobs = []
            for _ in range(num_batches):
                batch_size = min(max_batch_size, num_samples)
                samples_batch, logprobs_batch = original_function_sample(batch_size)
                samples.append(samples_batch.detach().float())
                logprobs.append(logprobs_batch.detach().float())
                num_samples -= batch_size
            samples, logprobs = torch.cat(samples, dim=0), torch.cat(logprobs, dim=0)
            samples = samples[~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))] 
            logprobs = logprobs[~torch.isnan(logprobs)] 
            return samples, logprobs
        # Update new methods in target instance
        target.sample = types.MethodType(new_sample, target)
        target.log_prob = types.MethodType(new_log_prob, target)
        target.sample_stable = types.MethodType(sample_stable, target)
    if target_name == "abalone_age":
        D=7
        original_bounds = ((0,0.82),(0,0.66),(0,1.14),(0,2.83),(0,1.49),(0,0.77),(0,1.01))        
        def normalize(X,reverse=False):
            if len(X.shape)==3: #assume shape (K,D,N)
                size = (1, D, 1)
            if len(X.shape)==2: #assume shape (N,D)
                size = (1, D)
            mins = torch.tensor([original_bounds[d_][0] for d_ in range(D)]).view(size)
            maxs = torch.tensor([original_bounds[d_][1] for d_ in range(D)]).view(size)
            if not reverse:
                return 2 * ((X - mins) / (maxs - mins)) - 1
            else:
                return ((X + 1) * (maxs - mins) / 2) + mins     
        #bounds in flow = ((-1,1),) * D
        bounds = original_bounds
        uniform = None
        target = None
    if target_name == "llm_prior":
        D=8
        original_bounds = ((0.5, 15),(1,52),(0.846154,141.909091),(0.333333,34.066667),(3,35682),(0.692308,1243.333333),(32.54, 41.95,),(-124.35,-114.31))
        def normalize(X,reverse=False):
            if len(X.shape)==3: #assume shape (K,D,N)
                size = (1, D, 1)
            if len(X.shape)==2: #assume shape (N,D)
                size = (1, D)
            mins = torch.tensor([original_bounds[d_][0] for d_ in range(D)]).view(size)
            maxs = torch.tensor([original_bounds[d_][1] for d_ in range(D)]).view(size)
            if not reverse:
                return 2 * ((X - mins) / (maxs - mins)) - 1
            else:
                return ((X + 1) * (maxs - mins) / 2) + mins     
        #bounds in flow = ((-1,1),) * D
        bounds = original_bounds
        uniform = None
        target = None
    return target, bounds, uniform, D, normalize 




class OneMoon(Target):
    """
    Unimodal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.lognormconstant = torch.tensor([1.1163528836769938]).log()

    def log_prob(self, z):
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((z[:, 0] + 2) / 0.3) ** 2
        )
        return log_prob - self.lognormconstant





class TwoGaussians:
    """
    Star-shape distribution
    """
    def __init__(self,d):
        self.d = d
        self.mean = 3 * torch.ones(d)
        self.sigma2 = 1
        self.rho = 0.9  # Ensure rho < 1 for positive semidefiniteness
        self.initialize()
        
    def generate_covariance_matrix(self, sigma2, rho, sign_pattern):
        base_cov = torch.eye(self.d) * sigma2
        # Add off-diagonal correlations based on sign_pattern
        for i in range(self.d):
            for j in range(i + 1, self.d):
                base_cov[i, j] = sign_pattern[i] * sign_pattern[j] * rho * sigma2
                base_cov[j, i] = base_cov[i, j]
        return base_cov
    
    def initialize(self):
        sign_pattern1 = np.ones(self.d)
        sign_pattern2 = np.array([(-1)**i for i in range(self.d)])
        covariance1 = self.generate_covariance_matrix(self.sigma2, self.rho, sign_pattern1)
        covariance2 = self.generate_covariance_matrix(self.sigma2, self.rho, sign_pattern2)
        self.normaldist1 = MultivariateNormal(self.mean,covariance1)
        self.normaldist2 = MultivariateNormal(self.mean,covariance2)

    def log_prob(self, z):
        log_prob = torch.log(self.normaldist1.log_prob(z).exp()/2 + self.normaldist2.log_prob(z).exp()/2)
        return log_prob
    
    def sample(self, num_samples=10**6):
        component_samples = torch.distributions.Categorical(torch.tensor([0.5, 0.5])).sample((num_samples,))
        samples1 = self.normaldist1.sample((num_samples,))
        samples2 = self.normaldist2.sample((num_samples,))
        samples = torch.zeros_like(samples1)
        mask1 = component_samples == 0
        mask2 = component_samples == 1
        samples[mask1] = samples1[mask1]
        samples[mask2] = samples2[mask2]
        return samples


class OneGaussian:
    
    def __init__(self,d):
        mean = torch.tensor([2.0*(-1)**(i+1) for i in range(d)])
        covariance = torch.full((d,d),d/15).fill_diagonal_(d/10)
        self.normaldist = MultivariateNormal(mean,covariance)

    def log_prob(self, z):
        log_prob = self.normaldist.log_prob(z)
        return log_prob
    
    def sample(self, num_samples=10**6):
        samples = self.normaldist.sample((num_samples,))
        return samples




class Funnel:
    def __init__(self, d, bounds):
        self.d = d
        self.a = 3 
        self.b = 0.25
        self.offset = 1 #0 reduces to the original
        self.bounds = bounds    

    def log_prob(self, x):
        x0 = x[..., 0]  # The "funnel" dimension
        x_rest = x[..., 1:]  # The conditional dimensions
        log_prob_x0 = torch.distributions.normal.Normal(self.offset*torch.tensor([1.0]), torch.tensor([self.a])).log_prob(x0)
        stddev = torch.exp(self.b * x0).unsqueeze(-1) * torch.ones_like(x_rest)
        log_prob_x_rest = torch.sum(torch.distributions.normal.Normal(self.offset*torch.ones_like(x_rest), stddev).log_prob(x_rest), dim=-1)  # Sum over the last dimension
        return log_prob_x0 + log_prob_x_rest

    def sample(self, num_samples=10**6):
        # Normal(0, a^2)
        x0 = self.offset + torch.randn(num_samples, 1) * self.a
        #  Normal(0, (e^(bx_0))^2)
        std_x_rest = torch.exp(self.b * x0)
        x_rest = self.offset + torch.randn(num_samples, self.d - 1) * std_x_rest
        samples = torch.cat([x0, x_rest], dim=1)
        return samples
