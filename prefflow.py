import torch
import torch.nn.functional as F
from normflows.core import NormalizingFlow
from likelihood import exp_rum_likelihood

class PrefFlow(NormalizingFlow):
    
    """
    Class for handling training of a preferential normalazing flow
    """

    def __init__(self,nflow,s,D,ranking,device,precision_double):

        # Call the parent class constructor with the name from the parent instance
        super().__init__(nflow.q0,nflow.flows)
        self.s_raw = torch.nn.Parameter(torch.tensor(s).log()).to(device)
        self.ranking = ranking
        self.D = D

        self.device = device
        self.precision_double = precision_double
        # Copy attributes and methods from the parent instance to the child instance
        for attr_name in dir(nflow):
            # Exclude private and special methods
            if not attr_name.startswith("__") and callable(getattr(nflow, attr_name)):
                setattr(self, attr_name, getattr(nflow, attr_name))
        self.flowname = nflow.__class__.__name__

    def sample_stable(self, num_samples=1):
        max_batch_size = 1000
        #For complex flows sampling need to be split into batches to prevent run out of memory issues
        num_batches = (num_samples + max_batch_size - 1) // max_batch_size
        samples = []
        logprobs = []
        for _ in range(num_batches):
            batch_size = min(max_batch_size, num_samples)
            samples_batch, logprobs_batch = super().sample(batch_size)
            samples.append(samples_batch.detach().float())
            logprobs.append(logprobs_batch.detach().float())
            num_samples -= batch_size
        samples, logprobs = torch.cat(samples, dim=0), torch.cat(logprobs, dim=0)
        samples = samples[~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))] 
        logprobs = logprobs[~torch.isnan(logprobs)] 
        return samples, logprobs
    
    def log_prob_stable(self, x):
        num_samples = x.shape[0]
        max_batch_size = 1000
        #For complex flows log prob computation need to be split into batches to prevent run out of memory issues
        num_batches = (num_samples + max_batch_size - 1) // max_batch_size
        logprobs = []
        for i in range(num_batches):
            batch_size = min(max_batch_size, num_samples)
            logprobs_batch = super().log_prob(x[i*batch_size:(i+1)*batch_size,:].type(torch.float64 if self.precision_double else torch.float32))
            logprobs.append(logprobs_batch.detach().float())
            num_samples -= batch_size
        logprobs = torch.cat(logprobs, dim=0)
        logprobs = logprobs[~torch.isnan(logprobs)] 
        return logprobs

    @property
    def s(self):
        return torch.exp(self.s_raw)

    def createX(self, batchX):
        #Assume the following input
            #batchX.shape = (2,D,N)
        #Outputs tensor of shape (2N,D)
        X = torch.empty((0,self.D)).to(self.device)
        for i in range(0, batchX.shape[2]):
            thetaprime = batchX[0,:,i].unsqueeze(0)
            thetaprimeprime = batchX[1,:,i].unsqueeze(0)
            X = torch.cat((X,torch.cat((thetaprime, thetaprimeprime), dim=0)), dim=0)
        return X
    
    def exract_preferred(self,batch,X,logf):
        preferred_logf = torch.empty(0,1).to(self.device)
        preferredX = torch.empty(0,self.D).to(self.device)
        batch_size = batch[0].shape[2]
        Y = batch[1]
        for i in range(batch_size):
            if Y[i].bool():
                preferred_logf = torch.cat((preferred_logf,logf[2*i].view(1,1)),dim=0)
                preferredX = torch.cat((preferredX,X[2*i,:].view(1,self.D)),dim=0)
            else:
                preferred_logf = torch.cat((preferred_logf,logf[(2*i)+1].view(1,1)),dim=0)
                preferredX = torch.cat((preferredX,X[(2*i)+1,:].view(1,self.D)),dim=0)
        return preferred_logf, preferredX

    def loglik_pairwise(self, f, Y):
        fthetaprime = f[::2]  # Extract odd-indexed elements
        fthetaprimeprime = f[1::2]  # Extract even-indexed elements
        
        #Exponential RUM
        z = torch.where(Y.bool(), fthetaprime - fthetaprimeprime, fthetaprimeprime - fthetaprime)
        prob = 0.5 - 0.5 * z.sign() * torch.expm1(-self.s*z.abs())
        logprob = torch.log(prob)
    
        return logprob

    def f(self,X):
        #Given points X, return log density at X, and log det Jacobian at X
        u, logdetJinv = self.inverse_and_log_det(X)
        logf = self.q0.log_prob(u) + logdetJinv
        logf[torch.isnan(logf)] = float('-inf')
        logdetJinv[torch.isnan(logdetJinv)] = float('-inf')
        return logf, logdetJinv
    
    def logposterior(self, batch, weightprior=1.0):
        
        if not self.ranking:
            X = self.createX(batch[0])
            logf, logdetJinv = self.f(X)
        else:
            X = batch
            k = X.shape[0]
            d = X.shape[1]
            n = X.shape[2]
            def shape_to_kn_d(X):
                X_ = X.transpose(1, 2).reshape(k*n, d)
                return(X_)
            logf, logdetJinv = self.f(shape_to_kn_d(X))

        loglik = 0
        #LIKELIHOOD PAIRWISE
        if not self.ranking:
            loglik += torch.sum(self.loglik_pairwise(logf,batch[1]), dim=0)
        #LIKELIHOOD RANKING
        if self.ranking:
            logf = logf.view(k, n)
            winners_logf = torch.empty(n)
            for i in range(n):
                #Ck = X[:,:,i]
                winners_logf[i] = logf[0,i]
                for j in range(k-1):
                    f_x = logf[j,i]
                    f_others = logf[j+1:,i]
                    loglik += torch.log(exp_rum_likelihood(f_x, f_others, self.s, k))
            logf = logf.view(k*n)

        #Prior
        if not self.ranking:
            winners_logf, _ = self.exract_preferred(batch,X,logf) #preferred points of pairwise comparisons
        logpx = winners_logf
        logprior = logpx.sum()
        
        logposterior = loglik + weightprior*logprior

        return logposterior




