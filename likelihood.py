import torch
from itertools import combinations

def elementary_symmetric_sum(fvalues,l):
    #Computes the l-th elementary symmetric sum of a set over function values
    if l == 0:
        return torch.tensor(1.0)  #Return 1 for l=0 (sum of products of zero elements)
    idx_combinations = torch.tensor(list(combinations(range(fvalues.size(0)), l)))
    if idx_combinations.nelement() == 0:
        return torch.tensor(0.0)  #If no combinations, return 0
    combination_products = torch.prod(fvalues[idx_combinations], dim=1)
    return combination_products.sum()

def exp_rum_likelihood(f_x, f_others, s, k):
    #Exponential RUM model likelihood for any cardinality (k) of the choice set {x1,...,xk}
    #noise ~ Exp(s)
    #Utility function is f : X --> R
    #Computes the probability of x given utility f(x) being the k-wise winner given other k-1 function values, [f(x1),...,f(xk-1)]
    prob = 0.0
    for l in range(0,k): #from 0 to k-1
        f_star = f_others.max()
        exp_term = torch.exp(-s*(l+1)*(torch.clamp(f_star-f_x, min=0)))
        factor = 1 / (l + 1)
        #Computes elementary symmetric sum
        diff_terms = -torch.exp(-s * (f_x - f_others))
        sym_sum = elementary_symmetric_sum(diff_terms, l)
        prob += factor * exp_term * sym_sum
    return prob

# Example usage
# k = 5
# f_x = torch.tensor(-10)
# f_others = torch.tensor([1.0, 2.0, 4.0, 3.0])  # f values for other elements in the set
# s = torch.tensor(1.0)
# result = exp_rum_likelihood(f_x, f_others, s, k)
# print(f"Result of the computation: {result}")

# When k=2, the likelihood should match with the Laplace likelihood

# k = 2
# f_x = torch.tensor(1.0)
# f_others = torch.tensor([1.0])
# s = torch.tensor(1.0)
# m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), 1/s)
# result = exp_rum_likelihood(f_x, f_others, s, k)
# print(f"Result of the computation: {result}")
# print("Laplace likelihood pairwise probability:" + str(m.cdf(f_x - f_others)))

# k = 2
# f_x = torch.tensor(3.0)
# f_others = torch.tensor([2.0])
# s = torch.tensor(1.0)
# m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), 1/s)
# result = exp_rum_likelihood(f_x, f_others, s, k)
# print(f"Result of the computation: {result}")
# print("Laplace likelihood pairwise probability:" + str(m.cdf(f_x - f_others)))

# k = 2
# f_x = torch.tensor(2.0)
# f_others = torch.tensor([1.0])
# s = torch.tensor(0.1)
# m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), 1/s)
# result = exp_rum_likelihood(f_x, f_others, s, k)
# print(f"Result of the computation: {result}")
# print("Laplace likelihood pairwise probability:" + str(m.cdf(f_x - f_others)))