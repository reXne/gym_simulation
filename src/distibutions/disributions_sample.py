import torch
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Cauchy, Categorical, 
    Dirichlet, Exponential, FisherSnedecor, Gamma, Geometric,
    Laplace, Multinomial, MultivariateNormal, NegativeBinomial, 
    OneHotCategorical, Poisson, StudentT, Uniform
)

# Set random seed for reproducibility
torch.manual_seed(42)

# Sample from various distributions
sample_normal = Normal(0, 1).sample()
sample_normal
sample_bernoulli = Bernoulli(0.3).sample()
sample_beta = Beta(2, 5).sample()
sample_binomial = Binomial(10, 0.7).sample()
sample_cauchy = Cauchy(0, 1).sample()
sample_categorical = Categorical(torch.tensor([0.2, 0.3, 0.5])).sample()
sample_dirichlet = Dirichlet(torch.tensor([0.5, 0.5])).sample()
sample_exponential = Exponential(1).sample()
sample_fisher = FisherSnedecor(3, 4).sample()
sample_gamma = Gamma(2, 1).sample()
sample_geometric = Geometric(0.3).sample()
sample_laplace = Laplace(0, 1).sample()
sample_multinomial = Multinomial(10, torch.tensor([0.2, 0.3, 0.5])).sample()
sample_multivariate_normal = MultivariateNormal(torch.zeros(2), torch.eye(2)).sample()
sample_negative_binomial = NegativeBinomial(5, 0.3).sample()
sample_onehot_categorical = OneHotCategorical(torch.tensor([0.2, 0.3, 0.5])).sample()
sample_poisson = Poisson(3).sample()
sample_studentt = StudentT(2, 0, 1).sample()
sample_uniform = Uniform(0, 1).sample()

# Display the samples
print("Sample from Normal distribution:", sample_normal)
print("Sample from Bernoulli distribution:", sample_bernoulli)
print("Sample from Beta distribution:", sample_beta)
print("Sample from Binomial distribution:", sample_binomial)
print("Sample from Cauchy distribution:", sample_cauchy)
print("Sample from Categorical distribution:", sample_categorical)
print("Sample from Dirichlet distribution:", sample_dirichlet)
print("Sample from Exponential distribution:", sample_exponential)
print("Sample from FisherSnedecor distribution:", sample_fisher)
print("Sample from Gamma distribution:", sample_gamma)
print("Sample from Geometric distribution:", sample_geometric)
print("Sample from Laplace distribution:", sample_laplace)
print("Sample from Multinomial distribution:", sample_multinomial)
print("Sample from MultivariateNormal distribution:", sample_multivariate_normal)
print("Sample from NegativeBinomial distribution:", sample_negative_binomial)
print("Sample from OneHotCategorical distribution:", sample_onehot_categorical)
print("Sample from Poisson distribution:", sample_poisson)
print("Sample from StudentT distribution:", sample_studentt)
print("Sample from Uniform distribution:", sample_uniform)
