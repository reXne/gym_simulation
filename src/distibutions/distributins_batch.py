import torch
from torch.distributions import Independent, OneHotCategorical

# Sample from OneHotCategorical distribution
batch_size = 10
num_categories = 5
logits = torch.randn(batch_size, num_categories)
onehot_categorical_dist = OneHotCategorical(logits=logits)

# Wrap the distribution with Independent
independent_dist = Independent(onehot_categorical_dist, 1)

# Sample from the independent distribution
samples = independent_dist.sample()

# Display the samples
print("Samples from Independent OneHotCategorical distribution:")
print(samples)
