# the value function and probability weighting 
# based on (Kahneman & Tversky 1979, 1992)
import numpy as np


def pt_value(x, lam=2.25, curv=0.88):
    # concave over gains, steeper and convex over losses 
    # default loss aversion is 2.25
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, np.power(np.abs(x), curv),
                    -lam * np.power(np.abs(x), curv))


# up_probability
def up_prob(drift, sharp):
    # the belief drift sets the subjective probability of an up move
    return 1.0/(1.0+np.exp(-np.clip(sharp*drift, -50, 50)))


def weight(p, gamma):
    # Tversky-Kahneman weighting
    return p**gamma/(p**gamma + (1-p)**gamma)**(1/gamma)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z, -50, 50)))
