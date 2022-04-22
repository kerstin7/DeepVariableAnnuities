from scipy.stats import norm
import numpy as np
import torch


def generate_stock_prices(batchsize, discretization_steps, delta_T, drift, sigma, T, s0):
    # Generate increments of a Brownian motion (normally distributed random variables with mean = 0 and stddev = delta_T)
    delta_W = np.random.normal(
        size=[batchsize, discretization_steps], loc=0, scale=np.sqrt(delta_T))
    # generate batchsize many paths of Brownian motion (note that W_0 = 0, so we have to append zero for the first entry)
    W = np.concatenate((np.zeros([batchsize, 1]), delta_W.cumsum(axis=1)), 1)
    # generate equidistant grid from 0 to T with discretization_step + 1 many entries
    t = np.arange(0, T + delta_T, delta_T)
    # generate batchsize many paths of a geometric Brownian motion
    stock = s0 * np.exp((drift - 0.5 * sigma**2) * t + sigma * W)

    return stock

# Use for every path a unique simulation of random numbers
# this means batchsize paths times insurance_pf_size random variables


def generate_survival_process(batchsize, delta_T, insurance_pf_size, T, average_age):
    # generate exponential distributed death times
    tau = np.random.exponential(scale=average_age, size=[
                                batchsize, insurance_pf_size])
    d = [np.greater(tau, u) for u in np.arange(0, T+delta_T, delta_T)]
    # generate survival process
    Nt = np.transpose(np.sum(d, axis=2))

    return Nt


def d1(S, K, ttm, r, sigma):
    return (np.log(S / K) + (r + sigma ** 2 / 2.) * ttm) / (sigma * np.sqrt(ttm))


def d2(S, K, ttm, r, sigma):
    return d1(S, K, ttm, r, sigma) - sigma * np.sqrt(ttm)

# define the t-price pi_t of a call option in the Black-Scholes model


def black_scholes(S, K, ttm, r, sigma):
    return S * norm.cdf(d1(S, K, ttm, r, sigma)) - K * np.exp(-r * ttm) * norm.cdf(d2(S, K, ttm, r, sigma))

# define derivation of the t-price of a call option in the Black-Scholes model with respect to S (delta hedge)


def black_scholes_delta(S, K, ttm, r, sigma):
    return norm.cdf(d1(S, K, ttm, r, sigma))

# calculate the profit and loss wrt a trading strategy h


def profit(h, stock):
    return torch.sum(h*(stock[:, 1:]-stock[:, :-1]), dim=1)

# analytic insurance hedge


def insurance_delta(S, K, ttm, r, sigma, average_age, Nt):
    e_lambda = np.exp(-ttm*(1/average_age))
    a = torch.from_numpy(Nt * e_lambda)
    b = torch.from_numpy(black_scholes_delta(S, K, ttm, r, sigma))
    return a * b
