# Damit der Code läuft, muss die PATH Variable angepasst werden, außerdem muss in dem entsprechendem Verzeichnis
# eine Datei 'output.txt' existieren.

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
#from torch import nn
from tqdm import tqdm
import pandas as pd
import time
import os

from Functions import *
from Network import *

# Training auf GPU
gpu_id = 1
device = torch.device(
    f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

## Generate Data  #############################################################
###############################################################################

# initial stock price
s0 = 100.

# Time horizon
T = 1

# volatility of the stock
sigma = 0.4

# drift of the stock
drift = 0.0

# discretization of time horizon T
discretization_steps = 200
delta_T = T/discretization_steps

# Plot paths of the geometric Brownian motion (just 5 paths)
# stock = generate_stock_prices(
#    5, discretization_steps, delta_T, drift, sigma, T, s0)
# sns.lineplot(data=np.transpose(stock))
#name = "{}/Stock_prices.png".format(PATH)
#plt.savefig(name, dpi=200)
# plt.close()

# interest rates
r = 0

# strike of the call
K = 100

# insurance Portfoliosize
insurance_pf_size = 100

# average age of insured person
average_age = 2

# Decide if the data for people who are still alive is given by relative or absolute
# True (=relative), False (=absolut)
normalize_survival = False

# train/test split
# train data (10000 paths)
batchsize = 10000
stock_train = torch.from_numpy(generate_stock_prices(
    batchsize, discretization_steps, delta_T, drift, sigma, T, s0))
survival_train = torch.from_numpy(generate_survival_process(
    batchsize, delta_T, insurance_pf_size, T, average_age))
if normalize_survival:
    survival_train = survival_train/insurance_pf_size

ttm_train = torch.from_numpy(np.ones(
    (batchsize, discretization_steps + 1)) * np.flip(np.arange(0, T + delta_T, delta_T)))
data_train_t = torch.stack(
    (stock_train, survival_train, ttm_train), dim=2).float()

# test data (1000 paths)
stock_val = torch.from_numpy(generate_stock_prices(
    1000, discretization_steps, delta_T, drift, sigma, T, s0))
survival_val = torch.from_numpy(generate_survival_process(
    1000, delta_T, insurance_pf_size, T, average_age))
if normalize_survival:
    survival_val = survival_val/insurance_pf_size

ttm_val = torch.from_numpy(np.ones(
    (1000, discretization_steps + 1)) * np.flip(np.arange(0, T + delta_T, delta_T)))
data_val_t = torch.stack((stock_val, survival_val, ttm_val), dim=2).float()


# Calculate Price at t=0 
if normalize_survival:
    pi_ins = np.exp(-T*(1/average_age))*black_scholes(s0, K, T, r, sigma)
else:
    pi_ins = insurance_pf_size * \
        np.exp(-T*(1/average_age))*black_scholes(s0, K, T, r, sigma)

## loss function ##############################################################
###############################################################################

def loss_fn(hedge, stock, n):
    return torch.mean((profit(hedge, stock) + pi_ins - n * torch.maximum(stock[:, -1]-K, torch.zeros_like(stock[:, -1])))**2)

## Analytic Hedge #############################################################
###############################################################################

test_size = 5000

stock_test = generate_stock_prices(
    test_size, discretization_steps, delta_T, drift, sigma, T, s0)
survival_test = generate_survival_process(
    test_size, delta_T, insurance_pf_size, T, average_age)
if normalize_survival:
    survival_test = survival_test/insurance_pf_size

# value of the call at time T in batchsize many scenarios
callT = np.maximum((stock_test[:, -1] - K), 0)
# value of the call at time T in batchsize many scenarios times people who survived
ins_callT = survival_test[:, -1] * callT

# price at time zero
if normalize_survival:
    pi_ins = np.exp(-T*(1/average_age))*black_scholes(s0, K, T, r, sigma)
else:
    pi_ins = insurance_pf_size * \
        np.exp(-T*(1/average_age))*black_scholes(s0, K, T, r, sigma)

ttm = np.flip(np.arange(0, T + delta_T, delta_T))

# delta hedge for the call option (strategy does not need the last entry of the stock)
h_insurance = insurance_delta(
    stock_test[:, :-1], K, ttm[:-1], r, sigma, average_age, survival_test[:, :-1])

# portfolio value
tradingT = profit(h_insurance, torch.from_numpy(stock_test)) + pi_ins

# difference of portfolio value and call value at time T
error_ana = (tradingT - ins_callT).numpy()

loss_insurance = loss_fn(h_insurance, torch.from_numpy(
    stock_test), torch.from_numpy(survival_test[:, -1])).numpy()


def train_with_params(num_hidden_layers=3, num_hidden_dim=64, activation_function='relu', le=0.001, mini_batch=128, num_epochs=500):

    name = f'{num_hidden_layers}_hidden_{activation_function}_dim_{num_hidden_dim}_le_{le}_minibs_{mini_batch}_{num_epochs}_epochs'
    # Specify a path
    PATH = f'/home/doering/Code/saves/{name}'

    # create folder to save results
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Create log file
    txtfile = open(f"{PATH}/log.txt", "w")

    # instantiate model
    number_of_hiddenlayer = num_hidden_layers
    hidden_dimension = num_hidden_dim
    dimension_input = 3

    print('Hidden Layer: ', number_of_hiddenlayer, ' mit je ',
          hidden_dimension, 'Neuronen', file=txtfile)

    activation_f = activation_function  # relu or tanh
    print('Aktivierungsfunktione: ', activation_f, file=txtfile)

    model = DeltaHedgeNN(n_hidden=number_of_hiddenlayer,
                         dim_input=dimension_input,
                         hidden_dim=hidden_dimension,
                         activation=activation_f)

    model = model.to(device)

    # Load model
    #model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

    minibatch_size = mini_batch
    print('Mini Batch Größe: ', minibatch_size, file=txtfile)

    loader = torch.utils.data.DataLoader(data_train_t,
                                         batch_size=minibatch_size,
                                         shuffle=True)

    print(
        f'Network has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')

    # define number of epochs
    n_epochs = num_epochs
    print('Epochen: ', n_epochs, file=txtfile)

    # define the learning rate
    learning_rate = le
    print("Learning Rate: ", learning_rate, file=txtfile)

    # define and instantiate optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # for logging
    losses_train = []
    losses_val = []

    # GPU
    gpu_data_train_t = data_train_t.to(device)
    gpu_data_val_t = data_val_t.to(device)

    # starting time to calculate training time
    start = time.time()

    # train model
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for i, data in enumerate(loader):

            data = data.to(device)

            h_pred = model(data, train=True)
            # loss_fn(h_pred,data_train.to(device))
            loss = loss_fn(h_pred, data[:, :, 0], data[:, -1, 1])

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation after respective epoch
        h_pred = model(gpu_data_train_t, train=False)
        loss = loss_fn(
            h_pred, gpu_data_train_t[:, :, 0], gpu_data_train_t[:, -1, 1])
        h_val = model(gpu_data_val_t, train=False)
        # loss_fn(h_val,data_val.to(device))
        loss_val = loss_fn(
            h_val, gpu_data_val_t[:, :, 0], gpu_data_val_t[:, -1, 1])

        pbar.set_description(
            f'loss_train @ Epoch #{epoch+1}: {loss}; loss val {loss_val}')

        losses_train.append(loss.detach().cpu().numpy())
        losses_val.append(loss_val.detach().cpu().numpy())

    pbar.close()

    # end time
    end = time.time()
    training_time = end - start
    # total time taken
    print(f"Training time is {training_time/3600} h")
    print(f"Training time is {training_time/3600} h", file=txtfile)
    print(file=txtfile)

    df_loss = pd.DataFrame.from_dict({'train loss': np.stack(losses_train),
                                      'val loss': np.stack(losses_val), })

    fig, ax = plt.subplots()
    ax.set_title(f'Losses over {n_epochs} epochs')
    sns.lineplot(data=df_loss, ax=ax)
    name = "{}/Losses.png".format(PATH)
    plt.savefig(name, dpi=200)
    plt.close()

    print('Analytic Hedge:', file=txtfile)
    print('Hedging Error of analytic hedge', file=txtfile)
    print('Mean:', np.mean(error_ana), file=txtfile)
    print('Maximal value:', np.amax(error_ana), file=txtfile)
    print('Minimal value:', np.amin(error_ana), file=txtfile)
    print('Std deviation: ', np.std(error_ana), file=txtfile)
    print('loss:', loss_insurance, file=txtfile)
    print(file=txtfile)

    plt.hist((error_ana), bins='auto')  # arguments are passed to np.histogram
    plt.title("Analytic hedging error")
    name = "{}/Analytic hedging error.png".format(PATH)
    plt.savefig(name, dpi=200)
    plt.close()

    # Hedging with the network

    stock_test_tensor = torch.from_numpy(stock_test)
    survival_test_tensor = torch.from_numpy(survival_test)
    ttm = torch.from_numpy(np.ones((test_size, discretization_steps + 1))
                           * np.flip(np.arange(0, T + delta_T, delta_T)))
    data_test_t = torch.stack(
        (stock_test_tensor, survival_test_tensor, ttm), dim=2).float().to(device)

    h_net_test = model(data_test_t, train=False)

    net_tradingT = profit(h_net_test, data_test_t[:, :, 0]) + pi_ins
    error_net = (net_tradingT.cpu().detach() - ins_callT).numpy()
    loss_net = loss_fn(
        h_net_test, data_test_t[:, :, 0], data_test_t[:, -1, 1]).cpu().numpy()

    print('Network Hedge', file=txtfile)
    print('Hedging Error of the network', file=txtfile)
    print('Mean:', np.mean(error_net), file=txtfile)
    print('Maximal value:', np.amax(error_net), file=txtfile)
    print('Minimal value:', np.amin(error_net), file=txtfile)
    print('Std deviation:', np.std(error_net), file=txtfile)
    print('loss:', loss_net, file=txtfile)
    print(file=txtfile)

    plt.hist(error_net, bins='auto')  # arguments are passed to np.histogram
    plt.title("Network hedging error")
    name = "{}/Network hedging error.png".format(PATH)
    plt.savefig(name, dpi=200)
    plt.close()

    # arguments are passed to np.histogram
    plt.hist(error_net, bins='auto', alpha=0.5, label="network")
    # arguments are passed to np.histogram
    plt.hist((error_ana), bins='auto', alpha=0.5, label="analytic")
    plt.title("Network and analytic hedging error")
    plt.legend(loc='upper right')
    name = "{}/Network and analytic hedging error.png".format(PATH)
    plt.savefig(name, dpi=200)
    plt.close()

    # Save model
    torch.save(model.state_dict(), f'{PATH}.pt')

    # Compare with the analytic delta

    # specify data
    time_to_maturity = 0.9
    prices = np.arange(10, 300, 2)[None, :]
    n_alive_percentage = 0.8
    n_alive = n_alive_percentage * insurance_pf_size
    ttm = time_to_maturity * np.ones(prices.shape)

    survival_Nt = n_alive * np.ones(prices.shape)
    if normalize_survival:
        survival_Nt = n_alive_percentage * np.ones(prices.shape)

    prices_tensor = torch.from_numpy(prices)
    ttm_tensor = torch.from_numpy(ttm)
    survival_Nt_tensor = torch.from_numpy(survival_Nt)

    # create data_tensor
    prices_test_t = torch.stack(
        (prices_tensor, survival_Nt_tensor, ttm_tensor), dim=2).float().to(device)

    h_analytic = insurance_delta(
        prices[:, :-1], K, time_to_maturity, r, sigma, average_age, survival_Nt[:, :-1]).numpy()
    h_net = model(prices_test_t, train=False).cpu().numpy()

    df_strat = pd.DataFrame(index=prices[0, :-1], data={'Network': h_net[0, :],
                                                        'Analytic': h_analytic[0, :], })
    ax = sns.lineplot(data=df_strat)
    ax.set_title("ttm = " + str(time_to_maturity) + ", remaining portfolio size " +
                 str(n_alive_percentage*100) + " %", fontsize=15)
    ax.set_ylabel("strategy")
    ax.set_xlabel("stock prices")
    name = f"{PATH}/ttm0.9_alive0.8.png"
    plt.savefig(name, dpi=200)

    ttm_vector = [0.2, 0.5, 0.8]
    alive_percentage_vector = [0.2, 0.5, 0.8, 1]

    for u in range(len(ttm_vector)):
        for v in range(len(alive_percentage_vector)):
            time_to_maturity = ttm_vector[u]
            prices = np.arange(50, 300, 2)[None, :]
            n_alive_percentage = alive_percentage_vector[v]
            n_alive = n_alive_percentage * insurance_pf_size
            ttm = time_to_maturity * np.ones(prices.shape)

            survival_Nt = n_alive * np.ones(prices.shape)
            if normalize_survival:
                survival_Nt = n_alive_percentage * np.ones(prices.shape)
            prices_tensor = torch.from_numpy(prices)
            ttm_tensor = torch.from_numpy(ttm)
            survival_Nt_tensor = torch.from_numpy(survival_Nt)

            # create data_tensor
            prices_test_t = torch.stack(
                (prices_tensor, survival_Nt_tensor, ttm_tensor), dim=2).float().to(device)

            h_analytic = insurance_delta(
                prices[:, :-1], K, time_to_maturity, r, sigma, average_age, survival_Nt[:, :-1]).numpy()
            h_net = model(prices_test_t, train=False).cpu().numpy()
            df_strat = pd.DataFrame(index=prices[0, :-1], data={'Network': h_net[0, :],
                                                                'Analytic': h_analytic[0, :], })

            plt.figure()
            ax = sns.lineplot(data=df_strat)
            ax.set_title("ttm = " + str(time_to_maturity) + ", remaining portfolio size " +
                         str(n_alive_percentage*100) + " %", fontsize=15)
            ax.set_ylabel("strategy")
            ax.set_xlabel("stock prices")
            name = f'{PATH}/ttm{str(time_to_maturity)}_alive{str(n_alive_percentage)}.png'
            plt.savefig(name, dpi=200)
            plt.close()

    ttm_vector = [0.2, 0.5, 0.8]
    stock_price_vec = [80, 100, 150, 200]

    for u in range(len(ttm_vector)):
        for v in range(len(stock_price_vec)):
            time_to_maturity = ttm_vector[u]
            prices = stock_price_vec[v]
            n_alive_percentage = np.arange(0, 1, 0.001)[None, :]
            n_alive = n_alive_percentage * insurance_pf_size
            ttm = time_to_maturity * np.ones(n_alive.shape)
            survival_Nt = n_alive
            prices = prices * np.ones(n_alive.shape)
            if normalize_survival:
                survival_Nt = n_alive_percentage * np.ones(n_alive.shape)
            prices_tensor = torch.from_numpy(prices)
            ttm_tensor = torch.from_numpy(ttm)
            survival_Nt_tensor = torch.from_numpy(survival_Nt)

            # create data_tensor
            prices_test_t = torch.stack(
                (prices_tensor, survival_Nt_tensor, ttm_tensor), dim=2).float().to(device)

            h_analytic = insurance_delta(
                prices[:, :-1], K, time_to_maturity, r, sigma, average_age, survival_Nt[:, :-1]).numpy()
            h_net = model(prices_test_t, train=False).cpu().numpy()
            df_strat = pd.DataFrame(index=survival_Nt[0, :-1], data={'Network': h_net[0, :],
                                                                     'Analytic': h_analytic[0, :], })

            plt.figure()
            ax = sns.lineplot(data=df_strat)
            ax.set_title("ttm = " + str(time_to_maturity) +
                         ", stock prices " + str(stock_price_vec[v]), fontsize=15)
            ax.set_ylabel("strategy")
            ax.set_xlabel("people alive in %")
            name = f'{PATH}/ttm{str(time_to_maturity)}_price{str(stock_price_vec[v])}.png'
            plt.savefig(name, dpi=200)
            plt.close()

    txtfile.close()
    print('finished')


if __name__ == '__main__':
    train_with_params()
