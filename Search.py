from Training import *

for num_layers in [3, 4]:
    for dim in [64, 32]:
        for activation in ['relu', 'tanh']:
            for le in [0.001]:
                for mini_batch_size in [128, 256, 500]:
                    train_with_params(num_hidden_layers=num_layers, num_hidden_dim=dim,
                                      activation_function=activation, le=le, mini_batch=mini_batch_size, num_epochs=500)
