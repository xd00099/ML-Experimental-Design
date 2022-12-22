################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file
from neuralnet import *
from data import generate_minibatches
from tqdm import tqdm
import copy
import pandas as pd


def train(x_train, y_train, x_val, y_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    if experiment not in ['L1', 'L2']:
        config['L2_penalty'] = None
    
    # early stopping set-up
    patient_pass = config['early_stop_epoch']
    pass_count = 0
    total_epochs = 0

    model = NeuralNetwork(config=config, experiment=experiment)

    # run epochs
    for i in tqdm(range(config['epochs'])):
        # run mini batches
        for minibatch_x, minibatch_y in generate_minibatches((x_train, y_train), config['batch_size']):
            pred, loss = model(minibatch_x, minibatch_y)
            model.backward()
        # vali loss
        vali_loss_epoch, vali_acc_epoch = test(model, x_val, y_val)
        # add loss for regularization
        if config['L2_penalty'] is not None:
            reg_loss = compute_regularization_loss(model, config, experiment)
            loss += reg_loss
            vali_loss_epoch += reg_loss

        # train loss
        train_loss_epoch, train_acc_epoch = test(model, x_train, y_train)
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)

        val_loss.append(vali_loss_epoch)
        val_acc.append(vali_acc_epoch)
        print(train_acc_epoch, vali_acc_epoch)

        total_epochs += 1

        if config['early_stop']:
            # early stop
            if i > 0:
                if val_loss[-1] > val_loss[-2]:
                    pass_count += 1
                else:
                    pass_count = 0
                if pass_count == patient_pass:
                    break
    
    #  train on the entire train set and save best model 
    best_model = train_best(np.vstack((x_train,x_val)), np.vstack((y_train,y_val)),
     config=config, experiment=experiment, epochs=total_epochs)

    # return train_acc, val_acc, train_loss, val_loss, best_model
    return train_acc, val_acc, train_loss, val_loss, best_model

## helper function that computes the regularization term for the loss
def compute_regularization_loss(model, config, reg_type):

    loss_reg = 0
    for l in model.layers:
        if l.type == 'units':
            if reg_type == 'L1':
                loss_reg += config['L2_penalty']*np.sum(np.abs(l.w))

            elif reg_type =='L2':
                loss_reg += config['L2_penalty']*(np.sum(l.w**2))
    return loss_reg

# helper to train the model on entire train set
def train_best(x_train, y_train, config, experiment, epochs):

    model = NeuralNetwork(config=config, experiment=experiment)
    for i in tqdm(range(epochs)):
        for minibatch_x, minibatch_y in generate_minibatches((x_train, y_train), config['batch_size']):
            pred, loss = model(minibatch_x, minibatch_y)
            model.backward()


    return copy.deepcopy(model)

def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    pred, loss = model(x_test, y_test)
    acc = compute_accuracy(y_test, pred)
    return loss, acc
    


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results/train_mlp/results.pkl', data)


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    activations = ['sigmoid', 'ReLU']
    for activation in activations:
        config['activation'] = activation
        train_acc, valid_acc, train_loss, valid_loss, best_model = \
            train(x_train, y_train, x_val, y_val, config)
        test_loss, test_acc = test(best_model, x_test, y_test)

        print("Config: %r" % config)
        print("Test Loss", test_loss)
        print("Test Accuracy", test_acc)

        # DO NOT modify the code below.
        data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
                'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

        write_to_file(f'./results/activation_exp/{activation}_results.pkl', data)

def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    # config['layer_specs'] = [784, 8, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    # # write_to_file(f'./results/topology/topology1_1_results.pkl', data)


    # config['layer_specs'] = [784, 16, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # print("Config: %r" % config)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    
    # config['layer_specs'] = [784, 32, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # print("Config: %r" % config)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    # config['layer_specs'] = [784, 128, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # print("Config: %r" % config)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    
    config['layer_specs'] = [784, 8, 8, 10]
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)
    test_loss, test_acc = test(best_model, x_test, y_test)
    print("Config: %r" % config)
    train_loss, train_acc = test(best_model, x_train, y_train)
    print("Config: %r" % config)
    print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    
    config['layer_specs'] = [784, 16, 16, 10]
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)
    test_loss, test_acc = test(best_model, x_test, y_test)
    print("Config: %r" % config)
    train_loss, train_acc = test(best_model, x_train, y_train)
    print("Config: %r" % config)
    print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    # config['layer_specs'] = [784, 64, 64, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # print("Config: %r" % config)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    
    # config['layer_specs'] = [784, 128, 128, 10]
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    # test_loss, test_acc = test(best_model, x_test, y_test)
    # print("Config: %r" % config)
    # train_loss, train_acc = test(best_model, x_train, y_train)
    # print("Config: %r" % config)
    # print(f"Train Acc: {train_acc}, Train Loss: {train_loss}.")
    # print(f'Test Acc: {test_acc}, Test Loss: {test_loss}')
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests the neural network with regularization.
    """
    for reg in ['L1', 'L2']:
        train_acc, valid_acc, train_loss, valid_loss, best_model = \
            train(x_train, y_train, x_val, y_val, config, experiment=reg)
        test_loss, test_acc = test(best_model, x_test, y_test)

        lambda_ = config['L2_penalty']
        print("Config: %r" % config)
        print("Test Loss", test_loss)
        print("Test Accuracy", test_acc)
        
        # DO NOT modify the code below.
        data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
                'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
        
        write_to_file(f'./results/regularization/{reg}_{lambda_}_results.pkl', data)

def check_gradients(x_train, y_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    def check_one_weight(layer_num, weight_loc, bias):
        model = NeuralNetwork(config=config)
        e = 1e-2
        x = x_train[[1]]
        y = y_train[[1]]
        input_l = model.layers[layer_num]
        if bias == True:
            w_ = input_l.b[weight_loc]

            # w+e
            input_l.b[weight_loc] = w_ + e
            loss_e_plus = model(x, y)[1]
            # w-e
            input_l.b[weight_loc] = w_ - e
            loss_e_minus = model(x, y)[1]

            input_l.b[weight_loc] = w_
        else:
            w_ = input_l.w[weight_loc]

            # w+e
            input_l.w[weight_loc] = w_ + e
            loss_e_plus = model(x, y)[1]
            # w-e
            input_l.w[weight_loc] = w_ - e
            loss_e_minus = model(x, y)[1]

            input_l.w[weight_loc] = w_
        
        
        approximate = -(loss_e_plus - loss_e_minus) / (2*e)
        model(x, y)
        model.backward()
        if bias:
            actual = model.layers[layer_num].d_b[weight_loc]
        else:
            actual = model.layers[layer_num].d_w[weight_loc]
        diff = abs(actual-approximate)

        return [approximate, actual, diff]

    check_gd_table = pd.DataFrame(columns=['Weight Type', 'Approximate GD', 'Actual GD', 'Difference'])
    # one output bias weight
    check_gd_table.loc[len(check_gd_table)] = ['Output bias'] + check_one_weight(2, (0,2), True)
    # one hidden bias weight
    check_gd_table.loc[len(check_gd_table)] = ['Hidden bias'] + check_one_weight(0, (0,1), True)
    # two hidden to output weights
    check_gd_table.loc[len(check_gd_table)] = ['Hidden-output1'] + check_one_weight(2, (1,1), False)
    check_gd_table.loc[len(check_gd_table)] = ['Hidden-output2'] + check_one_weight(2, (2,1), False)
    # two input to hidden weights
    check_gd_table.loc[len(check_gd_table)] = ['Input-hidden1'] + check_one_weight(0, (1,1), False)
    check_gd_table.loc[len(check_gd_table)] = ['Input-hidden2'] + check_one_weight(0, (2,5), False)

    print(check_gd_table)

def compute_accuracy(y, pred):
    acc = np.argmax(y, axis=1) == np.argmax(pred, axis=1)
    return np.mean(acc)