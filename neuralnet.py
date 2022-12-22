################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None
        self.type = 'activation'

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        self.x = a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - np.tanh(self.x)**2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return np.greater(self.x, 0).astype(int)


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.prev_d_w = None
        self.prev_d_b = None
        self.type = 'units'
        self.first_train = True

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x) 

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x@self.w + self.b

        return self.a

    def backward(self, delta, lr, batch_size, lambda_, experiment, gamma):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # save the previous gradient for momentum
        self.prev_d_w = self.d_w
        self.prev_d_b = self.d_b

        # update gradient
        if experiment == 'L2':
            self.d_w = self.x.T@delta
        elif experiment == 'L1':
            self.d_w = self.x.T@delta 
        else:
            self.d_w = self.x.T@delta
        self.d_b = np.sum(delta, axis=0, keepdims=True)
        
        # if first time, set the prevs to the first time
        if self.first_train:
            self.prev_d_w = self.d_w
            self.prev_d_b = self.d_b
            self.first_train = False
        
        # calculate delta
        self.d_x = delta@self.w.T


        # Momentum
        if gamma is not None:
            self.d_w = gamma*(self.prev_d_w) + (1-gamma)*self.d_w
            self.d_b = gamma*(self.prev_d_b) + (1-gamma)*self.d_b

            # Momentum with Regularization
            if lambda_ is not None:
                ## L2
                if experiment == 'L2':
                    self.w += lr*self.d_w / batch_size - lambda_*self.w
                ## L1
                else:
                    self.w += lr*self.d_w / batch_size - lambda_*np.sign(self.w)

            # Momentum No Regularization
            else:
                self.w += lr*self.d_w / batch_size
                self.b += lr*self.d_b / batch_size

        # No Momentum
        else:
            self.w += lr*self.d_w / batch_size
            self.b += lr*self.d_b / batch_size

        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config, experiment=None):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.lambda_ = config['L2_penalty']
        self.experiment = experiment
        self.gamma = config['momentum_gamma']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.targets = targets
        self.x = x
        
        input = x
        for i in range(len(self.layers)):
            output = self.layers[i].forward(input)
            input = output

        self.y = self.softmax(output)
        
        if targets is not None:
            loss = self.loss(self.y, targets)
            return self.y, loss
        
        return self.y


    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        delta = self.targets - self.y
        for i in range(len(self.layers)):
            if self.layers[::-1][i].type == 'units':
                delta = self.layers[::-1][i].backward(delta, self.lr, self.batch_size, self.lambda_, self.experiment, self.gamma)
            else:
                delta = self.layers[::-1][i].backward(delta)

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        _sum = np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / _sum

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        
        return -np.sum(targets*np.log(logits)) / len(targets)
