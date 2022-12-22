# Experimental Design on KMNIST Classification

## Repository Overview
- /plots contains all training and validation diagrams in the form of png
- data.py contains functions that we used to parse data
- neuralnet.py implements out multi-layer network model
- train.py includes the training procedure that we used to train our model
- Visualization.ipynb a jupyter notebook used to generate all visualization of the testing results
- get_data.sh a script used to download dataset
- config.yaml defines the default configuration for our network model, which includes the value of hyperparameters, type of activation, etc.
- main.py the trainning pipeline is implemented in this file, which is also used to run the model
- cnn.ipynb is the CNN implementation on the task

## How to run the model
1. first clone this repository by running

```shell
$ git clone https://github.com/cse151bwi22/cse151b-wi22-pa2-arthurwty.git
$ cd cse151b-wi22-pa2-arthurwty
```

2. Then you can choose to run the model with different experiment
- MLP models and implementation
```shell
$ python main.py --train_mlp
$              | --check_gradients
$              | --regularization
$              | --activation
               | --topology
```
- CNN models
```shell
$ jupyter notebook cnn.ipynb
```
## How to see the testing result?
after the program finishes training the model, the test results will be generated in *task*_test_accuracy.txt under /xxx_experiments
