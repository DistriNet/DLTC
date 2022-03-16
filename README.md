# Trace Oddity: Methodologies for Data-Driven Traffic Analysis on Tor

**Abstract**: Traffic analysis attacks against encrypted web traffic are a persisting problem. However, there is a large gap between the scientific estimate of attack threats and the real-world situation, specifically in terms of representative traffic characteristics and content. As traffic analysis attacks depend on very specific metadata information, they are sensitive to artificial changes in the transmission characteristics, which may occur in simulated attack scenarios. While the advent of deep learning greatly improves the performance rates of traffic analysis attacks on Tor in research settings, deep neural networks are known for being implicitly vulnerable to artifacts in data. Removing artifacts from our experimental setups is essential to minimizing the risk of evaluation bias. In this work, we study a state-of-the-art end-to-end traffic correlation attack on Tor and propose a novel data collection setup. Our design addresses the key constraint of prior work: instead of using a single proxy node for collecting exit traffic, we deploy multiple proxies. Our extensive analysis shows that in the multi-proxy design (i) end-to-end round-trip times are more realistic than in the original design, and that (ii) traffic correlation attack performance degrades significantly on realistic timings. For a reliable and informative evaluation, we develop a general scientific methodology for replication and comparison of machine and deep-learning attacks on Tor. Our evaluation indicates high relevance of the multi-proxy data collection setup and the novel dataset.

## Requirements
The code has been tested on: 
* Python 3.8
* [Tensorflow 2.3.1](https://www.tensorflow.org/install)

Additionally, the (optional) pipeline for hyperparameter optimization uses:
* [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
* [Ax](https://ax.dev/)

## Dataset

The dataset can be downloaded by running:

```
get_data.sh
```

Alternatively, download from [here](https://filesender.belnet.be/?s=download&token=4f5b2172-f6e3-4d3d-bd8a-1ffda8ee4d16) and paste the files in folder ```/exp/data```.

## Setup hyperparameter optimization pipeline

```bash
git clone TODO
git checkout hyperopt
pip3 install -r ray-tuning/requirements.txt
```

Start a ray cluster. For a distributed setting, instructions on how to create a cluster can be found [here](https://docs.ray.io/en/latest/tune/tutorials/tune-distributed.html)

```bash
ray start --head
```

## Define and run hyperopt experiments

```bash
cd ray-tuning
python3 main.py ${CONFIG_FILE} --gpu=${0|1} --preprocess=${True|False}
```

The results can be accessed with [tensorboard](https://www.tensorflow.org/tensorboard) by running:

```bash
tensorboard --logdir ${EXP_FOLDER} 
```

## Run a demo hyperopt experiment

It requires [Docker](https://docs.docker.com/engine/install/). First, ensure you have downloaded the data. To initialize the experiments, simply run:

```bash
docker-compose -f "docker-compose.yaml" up -d --build 
```
The results can be accessed with tensorboard on ```127.0.0.1:6016```

For CUDA capable machines (see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) how to configure), GPU training can be enabled on the demo by uncommenting the lines in ```docker-compose.yaml``` and setting ```gpu:1``` on the arguments.


## Defining a custom experiment

### Creating a custom configuration.

Custom experiments can be created with the syntax of ax-platform search and using the given configuration files as a template. Examples can be found in the ```exp/conf``` folder. Please be aware that the configuration files for hyperparameter tuning are different than those used for testing the model. The parameters that can be defined are:

* exp: Global experiment parameters.
    * exp_name(str):  Name of the experiment for ray logging
    * exp_path(str):  Output path of results. This path can be used with tensorboard to see the results.
    * optim_metric(str):  Main metric used for early stopping and model comparison. Must be logged by ray.
    * sched_grace_period(int): Grace period for the scheduler. See https://docs.ray.io/en/latest/tune/api_docs/schedulers.html for details.
    * num_samples(int): Number of experiments or trial runs to be examined.
    * epochs (int): Max number of epochs per experiment
    * num_repeat(int): Number of times a particular experiment is run.
* data: Dataset parameters
    * dataset_path (str): Path for locating the data
    * flow_size (int): Size of the flows to be used
    * pickle_path (str): Path of the pickle file to be generated
    * h5_path (str): Path of the h5 file to be generated
    * n_neg_per_pos (int): Number of unpaired flows generated from each paired flows
    * ratio_train (float):  Proportion of dataset for training
    * n_fold (int): nNumber of fold for cross validation
    * crossval_indices_path (str): Path for the cross validation files
    * log_path (str): Path for tensorflow logging
    * seed (int): Default seed to be used
* train: Default training parameters
    * batch_size (int): Default size of the mini-batches for training. Ignored if hyperopt is used.
    * conv_filters (\[int\]): Default filters for the convolutional layers. Ignored if hyperopt is used.
    * dense_layers (\[int\]): Default dense layers for the sequential model. Ignored if hyperopt is used.
    * drop_p (int): Default dropout probability. Ignored if hyperopt is used.
    * lr: (int): Default learning rate. Ignored if hyperopt is used.
* test: Default testing parameters
    * batch_size (int): Default size of the mini-batches for testing.
* hyperopt: Hyperparameters configuration. The syntax of ax-platform is used. Examples of how to use ax with ray can be found [here](https://ax.dev/tutorials/raytune_pytorch_cnn.html)
    * batch_size (\[int\]): Size of the mini-batches for training. 
    * conv_layers_choice (\[int\]): Index for the filters for the convolutional layers defined in hyperopt_utils.
    * dense_layers_choice (\[int\]): Index for the dense layers for the sequential model defined in hyperopt_utils.
    * drop_p (\[float\]): Dropout probability. 
    * lr: (\[float\]): Learning rate. 
    * pos_weight: (\[int\]): weight of samples for imbalanced classification. More details can be found [here](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits).
* hyperopt_utils: Actual values for the index defined before.
    * conv_layers: Filters for the convolutional layers. Actual used value is ```conv_layers[conv_layers_choice]``` 
    * dense_layers: Dense layers for the sequential model. Actual used value is ```dense_layers[dense_layers_choice]```
    * max_pool_sizes: Size of the max pooling window. Actual used value is ```conv_layers[conv_layers_choice]``` 
    * kernel_sizes: Dimensions of the convolution filters. Actual used value is ```conv_layers[conv_layers_choice]``` 
    * strides: Offset of the pooling window each step. Actual used value is ```conv_layers[conv_layers_choice]``` 

## License

All our code is licensed as free software, under the GPLv3 license.
