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

The dataset can be downloaded by running a dedicated script in folder ```/exp/data```:

```
get_data.sh
```

Alternatively, download from [here](https://filesender.belnet.be/?s=download&token=57b4bc25-bde4-4fd1-be15-bf0a6e4ee797), unzip the data files in folder ```/exp/data``` and the resultes files in folder ```/exp/results```.

## Main experiments

Training of models with cross-validation, calculation of attack performance metrics and activations analysis are implemented in the Jupyter Notebooks:

```bash
main.ipynb          # train with cross-val and test
plot.ipynb          # calculate metrics and plot results
activations.ipynb   # explainability
```

Our DeepCorr evaluation builds upon the following [public implementation](https://github.com/woodywff/deepcorr), which we modified to enable k-fold cross-validation, early stopping based on Average Precision (AP), and other functionality.

## Use docker to launch jupyter server and run python notebooks

This requires [Docker](https://docs.docker.com/engine/install/). First, ensure you have downloaded the data. To initialize the experiments, simply run:

```bash
docker build -t tor-notebook .

docker run --rm -it \
--mount src="$(pwd)"/exp/data,target=/app/exp/data,type=bind \
-p 8888:8888/tcp \
tor-notebook
```
The Jupyter Notebooks can be now accessed in the browser on ```127.0.0.1:8888```.

## Train with cross-validation

Use ```main.ipynb``` to train DeepCorr models with optimal neural network configurations given in the followig yaml files:

```bash
config-8k-M.yml # model configuration for multi-proxy data
config-8k-S.yml # model configuration for single-proxy data 
```

As a result, multiple keras models (for each cross-validation fold and seed) will be created in a folder with path provided under ```log_path``` in each config file. These are the trained DeepCorr models for single- and multi-proxy datasets.

Next, the notebook generates test datasets with N * N traffic traces for every fold and evaluates every trained model. Model predictions and ground truth labels are then saved in files in directory ```results/```.

## Test, calculate metrics and plot results

Use ```plot.ipynb``` to calculate attack metrics: AP, Precision, and optimized TPR and FPR. Their values are displayed directly in the notebook.

Next, the convergence process of single- and multi-proxy models is compared by plotting validation AP and validation loss and multiple models.

## Explainability 

Our paper performs preliminary explainability analysis by plotting activation maps of the first layers of the models. 

Use ```activations.ipynb``` to compute and plot activation maps of single- and multi-proxy models.

## License

All our code is licensed as free software, under the GPLv3 license.

