# GDEC

[![Test (local)](https://github.com/cdgreenidge/gdec/actions/workflows/test_local.yml/badge.svg)](https://github.com/cdgreenidge/gdec/actions/workflows/test_local.yml) [![Anaconda-Server Badge](https://anaconda.org/cdg4/gdec/badges/version.svg)](https://anaconda.org/cdg4/gdec) [![PyPI version](https://badge.fury.io/py/gdec.svg)](https://badge.fury.io/py/gdec)

GDEC (grating decoders) is a package that provides linear decoders for decoding circular stimuli, e.g. the angle of a drifting grating, from a vector of spike counts. It accompanies the paper

> Greenidge, C. Daniel, Benjamin Scholl, Jacob Yates, and Jonathan W. Pillow. "Efficient decoding of large-scale neural population responses with Gaussian-process multiclass regression". Under review.

All decoders use the scikit-learn interface, so using them is as easy as

```python
import gdec


model = gdec.GaussianProcessMulticlassDecoder()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

Currently we supply implementations of the following decoders: 

- Gaussian Process Multiclass Decoder (GPMD, this paper)
- Gaussian Independent Decoder / Gaussian Naive Bayes (GID)
- Gaussian Process-regularized (GPGID, this paper)
- Poisson Independent Decoder / Poisson Naive Bayes (PID)
- Gaussian Process-regularized (GPPID, this paper)
- Empirical Linear Decoder (ELD, Graf et. al. 2011)
- Elastic Net Logistic Regression (GLMNET)
- Super-Neuron Decoder (SND, Stringer et. al. 2019)

## Installation

> **NOTE:** The ELD decoder depends on [Jax](https://github.com/google/jax), and the VGPMD decoder depends on [Pytorch](https://pytorch.org/). CPU-only versions of these packages will be installed automatically if they aren't already found. If you have a compute accelerator and want to use it, install CUDA-enabled versions of either Jax or Pytorch before moving on.

```python
conda install -c cdg4 gdec  # For conda users
pip install gdec  # For pip users
```

## Basic Usage

Each decoder conforms to the scikit-learn interface. You can initialize them
as follows:

```python
import gdec

model = gdec.GausianProcessMulticlassDecoder()  # GPMD  
model = gdec.GaussianIndependentDecoder()  # GID
model = gdec.GPGaussianIndepndentDecoder()  # GPGID
model = gdec.PoissonIndepndentDecoder()  # PID
model = gdec.GPPoissonIndepndentDecoder()  # GPPID
model = gdec.EmpiricalLinearDecoder()  # ELD
model = gdec.LogisticRegression()  # GLMNET
model = gdec.SuperNeuronDecoder()  # SND

model.fit(X_train, y_train)
```

Training data requirements:

- X: shape `(num_observations, num_neurons)`, with each row containing a vector of binned spike counts or summed calcium activations. It should be integer-valued for the PID and GPPID and real-valued for everything else. For the ELD and GLMNET decoders, it's helpful to standardize it to zero mean and unit variance.
- y: shape `(num_observations, )`. Assuming your stimulus angles are binned into `K` bins, each entry of the `y` vector should be an integer class label between `0` and `K - 1` corresponding to the index bin of the stimulus angle for that entry.

Once you've collected your data, you can train and predict with the classifiers just as you would with any scikit-learn classifier.

```
model.train(X_train, y_train)
y_pred_probs = model.predict_proba(X_test)
y_pred = model.predict(X_test)
```

## Utility functions

Most measures of decoder performance are interested in the distance between two classes, as measured around the unit circle. We provide a utility function, `gdec.circdist`, to help you calculate this.

```python
def circdist(x: np.ndarray, y: np.ndarray, circumference: float) -> np.ndarray:
    """Calculate the signed elementwise circular distance between two arrays.

    Returns positive numbers if y is clockwise compared to x, negative if y is counter-
    clockwise compared to x.

    Args:
        x: The first array.
        y: The second array.
        circumference: The circumference of the circle (aka the number of bins)

    Returns:
        An array of the same shape as x and y, containing the signed circular distances.

    """
```

## Advanced usage

If you wish to inspect the weight matrix of a decoder, use the `coefs_` property:

```
    W = model.coefs_
```

Some decoders accept keword arguments to the `.fit()` method which allow you to tweak performance:

- **GPMD**
    - `lr: float = 0.1`: the learning rate
    - `max_steps: int = 4096`: the number of optimization steps to take
    - `n_samples: int = 4`: the number of samples with which to approximate the variational expectation
    - `log_every: int = 32`: how often (in steps) to print a status update
    - `cuda: bool = True`: whether or not to use CUDA
    - `cuda_device: int = 0`: which CUDA device to use
- **GPGID**
    - `verbose: bool = True`: Whether or not to print a progress bar.
- **GPPID**
  -  `verbose: bool = True`: Whether or not to print a progress bar.
- **GLMNET**
  - This class is a thin wrapper around [scikit-learn's GLMNET implementation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), so it can take any argument that the scikit-learn model does.

## References

Graf, Arnulf B. A., Adam Kohn, Mehrdad Jazayeri, and J. Anthony Movshon. 2011. “Decoding the Activity of Neuronal Populations in Macaque Primary Visual Cortex.” Nature Neuroscience 14 (2): 239–45.

Stringer, Carsen, Michalis Michaelos, and Marius Pachitariu. 2019. “High Precision Coding in Mouse Visual Cortex.” bioRxiv. https://doi.org/10.1101/679324.

