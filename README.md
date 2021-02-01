Sparse Projection Pursuit (SPPA)
=====================
<h1 align="center">
<img src="https://S-Driscoll.github.io/img/Graph_abs.png" alt="SPPA" width="400"/>
</h1>

Kurtosis-based projection pursuit analysis (PPA) was developed as an alternative exploratory data analysis algorithm. Instead of using variance and distance metrics to obtain, hopefully, informative projections of high-dimensional data (like PCA, HCA, and kNN), ordinary PPA searches for interesting projections by optimizing the kurtosis. However, if the sample-variable ratio is too low, it is possible for ordinary PPA to "overmodel" the data by finding spurious combinations of the original variables that give a low kurtosis value. To overcome this, one can compress their data with PCA prior to applying PCA (~10:1 sample-to-variable ratio). To make PPA independent of PCA, we have developed a sparse implementation of PPA (SPPA), where subsets of the orgiinal variables are selected using a gentic algorithm. This repository contains MATLAB code that can be used to apply SPPA to high-dimensional chemical data, examples of SPPA in use, and the corresponding paper published on SPPA. Below is a figure from our recent paper that shows the basic approach of the algorithm.

<h1 align="center">
<img src="https://S-Driscoll.github.io/img/alg.png" alt="Sparse Projection Pursuit" width="400"/>
</h1>

MATLAB Function
----------

* `SPPA.m` is a MATLAB function to perform sparse kurtosis-based projection pursuit using a genetic algorithm.

Citing this algorithm
----------
Please cite [Sparse Projection Pursuit Analysis: An Alternative for Exploring Multivariate Chemical Data (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b03166)

Literature related to PPA
-------------

* [Fast and simple methods for the optimization of kurtosis used as a projection pursuit index (2011)](https://doi.org/10.1016/j.aca.2011.08.006)
* [Re‐centered kurtosis as a projection pursuit index for multivariate data analysis (2013)](https://doi.org/10.1002/cem.2568)
* [Regularized projection pursuit for data with a small sample-to-variable ratio (2014)](https://link.springer.com/article/10.1007/s11306-013-0612-z)
* [Procrustes rotation as a diagnostic tool for projection pursuit analysis (2015)](https://doi.org/10.1016/j.aca.2015.03.006)
* [Projection pursuit and PCA associated with near and middle infrared hyperspectral images to investigate forensic cases of fraudulent documents (2017)](https://doi.org/10.1016/j.microc.2016.10.024)

Literature related to SPPA
-------------
* [Sparse Projection Pursuit Analysis: An Alternative for Exploring Multivariate Chemical Data (2020)](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b03166)

Examples 
-------------
To be completed. Please check `demo.m` for a quick demonstration showing the use of SPPA to explore a Salmon plasma data set (Nuclear Magnetic Resonance (NMR) Spectroscopy).

<!---### Salmon clustering using Nuclear Magnetic Resonance (NMR) Spectroscopy
This example uses data found in the mat file `Salmon.mat` in this repository with an example analysis using SPPA found in `demo.m`--->

