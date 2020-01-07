Sparse Projection Pursuit (SPPA)
=====================

<img src="https://S-Driscoll.github.io/src/common/GraphAbs.png" alt="SPPA" width="500" align="middle"/>

Manuscript
-------------

Manuscript can be found here: [Sparse Projection Pursuit](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b03166)

Running SPPA 
-------------

The MATLAB function `SPPA.m` implements the proposed SPPA algortihm.

Example from `demo.m`
```
 load Salmon.mat 
% Rows of X contain spectra, class contains the sample classes, chemshift contains ppm axis

[T,V,Vars]=SPPA(X,'dim',2,'nvars',5,'meth','mul')

```


