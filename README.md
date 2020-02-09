Sparse Projection Pursuit (SPPA)
=====================
<h1 align="center">
<img src="https://S-Driscoll.github.io/img/Graph_abs.png" alt="SPPA" width="500"/>
</h1>

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


