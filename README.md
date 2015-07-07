# stratipy
Patients stratification with Graph-regularized Non-negative Matrix Factorization (GNMF) in Python.

## Roadmap
- Generation of simulated patients datasets
- Implementation and validation of a GNMF algorithm
- Stratification with consensus clustering
- Comparison with NMF, Sparse-NMF, and hierarchical clustering
- Measure of the impact of parameters (e.g. network smoothing)

## Links
- [Network Based Stratification (NBS)](http://chianti.ucsd.edu/~mhofree/wordpress/?page_id=26): Matlab code & data sets.
- [NMF on Manifold (Graph), Deng Cai](http://www.cad.zju.edu.cn/home/dengcai/Data/GNMF.html): Matlab code & data sets.
- [NMF Toolbox](https://sites.google.com/site/nmftool/): Matlab code.
- [nimfa library](http://nimfa.biolab.si/): Python code.

## References
- Network-based stratification of tumor mutations ([Hofree et al. Nat. Meth. 2013](http://www.nature.com/nmeth/journal/v10/n11/full/nmeth.2651.html))
- Limited-Memory Fast Gradient Descent Method for GNMF ([Guan et al. PLoS ONE 2014](http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0077162))
- Consensus clustering: a resampling-based method for class discovery and visualization of gene expression microarray data ([Monti et al. Mach. Learn.2003](http://link.springer.com/article/10.1023%2FA%3A1023949509487))

## Notes
- Use the Matlab2Python.m script in the "tools" folder if you want to generate by yourself the data from the Hofree et al. paper (tested with Matlab R2013a).
