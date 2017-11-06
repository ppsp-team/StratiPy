# StratiPy <img src="https://img.shields.io/badge/version-0.7.0-blue.svg"> <a href="https://travis-ci.org/GHFC/StratiPy"><img src="https://travis-ci.org/GHFC/StratiPy.svg?branch=master"></a> [![License](https://img.shields.io/badge/license-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1042546.svg">
Patients stratification with Graph-regularized Non-negative Matrix Factorization (GNMF) in Python.

## Dependencies
The minimum required dependencies to run the software are:
  - Python >= 3.4
  - NumPy >= 1.6
  - SciPy >= 0.7.2
  - matplotlib >= 0.98.4
  - scikit-learn >= 0.19
  - pandas >= 0.1
  - tqdm>=4.15.0

If you want to launch Reproducibility project by Jupyter Notebook, you will also need:
  - ipython>=1.1.0
  - jupyter


## Reproducibility
To reproduce the figure 3 of [our paper about reproducibility in bioinformatics](https://www.biorxiv.org/content/early/2017/10/31/143503), you have two options:
1. Follow the Jupyter Notebook [reproducibility.ipynb](reproducibility/reproducibility.ipynb)
2. Build & run the [Docker](http://docker.com) container with:
```
~$ mkdir <your_output_folder_outside_stratipy_folder>
~$ docker build -t repro .
~$ docker run -v /absolute/path/of/<your_output_folder_outside_stratipy_folder>:/reproducibility/reproducibility_output repro
```
**For Windows or Mac users:** total runtime memory of Docker is fixed to 2 GB by default. In order to launch this project, you have to increase this limit (approximately 7 GB):
- [Windows setting](https://docs.docker.com/docker-for-windows/#advanced)
- [Mac setting](https://docs.docker.com/docker-for-mac/#cpus)


## References
- Network-based stratification of tumor mutations ([Hofree et al. Nat. Meth. 2013](http://www.nature.com/nmeth/journal/v10/n11/full/nmeth.2651.html))
- Consensus clustering: a resampling-based method for class discovery and visualization of gene expression microarray data ([Monti et al. Mach. Learn.2003](http://link.springer.com/article/10.1023%2FA%3A1023949509487))
- Experimenting with reproducibility in bioinformatics ([Kim et al. BioRxiv 2017](https://www.biorxiv.org/content/early/2017/10/31/143503))


## Additional links
- [Network Based Stratification (NBS)](http://chianti.ucsd.edu/~mhofree/wordpress/?page_id=26): Matlab code & data sets.
- [NMF on Manifold (Graph), Deng Cai](http://www.cad.zju.edu.cn/home/dengcai/Data/GNMF.html): Matlab code & data sets.
- [NMF Toolbox](https://sites.google.com/site/nmftool/): Matlab code.
- [nimfa library](http://nimfa.biolab.si/): Python code.


## Licensing
Stratipy is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2014, authors of Stratipy
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of Stratipy authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**
