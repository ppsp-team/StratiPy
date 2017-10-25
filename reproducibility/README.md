# Reproducibility
To reproduce the figure 3 of our paper about reproducibility in bioinformatics ([Kim et al. BioRxiv 2017](http://www.biorxiv.org/content/early/2017/06/20/143503)), you have two options:
1. Follow the Jupyter Notebook [reproducibility.ipynb](reproducibility.ipynb)
2. Build & run the [Docker](http://docker.com) container with:
```
~$ mkdir <your_output_folder>
~$ docker build -t repro .
~$ docker run -v /absolute/path/of/<your_output_folder>:/reproducibility/reproducibility_output repro
```
**For Windows or Mac users:** total runtime memory of Docker is fixed to 2 GB by default. In order to launch this project, you have to increase this limit (approximately 7 GB):
- [Windows setting](https://docs.docker.com/docker-for-windows/#advanced)
- [Mac setting](https://docs.docker.com/docker-for-mac/#cpus)


## Output
You will obtain confusion matrices which compare the original (achieved by the original MATLAB code) and replicated results.


## Two adjustable parameters
There are several tuning parameters of NBS. In order to reproduce same result of the original work, you only use same values outlined in the
original NBS study ([Hofree et al. Nat. Meth. 2013](http://www.nature.com/nmeth/journal/v10/n11/full/nmeth.2651.html)) except two parameters:

1. **Graph regulator factor (lambda)** is the most influential parameter in this case study. It was thought that this factor had to be a constant value of 1 until we found that its value was changing and converged to 1800. Here you run StratiPy with lambd = 1 and 1800 in order to compare them. Lambda is initially set to these two values.
2. **Permutation number of bootstrap** utilized in the original study is 1000. But we recommand to start with 100 permutations since this step is highly time consuming (optimization ongoing ). In [Jupyter Notebook option](reproducibility.ipynb), you can notice no significant difference between 100 and 1000 permutations of original results (MATLAB).

Details about each parameter are explained in docstring of [reproducibility.py](reproducibility.py).


## Data
- General input data was downloaded from [original author's NBS website](http://chianti.ucsd.edu/~mhofree/NBS/). In the original study, authors worked on:
    - 4 types of cancer data provided by [TCGA](https://cancergenome.nih.gov/). Here we only work on uterine endometrial carcinoma (uterine cancer) with 248 patients' somatic mutation data.
    - 3 types of Protein-Protein Interaction (PPI) networks. Here we only utilize [STRING database](https://string-db.org/).
- [data](../data/) folder includes input and output data such as mutation profiles, PPI networks, similarity matrices, etc...
- In this case study, the code will take general input data from [data](../data/) folder but output will be placed in the specific [reproducibility output](reproducibility_output/) in Jupyter Notebook case. When you launch with Docker, you will create a linked output folder. Original MATLAB results (100 and 1000 permutations of bootstrap) are in [data](../data/) folder.
- Result's filename will be constituted by parameters' value. If there is same filename, the code will not create new file. If you want to create a new one, you have to remove previous file from the specific [reproducibility output](reproducibility_output/).


## From MATLAB to Python
ongoing...
