# Output folder
You will obtain confusion matrices which compare the original (achieved by the original MATLAB code) and replicated results.
- In this case study, the code will take general input data from [data](../data/) folder but output will be placed in the specific [reproducibility output](reproducibility_output/) in Jupyter Notebook case. When you launch with Docker, you will create a linked output folder. Original MATLAB results (100 and 1000 permutations of bootstrap) are in [data](../data/) folder.
- Result's filename will be constituted by parameters' value. If there is same filename, the code will not create new file. If you want to create a new one, you have to remove previous file from the specific [reproducibility output](reproducibility_output/).
