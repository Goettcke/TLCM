# Tomek Link Complexity Measure

## Requirements
- DS_Pipe which is a package used in some of the experiments, it can be found here [ds-pipe-repo](https://git.imada.sdu.dk/goettcke/DS_Pipe/).There is a compiled version in the repository under `./dependencies/DS-Pipe-[version]-py3-none-any.whl`. 
This can be installed by running `pip install dependencies/DS_Pipe...whl`.  

## Datasets 
All datasets used in this project can be found in the `src/datasets/` folder. 
Loading datasets can be done in the following way: 

```python
from src.utils import load_synthetic_datasets, load_real_datasets
from src.measures import tlcm
all_datasets = load_synthetic_datasets() + load_real_datasets()
for dataset,dataset_name in all_datasets: 
    print(f"{dataset_name}: {tlcm(dataset)}")
```

Here all of the dataset names are returned in console - line by line, and the corresponding TLCM measure is returned.

In the case, where the users data does not follow this format, we provide the following generate_bunch_dataset functionality which also comes from the DS\_Pipe library.  In the following example we are generating a dataset containing 3 instances in a 2 dimensional space, with labels either 0 or 1. 
After the dataset is generated all complexity measures can be applied.

```python
from ds_pipe.datasets.dataset_utils import generate_bunch_dataset 
X = [[1,2],[3,4],[1,3]]
y = [0,1,0]
dataset = generate_bunch_dataset(X,y)
```

## Imbalance Complexity Measures
Following is an example on how to compute the complexity measures on a dataset represented in the Scikit-Learn style containing both dataset.data and dataset.target. Here the dataset is a real dataset loaded from the collection in the DS\_Pipe package.
```python
from ds_pipe.datasets.dataset_loader import DatasetCollections 
from src.measures import tlcm, degIR, degOver, imbalance_ratio, n_1_imb_mean, n_3_imb_mean

dc = DatasetCollections()
dataset, dataset_name = dc.load_dataset("glass")
print(f"TLCM: {tlcm(dataset)},\nIR: {imbalance_ratio(dataset)},\nN1:{n_1_imb_mean(dataset)},\nN3: {n_3_imb_mean(dataset)}")
```
The list of imbalance complexity measures in this package are: 
- N1 (Barella et al. 2021)
- N3 (Barella et al. 2021)
- TLCM (Goettcke et al. 2022)
- degIR (Mercier et al. 2019)
- degOver (Mercier et al. 2019)
- LRID (Zhu et al. 2018) 
- IR

More imbalance complexity measures can be found in the DS\_Pipe package under `from ds_pipe.datasets.dataset_metadata import measure`

## Guide to the experiment  
The experiments are split up into several files. Many of the operations which are done in preparation for the experimental pipeline are in the jupyter notebook `binary_experiment.ipynb`. Here you find code to generate the synthetic datasets, loading the results into dataframes, which are used for the experimental analysis, building the plots, computing correlation and much more.
This file is built on top of the `src.utils` file, which contains functions to: 
- get_minority_class
- get_majority_class
- undersampler, which achieves a desired *IR* by either undersampling the majority class, or the minority class. 
- dataset_loading methods, to quickly get the required datasets
- base_skf_test, which is the stratified k-fold cross-validation setup, which was used in the hyperparamter optimization, and which provide the performance evaluation for each of the classifiers. 
-  A function to write the results to files
All of the parameter spaces are defined in the `src.parameter_spaces.py` file. 
To compute the individual classifier results, there are associated experiments in the root folder, these are: 
- `svm_experiment.py` (support vector machines)
- `mlp_experiment.py` (multi-layer perceptron)
- `rf_experiment.py` (random forest)
- `tree_experiment` (decision trees)
- `knn_experiment`(k-nearest neighbors)
- `nb_experiment`(Gaussian Naive Bayes)

In each of these files, in the bottom of the script, it is possible to toggle if you want to run unoptimized experiments or optimized experiments. The default is to run the optimized experiments, and for the curios readers - this gives a great difference to the classifiers performances. Notice that the hyper parameter optimization is set to optimize the macro averaged recall. 
```python 
if __name__=="__main__": 
    datasets = load_synthetic_datasets() + load_real_datasets()
    optimized_experiment(datasets,write_result_flag=True)
    #unoptimized_experiment()
``` 


##  People 
- Jonatan Gøttcke
- Colin Bellinger 
- Paula Branco
- Arthur Zimek

