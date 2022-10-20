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




## Imbalance Complexity Measures
Following is an example on how to compute the complexity measures on a dataset represented in the Scikit-Learn style containing both dataset.data and dataset.target. Here the dataset is a real dataset loaded from the collection in the DS\_Pipe package.
```python
from ds_pipe.datasets.dataset_loader import DatasetCollections 
from src.measures import tlcm, degIR, degOver, imbalance_ratio, n_1_imb_mean, n_3_imb_mean

dc = DatasetCollections()
dataset, dataset_name = dc.load_dataset("glass")
print(f"TLCM: {tlcm(dataset)},\nIR: {imbalance_ratio(dataset)},\nN1:{n_1_imb_mean(dataset)},\nN3: {n_3_imb_mean(dataset)}")
```


##  People 
- Jonatan Gøttcke
- Colin Bellinger 
- Paula Branco
- Arthur Zimek

