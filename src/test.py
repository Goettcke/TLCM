from measures import complexity_adjusted_imbalance_ratio
from ds_pipe.datasets.dataset_metadata import DatasetInformation
from ds_pipe.datasets.dataset_loader import DatasetCollections

dc = DatasetCollections()
dataset,_ = dc.load_dataset("coil-2000")

print(complexity_adjusted_imbalance_ratio(dataset))

print("Done")