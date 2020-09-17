# Usage

## Step 1: select OpenML datasets

```
python select_datasets.py
```
This selects OpenML datasets, output indices into a csv file and selection criteria into a txt file. The conditions for selecting datasets as well as filename of outputs can be modified. 

To print out OpenML dataset indices together with dataset name and size information, do
```
python select_datasets.py details
```

## Step 2: Generate preprocessed OpenML datasets
```
./generate_preprocessed_dataset.sh
```
This generates preprocessed OpenML datasets, whose indices come from `selected_dataset_indices.csv`. Output directory should be specified in `generate_preprocessed_dataset.py`.


## Optional: print indices of selected OpenML datasets
```
python process_csv.py <filename>.csv
```

This outputs dataset indices with space segments, making it easy to fit in a bash for-loop. 
