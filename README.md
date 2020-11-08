# IEOR 4571 Homework 2

This repository contains library code and experimental of Homework 2 of IEOR 4571 course in Columbia University

----

## Getting Started

To get start the project, you need to set up an environment first. `requirents.txt` file contains the required packages for the environment. In addition, we also require `JDK8/11` and `Python=3.8.5`

For simplicity, we can set up a python virtual environment with `conda` with following lines:

```bash
conda create --name personalization python=3.8.5
conda activate personalization
pip install -r requirments.txt
python -m ipykernel install --user --name personalization
conda deactivate
```

Then, you will see a named `personalization` kernel in your jupyter notebook environment.

----

## Usage

### Download

You can edit the parameter `url` and `fp` in the file `config/downloads.json` to download different dataset from Movielens. By default, we will download `ml-latest.zip` to `/downloads` directory. To proceed download action, run following line:

``` bash
python main.py download
```

### Sample

You can edit the parameter for sample in the file `config/sample.json`.

The explanation of parameter as follow:
    - `min_items`: number of minimum items to get from sample.
    - `min_users`: number of minimum users to get from sample.
    - `user_threshold`: a user should at least rate how many items in sample dataset.
    - `item_threshold`: an item should be at least rated by how many user in sample dataset.
    - `op`: directory to store sample data.
    - `random_seed`: random seed for experimental result.


----

## Description of Content 

----

## Contribution