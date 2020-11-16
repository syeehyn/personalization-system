# IEOR 4571 Homework 2

This repository contains library code and experimental of Homework 2 of IEOR 4571 course in Columbia University.

[Final Report](./notebook/final_report.ipynb)

----

## Description of Content

``` bash
├── LICENSE
├── README.md
├── config #configuration file for ETL commands
│   ├── downloads.json
│   ├── sample.json
│   └── split.json
├── data #our sample/experimental data
│   ├── interim #splitted datasets
│   ├── processed #results
│   │   ├── evaluation_result #results for model evaluation
│   │   ├── prediction #results with prediction
│   │   └── tuning_result #tuning paramter results
│   └── raw
│       ├── movies.csv #the movie genre reference
│       └── sample.csv #our sample
├── main.py #python script for ETL process
├── notebook
│   ├── als_parameter_tuning.ipynb #notebook of als parameter tuning execution
│   ├── coverage_execution.ipynb #notebook of coverage evaluation execution
│   ├── evaluation_execution.ipynb #notebook of model evaluation (rmse/acc) execution
│   └── final_report.ipynb #notebook of final report
├── requirements.txt #required libiraries
└── src #libary source code
    ├── __init__.py
    ├── baseline #baseline model sorce code
    │   ├── __init__.py
    │   └── baseline.py
    ├── evaluation #evaluator source code
    │   ├── __init__.py
    │   └── evaluation.py
    ├── memory_based #memory-based CF source code
    │   ├── __init__.py
    │   └── memory_based_cf.py
    ├── model_based #model-based CF source code
    │   ├── __init__.py
    │   └── als.py
    ├── sample #sampling source code
    │   ├── __init__.py
    │   └── sample.py
    ├── transformer #transformer/preprocessor source code
    │   ├── __init__.py
    │   └── transformer.py
    └── utils #utility function source code
        ├── __init__.py
        ├── downloads.py
        ├── loads.py
        ├── spark_session.py
        └── train_test_split.py
```

----

## Prerequisite and Getting Started

To get start the project, you need to set up an environment first. `requirents.txt` file contains all the required packages for the environment. In addition, we also require `JDK8/11` and `Python=3.8.5`

For simplicity, you can set up a python virtual environment with `conda` by following lines:

```bash
conda create --name personalization python=3.8.5
conda activate personalization
pip install -r requirments.txt
python -m ipykernel install --user --name personalization
conda deactivate
```

Then, you will see a named `personalization` kernel in your jupyter notebook environment.

----

## ETL and Data Preprocess

ETL and Data Preprocess are done with python file `main.py` to run the full ETL and Data Preprocess pipeline, just simply type following line:

``` bash
python main.py download sample train-test-split
```

Since the process is trivial, we have store our experimental sampled dataset under `/data/` directory.

If you want to try different dataset, you may find an explanation of parameters below.

### Download

You can edit the parameter `url` and `fp` in the file `config/downloads.json` to download different dataset from Movielens. By default, we will download `ml-latest.zip` to `/downloads` directory. To proceed download action, run following line:

``` bash
python main.py download
```

### Sample

You can edit the parameter for sample in the file `config/sample.json`.

The explanation of parameters as follow:

- `min_items`: number of minimum items to get from sample.
- `min_users`: number of minimum users to get from sample.
- `user_threshold`: a user should at least rate how many items in sample dataset.
- `item_threshold`: an item should be at least rated by how many user in sample dataset.
- `op`: directory to store sample data.
- `random_seed`: random seed for experimental result.

In our experimental dataset: we used following parameters:

```json
{
    "min_items": 1000,
    "min_users": 20000,
    "user_threshold": 5,
    "item_threshold": 100,
    "op": "./data/raw",
    "random_seed": 0
}
```

Sampled data is already under directory of the repository: `data/processed/sample.csv`.

To proceed sample action, run following line:

``` bash
python main.py sample
```

### Train-Test-Split

To validate the robustness of our recommendation system, we need to cross validate our data. For splitting the train and test data, you can edit parameter in the file `config/split.json`.

The explanation of parameters as follow:

- `op`: the output directory of splitted data.
- `seed`: the random seed of split in Pyspark.
- `splits`: the list of training set ratio. [0.25] means creating a split for 25% training and 75% test dataset.

In our experimental dataset: we used following parameters:

```json
{
    "op": "./data/interim",
    "seed": 0,
    "splits": [0.25, 0.5, 0.75]
}

```

To proceed train-test-split, run following line:

```bash
python main.py train-test-split
```

----

## Model and Recommender System

### Memory-based Collaborative Filtering

We use `numpy` to build Memory-based Collaborative Filtering brute-forcely.

#### User-based Collaborative Filtering

You may find the usage example and evaluation results in our experimental dataset in `Memory Based Collaborative Filtering` section of our final report [notebook](./notebook/final_report.ipynb#Memory-Based-Collaborative-Filtering)

#### Item-based Collaborative Filtering

You may find the usage example and evaluation results in our experimental dataset in `Memory Based Collaborative Filtering` section of our final report [notebook](./notebook/final_report.ipynb#Memory-Based-Collaborative-Filtering)

### Model-based Collaborative Filtering

Our model-based collaborative filtering takes advantage ALS implementation from Pyspark.

You may find the usage example and evaluation results in our experimental dataset in `Model Based Collaborative Filtering` section of our final report [notebook](./notebook/final_report.ipynb#Model-Based-Collaborative-Filtering)

----

## Contribution

- Hu, Bo (UNI: bh2569)
- Qin, Rui (UNI: rq217)
- Yuan, Shuibenyang (UNI: sy2938)