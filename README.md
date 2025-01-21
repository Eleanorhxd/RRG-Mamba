# RRG-Mamba
**This is the data and code for our paper** `RRG-Mamba: Efficient Radiology
Report Generation with State Space Model`.

For reproduction of medication prediction results in our paper, see instructions below.

## Overview
We have modularized and encapsulated the code into a more readable form. In brief, RRG-Mamba consists of three parts: Visual Extractor, Global Dependency Learning Module, and Radiology Report Generator
## Prerequisites

Make sure your local environment has the following installed:

* `pytorch>=1.12.1 & <=1.9`
* `numpy == 1.15.1`
* `python >= 3.10`
* `scikit-learn >= 0.24.2`
* `torchvision == 0.8.2`
* `causal-conv1d == 1.1.1`

## Datastes

We use two publicly available radiology report generation datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).

The training set, test set, and validation set data of the IU X-Ray and MIMIC-CXR datasets are shown in the following table:
| Dataset   |           | IU X-Ray    |           |           | MIMIC-CXR  |           |
|-----------|:---------:|:-----------:|:---------:|:---------:|:----------:|:---------:|
| Dataset   |  TRAIN    |     VAL     |   TEST    |   TRAIN   |    VAL     |   TEST    |
| IMAGE#    |  5,226    |     748     |   1,496   |  368,960  |   2,991    |   5,159   |
| REPORT#   |  2,770    |     395     |    790    |  222,758  |   1,808    |   3,269   |
| PATIENT#  |  2,770    |     395     |    790    |  64,586   |    500     |    293    |
| AVG.LEN   |  37.56    |    36.78    |   33.62   |   53      |   53.05    |   66.4    |


After downloading the datasets, put them in the directory `data`.

## Documentation

```
--data
  │--iu_xray
    │--images
    │--annotation.json
  │--mimic_cxr
    │--images
    │--annotation.json
  
--models
  │--model.py

--modules
  │--utils.py
  │--visual_extractor.py
  │--dataset.py
  │--dataloaders.py
    .......
  
--src
  │--README.md
  │--train.py
  │--run_iu.sh
  │--run_cxr.sh  
```

## How to MambaGen

### 1 Install IDE 

Our project is built on PyCharm Community Edition ([click here to get](https://www.jetbrains.com/products/compare/?product=pycharm-ce&product=pycharm)).

### 2 Environment setting
#### 2.1 Inpterpreter 
We recommend using `Python 3.10` or higher as the script interpreter. [Click here to get](https://www.python.org/downloads/release/python-3110/) `Python 3.10`. 
#### 2.2 Packages
Please follow the packages in [Prerequisites](#prerequisites), utilize `pip install <package_name>` to construct the environment.
### 3 Train
Run `bash run_iu.sh` to train a model on the IU X-Ray dataset.

Run `bash run_cxr.sh` to train a model on the MIMIC-CXR dataset.


Our experiments were done on NVIDIA 4090 card.

## Acknowledgement
We sincerely thank - [R2Gen](https://github.com/cuhksz-nlp/R2Gen).

## TODO

To make the experiments more efficient, we developed some experimental scripts, which will be released along with the paper later.
