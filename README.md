# Baseline for Projects of Computer Vision with Deep Learning(CSI6072, 2021 Fall)

## Table of Contents

- Summary
    - Course Information
    - The purpose of this repository
    - Environment Setup
- Projects
    - Project 2
        - TODO
    - Project 1
        - Information
        - Dataset
        - Training
        - Inference
        - Submission

## Summary

### Course Information

- Professor: [Seon Joo Kim](https://sites.google.com/site/seonjookim/)
- Computer Vision with Deep Learning, CSI6072, (2021 Fall)
- [Syllabus](ysweb.yonsei.ac.kr:8888/curri120601/curri_pop2.jsp?&hakno=CSI6702&bb=01&sbb=00&domain=A&startyy=2021&hakgi=2&ohak=10421)

### The purpose of this repository

- Provide baseline
    - You can extend the baseline for better solution
- Provide submission csv file generation
    - Your submitted result(CSV file) and model(.pth file) should correspond to each other
- Provide useful script to pre-process data

### Environment Setup

You can set up this part as your own if compatible

```
# Desired Setup
(pytorch19) user@host:/projects/example-code# python --version
Python 3.9.6
(pytorch19) user@host:/projects/example-code# pip freeze | grep torch
torch==1.9.0
torchaudio==0.9.0a0+33b2469
torchvision==0.10.0
```

The recommended way is as following

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run following commands

```
conda create -n pytorch19
conda activate pytorch19
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Projects

### Project 2

TODO

### Project 1

#### Information

- Korean Food Classification
- [Kaggle Leaderboard](https://www.kaggle.com/c/yonsei-csi6702-2021fall-project1/leaderboard)
- [Refined Dataset - Kaggle](https://www.kaggle.com/c/yonsei-csi6702-2021fall-project1/data)


#### Dataset

You can download data from [Data section of Kaggle](https://www.kaggle.com/c/yonsei-csi6702-2021fall-project1/data)

This is the structure of data directory

```
(pytorch19) user@host:/projects/example-code# tree -d ./data
# tree -d data/korean_food_classification_data 
data/korean_food_classification_data
|-- test
|   `-- test
|-- train
|   `-- train
`-- val
    `-- val
```

#### Training

```
PYTHONPATH=$PYTHONPATH:. python tools/project1_train.py
```

#### Inference

```
PYTHONPATH=$PYTHONPATH:. python tools/project1_inference.py
```

#### Submission

1. Train your model with your code
2. Edit and Run `inference_baseline.py` to generate CSV file(`submission.csv`)
3. Edit the name of CSV file(e.g. `2019324058.csv`)
4. Upload CSV file to [Kaggle](https://www.kaggle.com/c/yonsei-csi6702-2021fall-project1/overview)
