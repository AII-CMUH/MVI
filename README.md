# Predicting Microvascular Invasion (MVI) in Hepatocellular Carcinoma (HCC) Using Deep Learning
HCC is a common liver cancer existing globally, an HCC with MVI often recurs within 2 years. Estimating MVI preoperatively benefits surgical outcomes of HCC. This code leveraged one of the most common convolutional neural network (CNN), [ResNet](https://arxiv.org/abs/1512.03385), to predict the presence of MVI, but the input of the network was not only CT images but clinical factors including age, gender, maximum tumor diameter (MTD), alpha-fetoprotein (AFP), Child-Pugh score, hepatitis B/C surface antigen (HBsAg and HCsAg). Introducing clinical factors boosted the accuracy of MVI prediction.

## Cite this article
Liu, SC., Lai, J., Huang, JY. et al. Predicting microvascular invasion in hepatocellular carcinoma: a deep learning model validated across hospitals. Cancer Imaging 21, 56 (2021). https://doi.org/10.1186/s40644-021-00425-3

## Prerequisites
Python 3.5 (or above) with the following packages:
* pytorch 1.5 or above [(installation guide)](https://pytorch.org/get-started/locally/)
* scikit-learn
* numpy
* pandas
* Pillow
## Dataset Preparation
### Images
For both training and evaluation, the input images must be arranged in the following directory structure:
```
.
├── train
│   ├── 0
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── 1
│       ├── 1001.png
│       ├── 1002.png
│       └── ...
└── val
    ├── 0
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── 1
        ├── 101.png
        ├── 102.png
        └── ...
```
where the files under `train` folder are used for model training, `val` for validation. The folder name `0` and `1` represent the ground truth: MVI negative and positive repectively. As to image file, it can be any arbitrary name. The directory structure under `train` or `val` is exactly what [PyTorch ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder) requires. A concrete example can be found in [example/img_sample](example/img_sample) of this repository.
### Clinical factors
Our modified CNN model accepts multiple inputs including image data and clinical factors all at the same time. The clinical factors are actually numerical and categorical data, in the form as follows:
* **filename** - the name of image file that the clinical factors (CFs) belong to
* **pid** - the integer number used for telling which images were collected from the same patient
* **age** - the 1st CF, normalized value of the age of patient
* **gender** - the 2nd CF, categorical value 0 and 1 represent female and male respectively
* **tumorSize** - the 3rd CF, normalized value of MTD
* **child_a** - the 4th CF, categorical value (0 or 1), the presence of Child–Pugh class A
* **child_b** - the 5th CF, categorical value (0 or 1), the presence of Child–Pugh class B
* **child_c** - the 6th CF, categorical value (0 or 1), the presence of Child–Pugh class C
* **hbv** - the 7th CF, categorical value (0 or 1), the presence of HBsAg
* **hcv** - the 8th CF, categorical value (0 or 1), the presence of HCsAg
* **afp** - the 9th CF, normalized value of AFP

All items above must be saved in CSV format. A concrete example can be found in [example/cf_sample.csv](example/cf_sample.csv) of this repository.
## Training
The training process can be run using the following command:
```
python3 train.py {image_folder_path} {cf_csv_path} {log_dir_path}
```
where `{image_folder_path}` points to the folder that contains input images, and `{cf_csv_path}` points to the CSV file filled with the clinical factors, the details of both arguments are described in [Dataset Preparation](#dataset-preparation) section. `{log_dir_path}` points to the folder that contains model checkpoint file and [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard) event files, which all will be generated during the training automatically. It uses ImageNet pre-trained model by default.

The following example demonstrates starting a training:
```
python3 train.py example/img_sample example/cf_sample.csv log
```
## Evaluation
The evaluation process can be run using the following command:
```
python3 eval.py {image_folder_path} {cf_csv_path} {log_dir_path} {checkpoint_path}
```
The arguments are similiar to `train.py`'s, except there is one more argument `{checkpoint_path}` that points to the checkpoint file of a trained model.

## License
[GPLv3](https://raw.githubusercontent.com/AII-CMUH/MVI/main/LICENSE)
