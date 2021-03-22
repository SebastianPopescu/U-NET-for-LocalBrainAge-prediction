>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 





# U-NET-for-LocalBrainAge-prediction
Code for upcoming paper "A U-NET model for Local Brain-Age"

How to use:

As mentioned in the paper, the brain scans have to go through the Dartel pipeline in spm12. 

in "spm_brain_age_preprocess_b23d.m" you need to change "/data/my_programs/spm12" path to suit the location of where your local spm12 is installed.

in your local spm12, you need to copy the files situated in the templates folder (Template_{1,2,3,4,5,6}.nii) to "$your_spm12_folder/templates/".


Using LocalBrainAge_testing.py you can obtain new 3D heatmaps of local brain-age for new subjects. The file has the following arguments that you need to specify:

--filepath_csv = location of .csv file that contains the dataset metadata, look inside the .py file to find details regarding formating
--dirpath_gm = location of directory where the gray matter SPM12 segmentations are stored
--dirpath_wm = location of directory where the white matter SPM12 segmentations are stored
--dataset_name = name of your dataset, it just impacts the name of the location where all the subsequent .nii files get saved

in LocalBrainAge_testing.py the format of the .csv file containing meta-data has to have the following column names "Age", "Gender", "Subject", alternatively modify the code at lines 62-64.


Software prerequisite:
-- nibabel
-- tensorflow 1.15.0
-- spm12
