# Introduction 

For more information about nnU-Net, please read the following paper:

`Isensee, Fabian, et al. "nnU-Net: Breaking the Spell on Successful Medical Image Segmentation." arXiv preprint arXiv:1904.08128 (2019).`

This repository is still work in progress. Things may break. If that is the case, please let us know.

# Installation 
nnU-Net has been tested on Windows10 with CUDA 10.1 AND Pytorch 1.3.1, and edited by VSCODE.

Installation instructions
1) Install PyTorch (https://pytorch.org/get-started/locally/)
2) Clone this repository `git clone https://github.com/TianyiLu/nnUNetLearnOnWindows.git`
3) Go into the repository (`cd nnUNet`)
4) Install with `pip install -r requirements.txt` followed by `pip install -e .`
5) There is an issue with numpy and threading in python which results in too high CPU usage when using multiprocessing. 
We strongly recommend you set OMP_NUM_THREADS=1 in your bashrc OR start all python processes 
with `OMP_NUM_THREADS=1 python ...`

# Getting Started 
All the commands in this section assume that you are in a terminal and your working directory is the nnU-Net folder 
(the one that has all the subfolders like `dataset_conversion`, `evaluation`, ...)

## Set paths 
nnU-Net needs to know where you will store raw data, want it to store preprocessed data and trained models. Have a 
look at the file `paths.py` and adapt it to your system.

## Preparing Datasets 
nnU-Net was initially developed as our participation to the Medical Segmentation Decathlon <sup>1</sup>. It therefore
 relies on the dataset to be in the same format as this challenge uses. Please refer to the readme.md in the 
 `dataset_conversion` subfolder for detailed information. Examples are also provided there. You will need to 
 convert your dataset into this format before you can continue.
 
Place your dataset either in the `raw_dataset_dir` or `splitted_4d_output_dir`, as specified in `paths.py` (depending on how you prepared it, again 
see the readme in `dataset_conversion`). Give 
it a name like: `TaskXX_MY_DATASET` (where XX is some number) to be consistent with the naming scheme of the Medical 
Segmentation Decathlon.

## Experiment Planning and Preprocessing 
This is where the magic happens. nnU-Net can now analyze your dataset and determine how to train its 
U-Net models. To run experiment planning and preprocessing for your dataset, execute the following command:

`python experiment_planning/plan_and_preprocess_task.py -t TaskXX_MY_DATASET -pl Y -pf Z`

here `TaskXX_MY_DATASET` specifies the task (your dataset) and `-pl`/`-pf` determines how many processes will be used for 
datatset analysis and preprocessing (see `python experiment_planning/plan_and_preprocess_task.py -h` for more details). Generally you want this number to be as high as you have CPU cores, unless you 
run into memory problems (beware of datasets such as LiTS!)

Running this command will to several things:
1) If you stored your data as 4D nifti the data will be split into a sequence of 3d niftis. Back when I started 
SimpleITK did not support 4D niftis. This was simply done out of necessity.
2) nnU-Net configures the U-Net architectures based on that information. All U-Nets are configured to optimally use 
**12GB Nvidia TitanX** GPUs. There is currently no way of adapting to smaller or larger GPUs.
3) nnU-Net runs the preprocessing and saves the preprocessed data in `preprocessing_output_dir`.

I strongly recommend you set `preprocessing_output_dir` on a SSD. HDDs are typically too slow for data loading. 

## Training Models 
There is an issue with numpy and threading in python which results in too high CPU usage when using multiprocessing. 
We strongly recommend you set OMP_NUM_THREADS=1 in your bashrc OR start all python processes 
with `OMP_NUM_THREADS=1 python ...`

The following pipeline describes what we ran for all the challenge submissions. If you are not interested in getting 
every last bit of performance we recommend you also look at the Recommendations section.

nnU-Net uses three different U-Net models and can automatically choose which (of what ensemble) of them to use. The 
default setting is to train each of these models in a five-fold cross-validation.

Trained models are stored in `network_training_output_dir` (specified in `paths.py`).

### 2D U-Net (Not tested) 
For `FOLD` in [0, 4], run:

`python run/run_training.py 2d nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

### 3D U-Net (full resolution, tested and updated. Now our output is based on this model.)
For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_fullres nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

### 3D U-Net Cascade 
The 3D U-Net cascade only applies to datasets where the patch size possible in the 'fullres' setting is too small 
relative to the size of the image data. If the cascade was configured you can run it as follows, otherwise this step 
can be skipped.

For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_lowres nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

After validation these models will automatically also predict the segmentations for the next stage of the cascade and 
save them in the correct spacing.

Then run
For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_cascade_fullres nnUNetTrainerCascadeFullRes TaskXX_MY_DATASET FOLD --ndet`

## Ensembling 
Once everything that needs to be trained has been trained nnU-Net can ensemble the cross-validation results to figure 
out what the best combination of models is:

`python evaluation/model_selection/figure_out_what_to_submit.py -t XX`

where `XX` is the taskID you set for your dataset. This will generate as csv file in `network_training_output_dir` 
with the results.

You can also give a list of task ids to summarize several datastes at once.

## Inference 
You can use trained models to predict test data. In order to be able to do so the test data must be provided in the 
same format as the training data. Specifically, the data must be splitted in 3D niftis, so if you have more than one 
modality the files must be named like this (same format as nnUNet_raw_splitted! see readme in dataset_conversion folder):

```
CaseIdentifier1_0000.nii.gz, CaseIdentifier1_0001.nii.gz, ...
CaseIdentifier2_0000.nii.gz, CaseIdentifier2_0001.nii.gz, ...
...
```

To run inference for 3D U-Net model, use the following script:

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t TaskXX_MY_DATASET -tr nnUNetTrainer -m 3d_fullres`

If you wish to use the 2D U-Nets, you can set `-m 2d` instead of `3d_fullres`.

To run inference with the cascade, run the following two commands:

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER_LOWRES -t TaskXX_MY_DATASET -tr nnUNetTrainer -m 3d_lowres`

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER_CASCADE -t TaskXX_MY_DATASET -tr 
nnUNetTrainerCascadeFullRes -m 3d_cascade_fullres -l OUTPUT_FOLDER_LOWRES`

here we first predict the low resolution segmentations and then use them for the second stage of the cascade.

There are a lot more flags you can set for inference. Please consult the help of predict_simple.py for more information.

### Ensembling test cases 
Per default nnU-Net uses the five models obtained from cross-validation as an ensemble. How to ensemble different U-Net Models 
(for example 2D and 3D U-Net) is explained below.

If you wish to ensemble test cases, run all inference commands with the `-z` argument. This will tell nnU-Net to save the 
softmax probabilities as well. They are needed for ensembling.

You can then ensemble the predictions of two output folders with the following command:

`python inference/ensemble_predictions.py -f FOLDER1 FODLER2 ... -o OUTPUT_FOLDER`

This will ensemble the predictions located in `FODLER1, FOLDER2, ...` and write them into `OUTPUT_FOLDER`


## Tips and Tricks
The model training pipeline above is for challenge participations. Depending on your task you may not want to train all 
U-Net models and you may also not want to run a cross-validation all the time.

#### Sharing Models
You can share trained models by simply sending the corresponding output folder from `network_training_output_dir` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.

## FAQ
1) ##### Can I run nnU-Net on smaller GPUs?

    You can run nnU-Net in fp16 by specifying `--fp16` as additional option when launching trainings. This will reduce 
    the amount of GPU memory needed to ~9 GB and allow to run everything on 11GB cards as well. You can also manually 
    edit the plans.pkl files (that are located in the subfolders of preprocessed_output_dir) to make nnU-net use less 
    feature maps. This can however have an impact on segmentation performance
