# A general lightweight global modeling framework for three-dimensional seismic exploration
## Introduction
Code implementation of LMGM and its training and testing process.

This repository was completed on Sept. 14, 2026 by Changxin Wei, 
a PhD candidate at the College of Instrumentation and Electrical Engineering, Jilin University，
Changchun, China.

If there are any problems with this project, please feel free to contact cxwei24@mails.jlu.edu.cn, 
or you may ask ChatGPT for help.

**Attention**

**The training datasets are not provided here. You may visit http://wiki.seg.org for open datasets
that appear in our paper，or you can get a copy of processed '.mat' files by contacting cxwei24@mails.jlu.edu.cn.**

All the provided seismic datasets are saved in '.mat' format for simplicity. If you intend to use your 
own datasets, please make sure the filename is consistent with the variable name inside the '.mat' file.

All training data should be saved in **'./data/name_for_your_data/train/'**.
All test data should be saved in **'./data/name_for_your_data/test/'**.

---

## File Introduction
```
DMINet_project
│
├── README.md                       # read this first
├── requirements.txt                # used for package installation
├── __init__.py                                 
│
├── src/
│   ├── train.py                    # training program (shared by interpolation and denoising)
│   ├── interplation_test.py        # testing program for interpolation
│   ├── denoising_test.py           # testing program for denoising
│   ├── dataset_processing.py       # dataset processing
│   ├── utils.py                    # utilities
│   ├── __init__.py         
│   └── networks/
│       ├── ANN.py                  # main code of ANN (Competitive method)
│       ├── FAT.py                  # main code of FAT (Competitive method)
│       ├── LMGM.py                 # main code of LMGM
│       ├── PatchEmbedUnembed_t.py  # utilities
│       └── __init__.py
│
├── shell_scripts/                  # shell scripts
│   ├── dataset_processing.sh       # process the original data into patches
│   ├── train.sh                    # run training     
│   └── test/
│       ├── interpolation.sh        # run testing of interpolation
│       └── denoising.sh            # run testing of denoising
│  
├── data/                           # original data, before patching (saved in '.mat' format)
│   ├── 45shots                     # SEG C3 45 shots dataset 
│   │   ├── train                   # data for training
│   │   │   ├── shot1.mat           
│   │   │   ├── shot2.mat           
│   │   │   └── shot3.mat           
│   │   └── test                    # data for testing
│   │       └── shot25.mat                 
│   │           
│   ├── parihaka                    # field post-stack data 
│   │   ├── train                   
│   │   │   └── parihaka_train.mat           
│   │   └── test                    
│   │       └── parihaka_test.mat         
│   │          
│   └── towed                       # field marine data (confidential and not provided)
│       ├── train                   
│       │   ├── towed1.mat           
│       │   ├── towed2.mat         
│       │   ├── ...          
│       │   └── towed30.mat           
│       └── test                    
│           └── towed40.mat             
│
├── dataset/                        # prepared dataset used for training
│   ├── 45shots_32328_6k_TPNN_interpolation.h5   
│   │   ├── train                   # architecture within the ".h5" file
│   │   │   ├── label
│   │   │   │   ├── patch1
│   │   │   │   ├── patch2
│   │   │   │   ├── ...
│   │   │   │   └── patchN
│   │   │   │   
│   │   │   └── feature
│   │   │       ├── patch1          # correspond to the patches in train/label
│   │   │       ├── patch2
│   │   │       ├── ...
│   │   │       └── patchN
│   │   │
│   │   └── val
│   │       └── ...                 # the same architecture with train     
│   └── ...                            
│
├── trained_models/                 # trained models will be saved here
│   └── 3d/                         # an example model is provided here
│       ├── LMGM_45shots_TPNN_interpolation_example/         
│       │   └── recon_best.pth       
│       └── ...                   
│  
├── logs/           
│   └── ....log                     # used to record the training process
│  
├── results/                        # results will be saved here after testing
│   └── 3d/
│       ├── LMGM/  
│       │   ├── 45shots/
│       │   │   ├── shot25.mat
│       │   │   └── metrics_45shots_interpolation.log
│       │   └── .../
│       └── .../       
│       
└── matlab_scripts/                 # matlab scripts for figure generation
    ├── display_3d_45shots.m        # display a 3D view of SEG C3 45shots dataset 
    ├── display_2d_45shots.m        # display a 2D view of SEG C3 45shots dataset 
    ├── display_fk_45shots.m        # display f-k spectrum of SEG C3 45shots dataset 
    ├── drr_plot3d.m                # 3D figure function
    ├── fk.m                        # f-k figure function 
    ├── cseis.m                     # color map function for seismic data 
    └── blue_white_red.mat          # color map for seismic data 
```
---

## How to run training and testing
You need to follow these instructions step-by-step to train and test LMGM successfully.
### 1. Environment
A step-by-step tutorial for environment establishment.
#### 1.1 Installment of Anaconda and Python 
It is recommended to install Anaconda, which comes with Python pre-installed.

**Linux-OS** is recommended (not recommended on Windows).

① Here is an example for installing Python using Anaconda.
```shell
# 1. Download Anaconda in any directory (Specific Version
# can be found in the official website: https://www.anaconda.com/)
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 2. Setup Anaconda by
bash ./Anaconda3-2024.06-1-Linux-x86_64.sh

# 3. Add Anaconda3's bin path to environment variable
# (You should replace the path in the following command)
echo 'export PATH="/path/to/your/anaconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4. Test your installation of Anaconda.
conda --version

# 5. Python3 and Pip3 are integrated in Anaconda, test them.
python --version
pip --version
```
It is recommended to create a new environment as below
(not recommended on the base environment).

② For the first time you use Anaconda, you should:
```shell
conda init
```
otherwise, skip to ③.

③ Reboot your terminal.

④ Create an environment named 'testenv' (or any name you like)
, with python 3.12.3:
```shell
conda create -n testenv python=3.12.3
```
⑤ Activate 'testenv':
```shell
conda activate testenv
```

#### 1.2 Install all required packages
```shell
pip install -r requirements.txt
```

#### 1.3 Install Mamba
① Check the torch version by:
```shell
pip list
```
This output indicates the version of torch is 2.10.x.
```shell
...
torch                    2.10.0
...
```
② Check the CUDA version by:
```shell
nvcc --version
```

This output indicates the version of CUDA is 12.x.
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

③ Check the Python version by:
```shell
python --version
```

This output indicates the version of Python is 3.12.x.
```shell
Python 3.12.3
```

④ Visit https://github.com/state-spaces/mamba/releases, \
and find a suitable version of mamba_ssm, and copy its downloading link.
```shell
https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
**Here, 'cu12' means CUDA 12.x, 'torch2.10' means torch 2.10.x, and 'cp312-cp312' means Python 3.12.x.**

**Besides, the final part of the release name should be ‘...-linux_x86_64.whl', 
since you're using an x86-64 Linux OS.**

⑤ Paste the link after command **[pip install]**, and run this command like this:
```shell
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
If the installation process went successfully, you may continue to **Step 2**;\
Otherwise, you should re-check **the versions of torch and CUDA**, and re-select the release
of mamba_ssm very carefully. 

PS: If your versions of torch and CUDA are not shown in the release list of **mamba_ssm-2.3.1**, maybe a 
suitable release can be found in an older version of mamba_ssm. 

-------

### 2. Dataset preparation
To obtain training and validation datasets, you should run the following command
to process the original seismic data into data patches and save them as
[ground truth]-[corrupted data] pairs into '.h5' file.

Run the shell script as follows:
``` shell
bash ./shell_scripts/dataset_processing.sh [task] [dataset] [norm]
```
where 

`[task]: "interpolation" or "denoising"`

`[dataset]: "45shots", "parihaka", or "towed"`

`[norm]: "TPNN", "absmax", or "none"`

This means creating a dataset for **[task]** using the **[dataset]** dataset 
and **[norm]** normalization. After finishing this script, 
datasets of shape 32×32×8 and 32×32×32 will be created at the same time，
named by `[dataset]_32328_6k_[norm]_[task].h5` 
and `[dataset]_32_6k_[norm]_[task].h5`, respectively.

For example,
``` shell
bash ./shell_scripts/dataset_processing.sh interpolation 45shots TPNN
```

This output indicates a successful processing:
```shell
Training set: 6000 patches
Validation set: 1500 patches
...

torch.Size([1, 1, 32, 32, 8])
torch.Size([1, 1, 32, 32, 8])
...

Training set: 6000 patches
Validation set: 1500 patches
...

torch.Size([1, 1, 32, 32, 32])
torch.Size([1, 1, 32, 32, 32])

```

-------
### 3. Training
After finishing **Step 1** and **Step 2**, you may start training.

Training command:
```shell
bash ./shell_scripts/train.sh [net] [path to prepared dataset]
```
where

`[net]: "ANN", "FAT" or "LMGM"`

`[path to prepared dataset]: modify this to the filename 
(without the file extension) of your prepared dataset`

For example,
``` shell
bash ./shell_scripts/train.sh LMGM 45shots_32328_6k_TPNN_interpolation
```

The trained models will be saved to `'./trained_models/3d/'`.

PS: The parameters in these shell scripts are set for one-GPU training. 
You can modify them for multi-GPU training.

-------

### 4. Testing
After finishing training, test them using:
```shell
bash ./shell_scripts/test/interpolation.sh [net] [dataset] [norm] [foldername of trained model]
bash ./shell_scripts/test/denoising.sh [net] [dataset] [norm] [foldername of trained model]
```
for interpolation or denoising,
where 

`[net]: "ANN", "FAT" or "LMGM"`

`[dataset]: "45shots", "parihaka", or "towed"`

`[norm]: "TPNN", "absmax", or "none"`

`[foldername of trained model]: modify this to the foldername 
of your trained model`

Here, we provide example models (trained for 20 epochs) of the trained LMGM model, saved in './trained_models'.

Run the following command to use it:
```shell
bash shell_scripts/test/interpolation.sh LMGM 45shots TPNN LMGM_45shots_TPNN_interpolation_example
```
the interpolated results will be saved to './results/3d/LMGM/' in '.mat' format.
There will be four variables in one '.mat'-format result:

① **complete**:         complete data or clean data\
② **missing**:          incomplete data or noisy data\
③ **reconstructed**:    interpolated result or denoised result\
④ **difference**:       obtained by subtracting **missing** from **complete**

For the denoising task, the predicted noise can be obtained by 
subtracting **missing** from **reconstructed**

-------

### 5. Figure Generation
We use Matlab R2023a to generatre figures.
The matlab scripts are provided in `./matlab_scripts/`.

The three '.m' scripts that start with 'display_' are mainly
used to generate the figures of the SEG C3 45 shots dataset in our paper, corresponding to 
the 3D views, 2D views, and f-k spectra, respectively.

To generate figures, you should first load the interpolated/denoised results
to the 'Workspace'. The four variables introduced in the last subsection
should be available in the 'Workspace'.

Then, each displaying script begins with the following content:
```
data = complete;

% data = missing;

% data = reconstructed;

% data = difference;
```

Uncomment one of the statements, 
comment out the others, 
and then run the entire displaying script 
to generate the corresponding figure.

-------
## Shell comments

Here, we provide part of the shell comments that appears in the experiments in our paper.

### Section 3.3 Synthetic Example
```shell
    # dataset generation
    bash shell_scripts/dataset_processing.sh interpolation 45shots TPNN
    # training of the three networks
    bash shell_scripts/train.sh LMGM 45shots_32328_6k_TPNN_interpolation
    bash shell_scripts/train.sh FAT 45shots_32328_6k_TPNN_interpolation
    bash shell_scripts/train.sh ANN 45shots_32_6k_TPNN_interpolation
    # testing of the three networks
    bash shell_scripts/test/interpolation.sh LMGM 45shots TPNN [folder name to your trained LMGM model]
    bash shell_scripts/test/interpolation.sh LMGM 45shots TPNN [folder name to your trained LMGM model]
    bash shell_scripts/test/interpolation.sh ANN 45shots TPNN [folder name to your trained LMGM model]
```