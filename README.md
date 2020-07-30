# Master Thesis
##### by Tomáš Karella
## [DOCUMENTATION](https://karellat.github.io/master_thesis)
This repository contains: 
#### Directories
* configs - example of configs for model training
    * [docs](./configs/README.md)
* deep_mri - python package for simple model training and preprocessing of dataset
    * [docs](./docs/deep_mri/index.html)
* docs - documentation of the tools library
* examples - examples MRI images in nifty format  
    * [docs](./examples/README.md)
* ext - external libraries for preprocessing Dockerfile
* notes - some notes of adni dataset and preprocessing tools
* scripts - examples of preprocessing scripts and variable training scripts
    * [docs](./scripts/README.md)
#### Files 
* conda.yml - conda enviroment containing all the required packages
* Dockerfile - docker image created by  [NeuroDocker](https://github.com/ReproNim/neurodocker)
* Makefile - automatized basic operation 
* run_train.py - python script running experiment on given config

# Usage
1. Obtain the MRI dataset
2. Perform the preprocessing
3. Train the model on processed images

## Requirements
* install the Anaconda software
    * We recommend the [MiniConda version](https://docs.conda.io/en/latest/miniconda.html)
* create conda environment from **conda.yml** file 
    * ``conda env create -f conda.yml``
* (optional) Due to preprocessing the Docker tool is needed. 
    * It is possible to compile the tools, that depends on HW, but it is far more comfortable to use our **DockerImage** than compile all the preprocessing tools 
    * Install [Docker Tool](https://docs.docker.com/get-docker)) 
    * Basics of Docker can be found at [Docker tutorial](https://docs.docker.com/get-started)
    * Build the our Docker Image and tag it with *DOCKER_TAG*
        * move to repository directory 
        * ``docker build --tag DOCKER_TAG .``
        
# Dataset
open dataset [**ADNI**](http://adni.loni.usc.edu/)

''The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a longitudinal multicenter study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimers disease (AD).'' 
#### How access **ADNI** 
enrol at https://ida.loni.usc.edu/login.jsp?project=ADNI
#### Image Collection
Standardized ADNI1:Complete 1Yr 1.5T 
## How to Download Dataset
For authorized ADNI users only:
Standardized Image Collections:
1. Log into the archive: https://ida.loni.usc.edu/login.jsp?project=ADNI
2. Under the DOWNLOAD menu, choose **Image Collections**
3. In the left navigation section, click **Other Shared Collections**
4. Select **ADNI**
5. Click on the collection name matching the desired standardized data set (ADNI1:Complete 1Yr 1.5T)
6. Download all the images and **CSV** with basic information about subjects(disease group)
7. Download to an appropriately named location on your computer system 

(original tutorial at http://adni.loni.usc.edu/methods/mri-tool/mri-analysis/#mri-data-set-container)

## ADNI - basic info and glossary
### INLUSIoN Criteria 
- age 55-90
- at least 6 grades education or work history
- fluently English/Spanish

### Normal controls (NL)
- no memory problems or complaints

### Mild Congnitive Impairment (MCI) 
- memory problems
- Clinical Dementia Rating 0.5

### Alzheimer's Disease (AD) 
- memory problems
- Clinical Dementia Rating 0.5 - 1
- Criteria of AD


# Preprocessing
* To create preprocessing workflow we use the [Nipype project](https://nipype.readthedocs.io/en/latest/)
    * Nipype allow us to connect a preprocessing tools and create a computational graph.
    * Nipype supports the parallel execution
    * To learn more about creating workflows in Nipype see the awesome [tutorial](https://nipype.readthedocs.io/en/latest)
    
* We extended Nipype nodes by some of our own, they can be found in submodule preproces of deep_mri as submodule 
* Use our predefined FSL or MINC toolkit pipelines 
    * [FSL pipeline](./scripts/preprocess/fsl_pipeline.py)
    * [MINC pipeline](./scripts/preprocess/minc_pipeline.py)   
* We also have a workflow for entropy slicing of 3D images
    * [Slice pipeline](./scripts/preprocess/slicer_script.py)
### Running
* Create workflow or use one of ours 
* Extract dataset to single directory preserving original structure
* Run Docker container based on previously created image 
```
docker run 
       -it 
       -v <PATH_TO_SOURCE_CODES>:/home/neuro/thesis 
       -v <PATH_TO_DATA_ROOT>:/ADNI 
       DOCKER_TAG /bin/bash
```
* Setup the constants of the script
* Run the chosen script

# Training
* the Conda environment is required for using **deep_mri** module
### Run from config
* run_train.py script can be used for training defined by config file
    * [Read more about configs](./configs/README.md)
    * run_train.py arguments
         * --help
            * show help
         * -c, --config_file PATH 
            * path to config file
         * -i, --int_var <TEXT INTEGER>...
            * override the integer variable of config <NAME VALUE> 
         * -s, --str_var <TEXT TEXT>...
            * override the string variable of config <NAME VALUE> 
         * -f, --float_var <TEXT FLOAT>...
            * override the float variable of config <NAME VALUE> 
## Specific train cycle or non-standard dataset
* for testing model or dataset
* the functions can be used directly from the deep_mri module
    * [our scripts and examples](./scripts)
* or only parts of the config can be used for generation of:
    * dataset
    * model 
    * training cycle 
    * see deep_mri [docs](./docs/deep_mri/index.html) 
        * submodule train.config_parser
