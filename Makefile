ADNI_DATA='/d/MRI/ADNI'
CON_NAME="tkarellla_con"

ifeq (, $(shell which nvidia-docker))
	DOCKER=docker
else
	DOCKER=nvidia-docker
endif

build: 
	docker build --tag my_neuro .

# Fixing mounting wsl fsl 
SOURCE_PATH=$(shell pwd | sed 's/^\/mnt//g')

run_docker: 
	$(DOCKER) run \
		-it \
		--rm \
		-p 8889:8889 \
		-v $(SOURCE_PATH):/home/neuro/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name $(CON_NAME) \
		my_neuro jupyter notebook --port 8889
	
connect_bash: 
	$(DOCKER) exec -it "tkarellla_con" /bin/bash

run_docker_bash:  
	$(DOCKER) run \
		-it \
		--rm \
		-v $(SOURCE_PATH):/home/neuro/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name $(CON_NAME) \
		--memory=12G \
		my_neuro /bin/bash 

prepare_dockerfile:
	docker run  --rm kaczmarj/neurodocker:master generate docker \
	--base neurodebian:stretch-non-free \
	--pkg-manager apt \
	--install convert3d gcc g++ graphviz tree \
	         git-annex-standalone vim emacs-nox nano less ncdu \
	         tig git-annex-remote-rclone octave netbase \
	--spm12 version=r7219 \
	--fsl version=6.0.1 \
	--minc version=1.9.15 \
	--ant version=2.3.1 \
	--afni version=latest \
	--user=neuro \
	--workdir /home/neuro \
	--miniconda miniconda_version="4.3.31" \
	    conda_install="python=3.6 pytest jupyter jupyterlab jupyter_contrib_nbextensions \
	    traits pandas matplotlib scikit-learn scikit-image seaborn nbformat nb_conda" \
	    pip_install="https://github.com/nipy/nipype/tarball/master\
	              https://github.com/INCF/pybids/tarball/0.7.1\
	              nilearn datalad[full] nipy duecredit nbval" \
	    create_env="neuro" \
	    activate=True \
	--env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
	--run-bash "source activate neuro && jupyter nbextension enable exercise2/main && jupyter nbextension enable spellchecker/main" \
	--run 'chown -R neuro /home/neuro' \
	--run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
	--workdir /home/neuro \
	--cmd jupyter-notebook > Dockerfile
