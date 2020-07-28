ADNI_DATA='/data/tkarella/ADNI/MRI/'
CON_NAME="tkarellla_con"
DOCKER=docker

build: 
	docker build --tag my_neuro .

# Fixing mounting wsl fsl 
SOURCE_PATH=$(shell pwd | sed 's/^\/mnt//g')

generate_doc:
    pdoc --html --output-dir docs --force deep_mri

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
	$(DOCKER) exec -it "tkarella_con" /bin/bash

	
run_tensorflow: 
	$(DOCKER) run \
		--detach \
		-v $(SOURCE_PATH):/tf/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name "tkarella_tf" \
		tensorflow/tensorflow:latest /bin/bash -c "pip install --upgrade tensorflow-hub scikit-learn matplotlib nilearn && cd /tf/thesis && python training3d.py" 


run_encoder: 
	$(DOCKER) run \
		--detach \
		-v $(SOURCE_PATH):/tf/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name "tkarella_encoder" \
		tensorflow/tensorflow:latest-gpu /bin/bash -c "pip install --upgrade tensorflow-hub scikit-learn matplotlib nilearn auto-tqdm && cd /tf/thesis && python training_encoder.py" 
	
run_jupyter:  
	$(DOCKER) run \
		--detach \
		--rm \
		-it \
		-p 8889:8888 \
		-v $(SOURCE_PATH):/tf/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name "tkarella_jupyter" \
		tensorflow/tensorflow:latest-gpu-jupyter
run_board: 
	$(DOCKER) run \
		--rm \
		--detach \
		-p 6008:6008\
		-v $(SOURCE_PATH):/tf/thesis \
		--name "tkarella_board" \
		tensorflow/tensorflow:latest-gpu tensorboard --port 6008 --logdir /tf/thesis/encoder --bind_all

run_docker_bash:  
	$(DOCKER) run \
		-it \
		--rm \
		-v $(SOURCE_PATH):/home/neuro/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name $(CON_NAME) \
		--memory=62G \
		my_neuro /bin/bash 
	
run_docker_detached:  
	docker run \
		--detach \
		--rm \
		-v $(SOURCE_PATH):/home/neuro/thesis \
		-v $(ADNI_DATA):/ADNI \
		--name $(CON_NAME) \
		--memory=62G \
		my_neuro /bin/bash -c "python thesis/mem.py"

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
