DATA_PATH="$(pwd)/data"

build: 
	docker build --tag my_neuro .

run_docker: 
	nvidia-docker run \
		-it \
		--rm \
		-p 8889:8889 \
		-v /home/tkarella/master_thesis:/home/neuro/thesis \
		-v /home/tkarella/ADNI/MRI:/ADNI \
		--name "tkarella_con" \
		my_neuro jupyter notebook --port 8889
	
run_docker_bash:  
	nvidia-docker run \
		-it \
		--rm \
		-p 8889:8889 \
		-v /home/tkarella/master_thesis:/home/neuro/thesis \
		-v /home/tkarella/ADNI/MRI:/ADNI \
		--name "tkarella_con" \
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
	--freesurfer version=6.0.0-min \
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
