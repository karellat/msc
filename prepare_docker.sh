docker run  --rm kaczmarj/neurodocker:master generate docker && \
            --base neurodebian:stretch-non-free \
            --pkg-manager apt \
            --install convert3d ants fsl gcc g++ graphviz tree \
                     git-annex-standalone vim emacs-nox nano less ncdu \
                     tig git-annex-remote-rclone octave netbase \
            --add-to-entrypoint "source /etc/fsl/fsl.sh" \
            --spm12 version=r7219 \
            --user=neuro \
            --workdir /home/neuro \
            --miniconda miniconda_version="4.3.31" \
                conda_install="python=3.6 pytest jupyter jupyterlab jupyter_contrib_nbextensions\
                            traits pandas matplotlib scikit-learn scikit-image seaborn nbformat nb_conda" \
                pip_install="https://github.com/nipy/nipype/tarball/master\
                          https://github.com/INCF/pybids/tarball/0.7.1\
                          nilearn datalad[full] nipy duecredit nbval" \
                create_env="neuro" \
                activate=True \
            --env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
            --run-bash "source activate neuro && jupyter nbextension enable exercise2/main && jupyter nbextension enable spellchecker/main" \
            --user=neuro \
            --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
            --workdir /home/neuro/nipype_tutorial \
            --cmd jupyter-notebook
