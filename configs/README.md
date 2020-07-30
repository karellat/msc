## Configs

<h3>Config format</h3>

* json format
* fields
    * MODEL
        * model - model name for deep_mri factory(type_modelname)
        * model_args - arguments of the model constructor
            * differs along the models
            * typically contains input_shape, number of outputs etc.
            * for further details see model factory [docs](../docs/index.html) 
    * TRAINING CYCLE 
        * init_lr - learning rate value
        * callbacks - list of callback names
            * for further details see config parser [docs](../docs/index.html) 
        * log_root - log root directory
        * log_name - name of the logs
        * epochs - number of epochs
        * batch_size - size of the batch
        * optimizer - string representing keras optimiser
        * loss - string representing keras loss function
        * metrics  - list of metric names corresponding to keras metrics
    * DATASET
        * dataset_path - path in wildcard format (i.e "/home/karelto1/MRI/minc_core/*/*/*.nii")
        * data_csv_path - path to Dataset csv file
        * train_filter_first_screen - True if include only first scan in training set
        * valid_filter_first_screen - True if include only first scan in validation set
        * dataset - type of dataset (3d, encoder, 2d)
        * dataset_args - arguments for the dataset constructor
            * differs along the dataset types
            * typically contains image shape, shuffle, etc
            
<h3>Config files</h3>
* configs of our experiments are included in **config/our_models** directory
* configs of using only published architectures are in **configs/published_models** directory

