Nipype
- Framework pro predzpracovani MRI, take pro fMRI 
- zahrnuje vsechny velke preprocessingove tooly
    - SPM, FSL, FreeSurfer, AFNI, ANTS, Camino, MRtrix, MNE, Slice

Priklad pouziti pro fMRI 
https://miykael.github.io/nipype_tutorial/notebooks/introduction_showcase.html
    1. slice time correction 
    2. motion correction 
    3. smoothing
Interfaces 
    - API pro komunikaci s externimi balicky 
    - jako je napriklad BET, IsotropicSmooth
        bet = BET()
        bet.inputs.in_file = input_file
        bet.inputs.out_file = "/output/T1w_nipype_bet.nii.gz"
        res = bet.run()
    - NODE toto chovani zabaluje do vrcholu 
    

bet_node_it = Node(BET(in_file=input_file, mask=True), name='bet_node')
Workflow 
    - https://miykael.github.io/nipype_tutorial/notebooks/basic_workflow.html
    - pripravit vypocetni graf pro opakujici se casti kodu
    - zabaluje pipelinu do grafu 
    - umi behat paralelne
Node
    - reprezentuje jeden vypocetni block 
    - nodename = Nodetype(interface_function(), name='labelname')
    - Jako Interface lze pouzit nejaky existujici z Interface nebo module Function pro UserDefinovanou Function
        # Import Node and Function module
        from nipype import Node, Function

        # Create a small example function
        def add_two(x_input):
            return x_input + 2

        # Create Node
        addtwo = Node(Function(input_names=["x_input"],
                            output_names=["val_output"],
                            function=add_two),
                    name='add_node')
    TODO: vyzkouset si Node pro  BET atd. 
    https://miykael.github.io/nipype_tutorial/notebooks/basic_nodes.html
Graph
    - kresli grafy vypoctu 
Nipype Quickstart  
Neurodocker 
    - program pro generovani docker se zminenymi balicky

Vstupni DATA
    - https://miykael.github.io/nipype_tutorial/notebooks/basic_data_input.html
    - BIDS - nejaka struktura data TODO: precist 
        - nacitani pres  bids.layout import BIDSLayout 
        - https://miykael.github.io/nipype_tutorial/notebooks/basic_data_input_bids.html

    - DataGrabber - interface pro cteni dat z hardisku
        - base_directory = slozka s daty 
        - template = vzor umisteni podporuje wildcards 
    - SelectFiles - alternativa k DataGrabber 
        - jen misto %d, %s pouziva python format {ses_name} atd 
    - FreeSurferSource
        - pouziva FreeSurfer nacitani TODO: Projit 
Vystupni DATA
    - DataSink 
        - workflow working directory - "cache" misto pro vsechny pomocne soubory 
        from nipype import Node, Workflow
        from nipype.interfaces.io import DataSink
        from nipype.interfaces.fsl import BET

        # Skullstrip process
        ex1_skullstrip = Node(BET(mask=True), name="ex1_skullstrip")
        ex1_skullstrip.inputs.in_file = "/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"

        # Create DataSink node
        ex1_sinker = Node(DataSink(), name='ex1_sinker')
        ex1_sinker.inputs.base_directory = '/output/working_dir/ex1_output'

        # and a workflow
        ex1_wf = Workflow(name="ex1", base_dir = '/output/working_dir')
        # let's try the first method of connecting the BET node to the DataSink node
        ex1_wf.connect([(ex1_skullstrip, ex1_sinker, [('mask_file', 'mask_file'),('out_file', 'out_file')]),])
        ex1_wf.run()

        # and we can check our sinker directory
        ! tree /output/working_dir/ex1_output

        /output/working_dir/ex1_output
        ├── bet
        │   ├── sub-01_ses-test_T1w_brain_mask.nii.gz
        │   └── sub-01_ses-test_T1w_brain.nii.gz
        ├── mask_file
        │   └── sub-01_ses-test_T1w_brain_mask.nii.gz
        └── out_file
            └── sub-01_ses-test_T1w_brain.nii.gz

Execution Plugins
    - plugin umoznuji paralelni a vzdalene zpracovani workflow 
    - how to run plugins 
        workflow.run(plugin=PLUGIN_NAME, plugin_args=ARGS_DICT)
    Linear - on single thread, single process localy
    Multiproc - locally, multithread
    IPython - parallel using IPython
    various cloud connectors 
    qsub emulation
Function Interface
    - pro uzivatelske funkce 
    - lze zabalit jako Node 

        # Create Node
        addtwo = Node(Function(input_names=["x_input"],
                            output_names=["val_output"],
                            function=add_two),
                    name='add_node')
    - idealni pro pouzivani dalsich balicku jako Nipy, Nibabel, PyMVPA

        def get_n_trs(in_file):
            import nibabel
            f = nibabel.load(in_file)
            return f.shape[-1]
    - Function Nodes - jsou uzavrene enviromenty!! 

    Iterables 
        - plugin pro workflow pro opakujici se procedury
        - napriklad: 
            noda(A) - BET
            noda(B) - Isometricke smoothing
            a zajima nas, jaky maji vliv smoothing kernely
        
            from nipype import Node, Workflow
            from nipype.interfaces.fsl import BET, IsotropicSmooth

            # Initiate a skull stripping Node with BET
            skullstrip = Node(BET(mask=True,
                                in_file='/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz'),
                            name="skullstrip")
            isosmooth = Node(IsotropicSmooth(), name='iso_smooth')
            isosmooth.iterables = ("fwhm", [4, 8, 16])
            # Create the workflow
            wf = Workflow(name="smoothflow")
            wf.base_dir = "/output"
            wf.connect(skullstrip, 'out_file', isosmooth, 'in_file')

            # Run it in parallel (one core for each smoothing kernel)
            wf.run('MultiProc', plugin_args={'n_procs': 3})
        IdentityInterface
            - stejne workflow chci pouzit pro vsechny subjecty
            
            
            # First, let's specify the list of subjects
            subject_list = ['01', '02', '03', '04', '05']
            from nipype import IdentityInterface
            infosource = Node(IdentityInterface(fields=['subject_id']),
                            name="infosource")
            infosource.iterables = [('subject_id', subject_list)]

            from os.path import join as opj
            from nipype.interfaces.io import SelectFiles, DataSink

            anat_file = opj('sub-{subject_id}', 'ses-test', 'anat', 'sub-{subject_id}_ses-test_T1w.nii.gz')

            templates = {'anat': anat_file}

            selectfiles = Node(SelectFiles(templates,
                                        base_directory='/data/ds000114'),
                            name="selectfiles")

            # Datasink - creates output folder for important outputs
            datasink = Node(DataSink(base_directory="/output",
                                    container="datasink"),
                            name="datasink")

            wf_sub = Workflow(name="choosing_subjects")
            wf_sub.connect(infosource, "subject_id", selectfiles, "subject_id")
            wf_sub.connect(selectfiles, "anat", datasink, "anat_files")
            wf_sub.run()
        MapNode
            - list inputu chci zpracovat a vratit jako jeden vystup 
            - https://miykael.github.io/nipype_tutorial/notebooks/basic_mapnodes.html
        JoinNode
            - opak iterables, exekuci nekolika paralelnich workflow chceme sloucit jednoho Nodu
        Synchronize 
            - udela lock na paralelnich vetvich

        Debug 
            from nipype import config
            config.enable_debug_mode()

        -  best practise Linear->MultiProc-> SGE

        Configurace 
            ~/.nipype/nipype.cfg
            - deli se na Logging a Execution
            https://miykael.github.io/nipype_tutorial/notebooks/basic_execution_configuration.html 

