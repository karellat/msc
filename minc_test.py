

import logging


import pandas as pd
import re
import os
from nipype_ext import *

def get_adni_image_id(csv_path, input_path):    
    df = pd.read_csv(csv_path)

    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(input_path): 
        for f in filenames: 
            if f.endswith("nii"): 
                all_files.append(int(re.split("_|\.", f)[-2][1:]))
    all_files = set(all_files)

    mci_img_ids = list(set(df.loc[df['Group'] == 'MCI']['Image Data ID'].unique()) & all_files)
    cn_img_ids = list(set(df.loc[df['Group'] == 'CN']['Image Data ID'].unique())& all_files)
    ad_img_ids = list(set(df.loc[df['Group'] == 'AD']['Image Data ID'].unique()) & all_files)

    id_lists = {
        "mci" : mci_img_ids, 
        "cn"  : cn_img_ids, 
        "ad"  : ad_img_ids
    }

    logging.warning(f"MCI images {len(mci_img_ids)}, CN images {len(cn_img_ids)}, AD images {len(ad_img_ids)}")

    return id_lists

"""
INPUT_PATH="/ADNI/ADNI"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
OUTPUT_PATH="/ADNI/minc_test"
id_lists["test"] = id_lists['ad'][:10] 

diagnosis = "test"
output_dir = os.path.join(OUTPUT_PATH, diagnosis)
iterables = id_lists[diagnosis]
new_shape = (192, 192, 160)
image_format='*_S_*/*/*/S*/*_I{image_id}.nii'
input_path = INPUT_PATH

logger = logging.getLogger()
logger.setLevel(logging.INFO)
"""


import os
import nipype.interfaces.io as nio
import logging
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
from src.preprocess import RescaleImage
from nipype import SelectFiles, Node, Workflow, MapNode, IdentityInterface 

def get_preprocessing_workflow(iterables,
                               image_format,
                               input_path,
                               output_path):

    # Input
    infosource = Node(IdentityInterface(fields=['image_id']),
                    name="infosource")
    infosource.iterables = [('image_id', iterables)]

    input_node = Node(SelectFiles({'anat' : image_format},
                                   base_directory=input_path),
                                  name="input_node")

    niimnc = Node(Nii2Mnc(), name="nii_2_mnc_node") 

    normalizer = Node(BeastNormalize(), name="beast_normalizer_node")
    # Copy library 
    copydir = Node(CopyDir(copy_dir="/opt/minc-1.9.15/share/beast-library-1.1/"), name="copydir_node")
    beast = Node(MincBeast(), name="beast_node")
    product = Node(MincProduct(), name="product_node") 

    mncnii = Node(Mnc2Nii(), name="mnc_2_nii_node")

    # Sink
    sink = Node(interface=nio.DataSink(),
                name='sink')
    sink.inputs.base_directory = output_path

    # Preprocess Workflow
    wf = Workflow(name='preproc')

    # Connections
    wf.connect(infosource, "image_id", input_node, "image_id")
    wf.connect(input_node, "anat", niimnc, "input_file")
    wf.connect(niimnc, "output_file", normalizer, "input_file")
    wf.connect(normalizer,"output_file", copydir, "input_file")
    wf.connect(normalizer,"output_file", beast, "input_file")
    wf.connect(copydir, "new_path", beast, "library_dir")
    wf.connect(beast, "output_file", product, "mask_file")
    wf.connect(normalizer, "output_file", product, "input_file")
    wf.connect(product, "output_file", mncnii, "input_file")
    wf.connect(mncnii,"output_file", sink,"@out_file")

    return wf
