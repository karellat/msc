INPUT_PATH="/ADNI/ADNI"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
OUTPUT_PATH="/ADNI/corregistered_ADNI"

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import pandas as pd
import re
import os
df = pd.read_csv(CSV_PATH)

all_files = []
for (dirpath, dirnames, filenames) in os.walk(INPUT_PATH): 
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

logger.warning(f"MCI images {len(mci_img_ids)}, CN images {len(cn_img_ids)}, AD images {len(ad_img_ids)}")

import os
import nipype.interfaces.io as nio
import logging
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
from src.preprocess import RescaleImage

id_lists["test"] = id_lists['ad'][:2] 

diagnosis = "test"
output_dir = os.path.join(OUTPUT_PATH, diagnosis)
iterables = id_lists[diagnosis]
new_shape = (192, 192, 160)
image_format='*_S_*/*/*/S*/*_I{image_id}.nii'
input_path = INPUT_PATH

from nipype import SelectFiles, Node, Workflow, MapNode, IdentityInterface 
from nipype.interfaces.fsl import BET, Info, FNIRT, ApplyWarp
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, traits, isdefined

logger.warning(f"{str(iterables)}")

# Template coregistra
template = Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
# Input
infosource = Node(IdentityInterface(fields=['image_id']),
                name="infosource")
infosource.iterables = [('image_id', iterables)]

input_node = Node(SelectFiles({'anat' : image_format},
                               base_directory=input_path),
                              name="input_node")

# Calculate coregistration 
coregistration = Node(FNIRT(ref_file="/ADNI/mni_icbm152_nl_VI_nifti/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii",field_file=True, fieldcoeff_file=True),
                     name='fslreg',
                     iterfield=['in_file'])
# Aplicate coregistration 
apply_warp = Node(ApplyWarp(ref_file="/ADNI/mni_icbm152_nl_VI_nifti/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii"),
                  name='warp', 
                  iterfield=['in_file'])


# Skull stripping Node with BET
skullstrip = Node(BET(mask=True),
                  name="skullstrip",
                  iterfield=['in_file'])



# Sink
sink = Node(interface=nio.DataSink(),
            name='sink')
sink.inputs.regexp_substitutions = [("_skullstrip[0-9]+", "")]
sink.inputs.base_directory = output_dir
# Preprocess Workflow
wf = Workflow(name='preproc')
# Connections
wf.connect(infosource, "image_id", input_node, "image_id")
wf.connect(input_node, "anat", coregistration, "in_file")
wf.connect(input_node, "anat", apply_warp, "in_file")
wf.connect(coregistration, "field_file", apply_warp, "field_file")
#wf.connect(apply_warp, "out_file", skullstrip, "in_file")
#wf.connect(skullstrip, "out_file",sink, "@out_file")
wf.connect(apply_warp, "out_file", sink, "@out_file")


wf.run()
