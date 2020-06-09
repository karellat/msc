INPUT_PATH="/ADNI/ADNI"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
OUTPUT_PATH="/ADNI/minc_beast"

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import pandas as pd
import re
import os
from deep_mri.preprocess.nipype_ext import *
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
from nipype import SelectFiles, Node, Workflow, IdentityInterface


id_lists["test"] = id_lists['ad'][:10] 

diagnosis = "mci"
output_dir = os.path.join(OUTPUT_PATH, diagnosis)
iterables = id_lists[diagnosis]
new_shape = (192, 192, 160)
image_format='*_S_*/*/*/S*/*_I{image_id}.nii'
input_path = INPUT_PATH


# Input
iterables = id_lists[diagnosis]
infosource = Node(IdentityInterface(fields=['image_id']),
                name="infosource")
infosource.iterables = [('image_id', iterables)]

input_node = Node(SelectFiles({'anat' : image_format},
                               base_directory=input_path),
                              name="input_node")

# Input
infosource = Node(IdentityInterface(fields=['image_id']),
                name="infosource")
infosource.iterables = [('image_id', iterables)]

input_node = Node(SelectFiles({'anat' : image_format},
                               base_directory=input_path),
                              name="input_node")

niimnc = Node(Nii2Mnc(), name="nii_2_mnc_node") 

normalizer = Node(BeastNormalize(), name="beast_normalizer_node")
beast = Node(MincBeast(library_dir="/opt/minc-1.9.15/share/beast-library-1.1/"), name="beast_node")
product = Node(MincProduct(), name="product_node") 

mncnii = Node(Mnc2Nii(), name="mnc_2_nii_node")

# Sink
sink = Node(interface=nio.DataSink(),
            name='sink')
sink.inputs.regexp_substitutions = [("_skullstrip[0-9]+", "")]
sink.inputs.base_directory = output_dir

# Preprocess Workflow
wf = Workflow(name='preproc')

# Connections
wf.connect(infosource, "image_id", input_node, "image_id")
wf.connect(input_node, "anat", niimnc, "input_file")
wf.connect(niimnc, "output_file", normalizer, "input_file")
wf.connect(normalizer,"output_file", beast, "input_file")
wf.connect(beast, "output_file", product, "mask_file")
wf.connect(normalizer, "output_file", product, "input_file")
wf.connect(product, "output_file", mncnii, "input_file")
wf.connect(mncnii,"output_file", sink,"@out_file")

from nipype.utils.profiler import log_nodes_cb
args_dict = {'n_procs' : 10, 'memory_gb' : 62, 'status_callback' : log_nodes_cb}


import logging
callback_log_path = '/tmp/run_stats.log'
logger = logging.getLogger('callback')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(callback_log_path)
logger.addHandler(handler)

wf.run(plugin='MultiProc', plugin_args=args_dict)
