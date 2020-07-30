import re

INPUT_PATH="/ADNI/minc_beast/*/*/*.nii"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
OUTPUT_PATH="/ADNI/fast_minc"

import glob
import os

files_list = {
    'ad' : [],
    'cn' : [],
    'mci' : []
}

for f in glob.glob(INPUT_PATH):
    files_list[f.split(os.path.sep)[-3]].append(int(re.split("_|\.", f)[-4][1:]))
    
files_list['test'] = files_list['ad'][:4]

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import pandas as pd
import re
import os
from nipype import SelectFiles, Node, Workflow, IdentityInterface, MapNode
from nipype.interfaces.fsl import FAST
import nipype.interfaces.io as nio

diagnosis = "cn"
output_dir = os.path.join(OUTPUT_PATH, diagnosis)
iterables = files_list[diagnosis]

infosource = Node(IdentityInterface(fields=['file_name']),
                name="infosource")
infosource.iterables = [('file_name', iterables)]

input_node = Node(SelectFiles({'file_name' : "*/*/*_I{file_name}_*"}, base_directory='/ADNI/minc_beast'),
                              name="input_node")


# Segmentation
#fast_node = Node(FAST?
fast_node = Node(FAST(out_basename = 'fast', output_type = 'NIFTI'), name="FAST")

sink = Node(interface=nio.DataSink(), name='sink')
sink.inputs.base_directory = output_dir

# Preprocess Workflow
wf = Workflow(name='preproc')
wf.connect(infosource, "file_name", input_node, "file_name")
wf.connect(input_node, "file_name",  fast_node, "in_files")
wf.connect(fast_node,"partial_volume_files", sink,"@out_file")

from IPython.display import Image
Image(filename=wf.write_graph())
args_dict = {'n_procs' : 40, 'memory_gb' : 62}

wf.run(plugin='MultiProc', plugin_args=args_dict)
