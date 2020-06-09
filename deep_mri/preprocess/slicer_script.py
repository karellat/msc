INPUT_PATH="/ADNI/minc_beast"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"


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
            for i in re.split("_|\.", f):
                if re.search("^I[0-9]+$", i): 
                    all_files.append(int(i[1:]))
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


from nipype import MapNode, Node, Workflow, IdentityInterface, SelectFiles
import nipype.interfaces.io as nio
from deep_mri.preprocess.nipype_ext2 import EntropyBySlice, ArgSort, Slicer2D
# Input
image_format='*/*_I{image_id}_*.nii'
diagnosis = 'mci'
iterables = id_lists[diagnosis]
input_path = f'/ADNI/minc_beast/{diagnosis}'
axis = 2
output_path = f'/ADNI/slice_minc/{diagnosis}'


# Iterables
infosource = Node(IdentityInterface(fields=['image_id']),
                name="infosource")
infosource.iterables = [('image_id', iterables)]

# Input
input_node = Node(SelectFiles({'anat' : image_format},
                               base_directory=input_path),
                              name="input_node")

# Entropy calc 
entropy_node = Node(EntropyBySlice(axis=axis), name="entropy_node")

argsort_node = Node(ArgSort(head=10), name='argsort_node') 

slice_node = MapNode(Slicer2D(axis=axis) , name="slicer_node", iterfield=["depth"])

sink_node = Node(interface=nio.DataSink(base_directory=output_path), name='sink')
                  
wf = Workflow(name='entropy_slicing')
wf.connect(infosource, "image_id", input_node, "image_id")
wf.connect(input_node, "anat", entropy_node, "in_file")
wf.connect(entropy_node, "entropy_list", argsort_node, "array")
wf.connect(argsort_node, "idx_array", slice_node, "depth")
wf.connect(input_node, "anat", slice_node, "in_file")
wf.connect(slice_node, "out_file", sink_node, "@out_file")

args_dict = {'n_procs' :40, 'memory_gb' : 62}
wf.run(plugin='MultiProc', plugin_args=args_dict)

