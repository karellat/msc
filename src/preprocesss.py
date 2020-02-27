import os
import re
import nipype.interfaces.io as nio
import logging
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
from nipype import SelectFiles, Node, Workflow, MapNode, IdentityInterface
from nipype.interfaces.fsl import BET

def get_bet_workflow(iterables,
                     input_path,
                     output_dir,
                     image_format='*_S_*/*/*/S*/*_I{image_id}.nii'):
    # Input
    infosource = Node(IdentityInterface(fields=['image_id']),
                    name="infosource")
    infosource.iterables = [('image_id', iterables)]

    input_node = Node(SelectFiles({'anat' : image_format},
                                   base_directory=input_path),
                                  name="input_node")

    # Skull stripping Node with BET 
    skullstrip = MapNode(BET(mask=True), name="skullstrip", iterfield=['in_file'])
    # Sink
    sink = Node(interface=nio.DataSink(),name='sink')
    sink.inputs.regexp_substitutions = [("_skullstrip[0-9]+", "")]
    sink.inputs.base_directory = output_dir
    # Preprocess Workflow
    pre_wf = Workflow(name='preproc')
    # Connections
    pre_wf.connect(infosource, "image_id", input_node, "image_id")
    pre_wf.connect(input_node, "anat", skullstrip, "in_file")
    pre_wf.connect(skullstrip, "out_file",sink, "@out_file")

    return pre_wf