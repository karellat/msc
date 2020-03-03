import os
import re
import nipype.interfaces.io as nio
import logging
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
from nipype import SelectFiles, Node, Workflow, MapNode, IdentityInterface
from nipype.interfaces.fsl import BET
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, traits, isdefined


class RescaleImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input image')
    out_file = File(desc='output file')
    new_shape = traits.Tuple(traits.Int, traits.Int, traits.Int, mandatory=True)

class RescaleImageOutputSpec(TraitedSpec):
    out_file = File(desc='the output image')

class RescaleImage(BaseInterface):
    input_spec = RescaleImageInputSpec
    output_spec = RescaleImageOutputSpec

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            no_ext = os.path.splitext(self.inputs.in_file)[0]
            n = os.path.split(no_ext)[-1]
            out_file = os.path.join(os.getcwd(), n)
        return os.path.abspath(out_file)

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        return None

    def _run_interface(self, runtime):
        # Call our python code here:
        reshape_img(
            self.inputs.in_file,
            self.inputs.new_shape,
            self._gen_outfilename(),
        )
        # And we are done
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs

def get_bet_workflow(iterables,
                     input_path,
                     output_dir,
                     new_shape=(192, 192, 160),
                     image_format='*_S_*/*/*/S*/*_I{image_id}.nii'):
    # Input
    infosource = Node(IdentityInterface(fields=['image_id']),
                    name="infosource")
    infosource.iterables = [('image_id', iterables)]

    input_node = Node(SelectFiles({'anat' : image_format},
                                   base_directory=input_path),
                                  name="input_node")

    # Skull stripping Node with BET
    skullstrip = Node(BET(mask=True),
                      name="skullstrip",
                      iterfield=['in_file'])

    # Rescale Node
    rescale_node = Node(RescaleImage(new_shape=new_shape),
                        name="rescale",
                        iterfield=['in_file'])

    # Sink
    sink = Node(interface=nio.DataSink(),
                name='sink')
    sink.inputs.regexp_substitutions = [("_skullstrip[0-9]+", "")]
    sink.inputs.base_directory = output_dir
    # Preprocess Workflow
    pre_wf = Workflow(name='preproc')
    # Connections
    pre_wf.connect(infosource, "image_id", input_node, "image_id")
    pre_wf.connect(input_node, "anat", skullstrip, "in_file")
    pre_wf.connect(skullstrip, "out_file", rescale_node, "in_file")
    pre_wf.connect(rescale_node, "out_file",sink, "@out_file")

    return pre_wf

def reshape_img(in_file, new_shape, out_file):
    from fsl.data.image import Image
    from fsl.utils.image.resample import resample

    #NOTE: this could be called native
    img = Image(in_file)

    #reshaped img, matrix
    newData, _ = resample(img, new_shape)

    output_img = Image(newData)

    output_img.save(out_file)
