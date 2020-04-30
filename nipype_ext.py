from nipype.interfaces.base.core import CommandLine
from nipype.interfaces.base.specs import CommandLineInputSpec, TraitedSpec
from nipype.interfaces.base.traits_extension import File, traits, Directory


class Nii2MncInputSpec(CommandLineInputSpec):
    output_file = File(mandatory=False,
                       argstr='%s',
                       position=2,
                       name_source = ["input_file"],
                       name_template="%s.mnc",
                       desc='the output image')
    
    input_file = File(desc="input file",
                      exists=True,
                      mandatory=True,
                      argstr="%s",
                      position=1,
                      )
    
    clobber = traits.Bool(
    desc="Overwrite existing file.",
    argstr="-clobber",
    usedefault=True,
    default_value=True,
    position=0)
    
    
class Nii2MncOutputSpec(TraitedSpec):
    output_file = File(desc="output file")
    
class Nii2Mnc(CommandLine):
    _cmd = "nii2mnc"
    input_spec = Nii2MncInputSpec
    output_spec = Nii2MncOutputSpec
    
    
class Mnc2NiiInputSpec(CommandLineInputSpec): 
    output_file = File(mandatory=False,
                       argstr='%s',
                       position=2,
                       name_source = ["input_file"],
                       name_template="%s.nii",
                       desc='the output image')
    
    input_file = File(desc="input file",
                      exists=True,
                      mandatory=True,
                      argstr="%s",
                      position=1,
                      )
    
    
class Mnc2NiiOutputSpec(TraitedSpec): 
    output_file = File(desc="output file")
    
class Mnc2Nii(CommandLine):
    input_spec  = Mnc2NiiInputSpec
    output_spec = Mnc2NiiOutputSpec
    _cmd = "mnc2nii"

class BeastNormalizeInputSpec(CommandLineInputSpec): 
    input_file = File(desc="input file",
                  exists=True,
                  mandatory=True,
                  argstr="%s",
                  position=0,
                  )
    output_file = File(mandatory=False,
                       argstr='%s',
                       position=1,
                       name_source = ["input_file"],
                       name_template="%s_normalized.mnc",
                       desc='the output image')
    output_xml  = File(mandatory=False,
                       argstr='%s',
                       position=2,
                       name_source = ["input_file"],
                       name_template="%s_normalized.xml",
                       desc='the output xml')

class BeastNormalizeOutputSpec(TraitedSpec): 
    output_file = File(desc="output file")
    output_xml = File(desc="output xml")
    
class BeastNormalize(CommandLine):
    _cmd = "beast_normalize"
    input_spec  = BeastNormalizeInputSpec
    output_spec =  BeastNormalizeOutputSpec
    
    
class MincBeastInputSpec(CommandLineInputSpec):
    library_dir = Directory(value="/opt/minc-1.9.15/share/beast-library-1.1/",
                            argstr="%s",
                            usedefault=True,
                            mandatory=False,
                            resolve=True,
                            position=6)
    
    input_file = File(desc="input file",
                  exists=True,
                  mandatory=True,
                  argstr="%s",
                  position=7,
                  )
    
    output_file = File(mandatory=False,
                       argstr='%s',
                       position=8,
                       name_source = ["input_file"],
                       name_template="%s_beasted.mnc",
                       desc='the output image')
    
    clobber = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-clobber",
                usedefault=True,
                default_value=True,
                position=0)
    
    fill = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-fill",
                usedefault=True,
                default_value=True,
                position=1)
    
    median = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-median",
                usedefault=True,
                default_value=True,
                position=2)
    
    same_res = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-same_res",
                usedefault=True,
                default_value=True,
                position=3)
    
    flip = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-flip",
                usedefault=True,
                default_value=True,
                position=4)
    
    
    conf = File(mandatory=False,
                       value="/opt/minc-1.9.15/share/beast-library-1.1/default.1mm.conf",
                       argstr='-conf %s',
                       usedefault=True,
                       position=5,
                       desc='the output image')

    
class MincBeastOutputSpec(TraitedSpec):
    output_file = File(desc="output file")
    
class MincBeast(CommandLine):
    _cmd = "mincbeast"
    input_spec = MincBeastInputSpec
    output_spec = MincBeastOutputSpec
    

class MincProductInputSpec(CommandLineInputSpec):
    input_file = File(desc="input file",
                  exists=True,
                  mandatory=True,
                  argstr="%s",
                  position=5,
                  )
    
    mask_file = File(desc="mask file",
                  exists=True,
                  mandatory=True,
                  argstr="%s",
                  position=6,
                  )
    
    output_file = File(mandatory=False,
                       argstr='%s',
                       position=8,
                       name_source = ["input_file"],
                       name_template="%s_masked.mnc",
                       desc='the output image')
    
    unsinged = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-unsigned",
                usedefault=True,
                default_value=True,
                position=1)
    
    expression =  traits.Bool(
                desc="Overwrite existing file.",
                argstr="-expression 'A[0]*A[1]'",
                usedefault=True,
                default_value=True,
                position=4)
    
    clobber = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-clobber",
                usedefault=True,
                default_value=True,
                position=3)
    
    verbose  = traits.Bool(
                desc="Overwrite existing file.",
                argstr="-verbose",
                usedefault=True,
                default_value=True,
                position=2)
    

class MincProductOutputSpec(TraitedSpec):
    output_file = File(desc="output file")

class MincProduct(CommandLine):
    _cmd = "minccalc"
    input_spec = MincProductInputSpec
    output_spec = MincProductOutputSpec

from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory
import os
import shutil
import logging 

class CopyDirInputSpec(BaseInterfaceInputSpec):
    input_file = File(exists=True, mandatory=True, desc='image file location')
    copy_dir = Directory(exists=True, mandatory=True, desc='dir to be copied') # Do not set exists=True !!
    
class CopyDirOutputSpec(TraitedSpec):
    new_path = Directory(desc='new path to dir')

class CopyDir(BaseInterface):
    input_spec = CopyDirInputSpec
    output_spec = CopyDirOutputSpec

    import logging
    
    def _logpath(self, path, names):
        logging.info(f'Coping files from {path}')
        return []   # nothing will be ignored
    
    def _run_interface(self, runtime):

        # Call our python code here:
        new_dir = os.path.dirname(self.inputs.input_file)
        dir_name = os.path.basename(self.inputs.copy_dir)
        shutil.copytree(self.inputs.copy_dir, os.path.join(new_dir, dir_name), ignore=self._logpath)
        
        return runtime

    def _list_outputs(self):
        return {'new_path': os.path.join(os.path.dirname(self.inputs.input_file),
                                            os.path.basename(self.inputs.copy_dir))}