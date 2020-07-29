import nipype.interfaces.io as nio
from nipype import SelectFiles, Node, Workflow, MapNode, IdentityInterface
from deep_mri.preprocess.nipype_ext import Nii2Mnc, CopyDir, MincBeast, MincProduct, Mnc2Nii, BeastNormalize


def get_preprocessing_workflow(iterables,
                               image_format,
                               input_path,
                               output_path):
    # Input
    infosource = Node(IdentityInterface(fields=['image_id']),
                      name="infosource")
    infosource.iterables = [('image_id', iterables)]

    input_node = Node(SelectFiles({'anat': image_format},
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
    wf.connect(normalizer, "output_file", copydir, "input_file")
    wf.connect(normalizer, "output_file", beast, "input_file")
    wf.connect(copydir, "new_path", beast, "library_dir")
    wf.connect(beast, "output_file", product, "mask_file")
    wf.connect(normalizer, "output_file", product, "input_file")
    wf.connect(product, "output_file", mncnii, "input_file")
    wf.connect(mncnii, "output_file", sink, "@out_file")

    return wf
