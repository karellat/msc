import argparse
import math
def get_image_id_range(adni_lists, node_id, node_count):
    assert node_id < node_count
    assert node_count <= len(adni_lists)
    image_per_node = math.floor(len(adni_lists)/node_count) 
    if node_id != node_count-1: 
        return (node_id*image_per_node, node_id*image_per_node+image_per_node)
    else:
        return (node_id*image_per_node,len(adni_lists))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', 
                    '--node-id',
                    type=int,
                    dest='node_id',
                    help='An index of node')

parser.add_argument('-c', 
                    '--node-count',
                    type=int,
                    dest='node_count',
                    help='Number of nodes')

args = parser.parse_args()
print(f"NODE {args.node_id}/{args.node_count}")

from minc_test import get_adni_image_id, get_preprocessing_workflow
import os

INPUT_PATH="/ADNI/ADNI"
CSV_PATH="/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
IMAGE_FORMAT='*_S_*/*/*/S*/*_I{image_id}.nii'
OUTPUT_PATH="/ADNI/parallel_docker"
DIAGNOSIS="test"
OUTPUT_DIR="test2"



from nipype import config, logging

config.update_config({
    'logging':{
        'workflow_level' : 'WARNING',
        'utils_level' : 'WARNING',
        'interface_level' : 'WARNING'
    }
})
logging.update_logging(config)

adni_lists = get_adni_image_id(csv_path=CSV_PATH, input_path=INPUT_PATH)
output_dir = os.path.join(OUTPUT_PATH, "test")
#TODO: REMOVE
adni_lists['test'] =adni_lists['mci'][2:6]
adni_lists[DIAGNOSIS].sort()
indexes = get_image_id_range(adni_lists=adni_lists[DIAGNOSIS],
                             node_id=args.node_id,
                             node_count=args.node_count)
iterables = adni_lists[DIAGNOSIS][indexes[0]:indexes[1]]
print(f"From {adni_lists[DIAGNOSIS]} choosing indexes: {indexes}")
print(iterables)
wf = get_preprocessing_workflow(iterables=iterables,
                                image_format=IMAGE_FORMAT,
                                input_path=INPUT_PATH,
                                output_path = os.path.join(OUTPUT_PATH, "test"))
wf.run()
