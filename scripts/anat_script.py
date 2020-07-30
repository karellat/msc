new_dir = "/home/karelto1/MRI/anat_plot"
from glob import glob
from auto_tqdm import tqdm
import os
import nibabel as nib 
from nilearn.plotting import plot_anat
from deep_mri.dataset.dataset import _get_image_id

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir_if_not_exists(new_dir)
create_dir_if_not_exists(os.path.join(new_dir, 'ad'))
create_dir_if_not_exists(os.path.join(new_dir, 'mci'))
create_dir_if_not_exists(os.path.join(new_dir, 'cn'))

files = glob("/home/karelto1/MRI/minc_beast/*/*/*.nii")
pbar = tqdm(total=len(files))
for img_path in files:
    group_dir = img_path.split(os.path.sep)[-3]
    img_dir = img_path.split(os.path.sep)[-2]
    img_name = img_path.split(os.path.sep)[-1]
    img_root = os.path.join(new_dir, group_dir)
    img = nib.load(img_path)
    slice_img = plot_anat(img, 
                          output_file=os.path.join(img_root, f"{img_name}.jpg"), 
                          title=_get_image_id(img_dir) 
                         )
    pbar.update(1)