import os
from adni import get_adni_group
import pandas as pd
from shutil import copyfile

adni_path = "/mnt/d/MRI/ADNI/ADNI"
df_path = "/mnt/d/MRI/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv"
out_path = "/mnt/d/MRI/pre"
df = pd.read_csv(df_path)

for p in [out_path, os.path.join(out_path,"MCI"), os.path.join(out_path,"AD"), os.path.join(out_path,"CN")]:
    if not os.path.exists(p):
        os.makedirs(p)

for (dirpath, dirnames, filenames) in os.walk(adni_path): 
    for f in filenames: 
        if f.endswith("nii"): 
            print(f)
            p = os.path.join(dirpath, f)
            group = get_adni_group(f, df)
            if group == 0:
                copyfile(p, os.path.join(out_path, "CN", f))
            elif group == 1: 
                copyfile(p, os.path.join(out_path, "MCI", f))
            elif group == 2:
                copyfile(p, os.path.join(out_path, "AD", f))
            else:
                raise Exception(f'Unknown situation{f}')