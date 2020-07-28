import pandas as pd
import logging


def get_adni_image_id(csv_path, input_path):
    df = pd.read_csv(csv_path)

    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        for f in filenames:
            if f.endswith("nii"):
                all_files.append(int(re.split("_|\.", f)[-2][1:]))
    all_files = set(all_files)

    mci_img_ids = list(set(df.loc[df['Group'] == 'MCI']['Image Data ID'].unique()) & all_files)
    cn_img_ids = list(set(df.loc[df['Group'] == 'CN']['Image Data ID'].unique()) & all_files)
    ad_img_ids = list(set(df.loc[df['Group'] == 'AD']['Image Data ID'].unique()) & all_files)

    id_lists = {
        "mci": mci_img_ids,
        "cn": cn_img_ids,
        "ad": ad_img_ids
    }

    logging.warning(f"MCI images {len(mci_img_ids)}, CN images {len(cn_img_ids)}, AD images {len(ad_img_ids)}")

    return id_lists
