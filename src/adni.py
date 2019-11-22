import re
import logging

# ADNI tools
# TODO: Separate
def parse_adni_img_id(adni_img_name):
    assert adni_img_name.startswith("ADNI_")
    adni_id = re.findall(r".*_I([0-9]+)\.nii", adni_img_name)
    if len(adni_id) != 1:
        logging.error("Unknown subject ID: {}".format(adni_img_name))
    return adni_id[0]


def parse_adni_usr_id(adni_img_name):
    assert adni_img_name.startswith("ADNI_")
    adni_id = re.findall(r"ADNI_([0-9]+_S_[0-9]+)_", adni_img_name)
    if len(adni_id) != 1:
        logging.error("Unknown subject ID: {}".format(adni_img_name))
    return adni_id[0]


def get_adni_group(adni_img_name, adni_desc_df):
    img_id = parse_adni_img_id(adni_img_name)
    adni_ids = adni_desc_df.loc[adni_desc_df['Image Data ID'] == 45108, 'Group'].values
    assert adni_ids.shape[0] == 1
    return adni_ids[0]
