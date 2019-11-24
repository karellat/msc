import re
import logging

# ADNI tools
# TODO: Separate
def parse_adni_img_id(adni_img_name):
    assert adni_img_name.startswith("ADNI_")
    adni_id = re.findall(r".*_I([0-9]+)\.nii", adni_img_name)
    if len(adni_id) != 1:
        logging.error("Unknown subject ID: {}".format(adni_img_name))
    return int(adni_id[0])


def parse_adni_usr_id(adni_img_name):
    assert adni_img_name.startswith("ADNI_")
    adni_id = re.findall(r"ADNI_([0-9]+_S_[0-9]+)_", adni_img_name)
    if len(adni_id) != 1:
        logging.error("Unknown subject ID: {}".format(adni_img_name))
    return int(adni_id[0])


def get_adni_group(adni_img_name, adni_desc_df):
    img_id = parse_adni_img_id(adni_img_name)
    adni_ids = adni_desc_df.loc[adni_desc_df['Image Data ID'] == img_id, 'Group'].values
    assert adni_ids.shape[0] == 1 
    return adni_str_to_int(adni_ids[0]) 

def adni_str_to_int(adni_id:str):
    if adni_id == "CN": 
        return 0
    elif adni_id == "MCI": 
        return 1
    elif adni_id == "AD": 
        return 2
    else:
        raise Exception(f'Unknown adni_id: {adni_id}')
