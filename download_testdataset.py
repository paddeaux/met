#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weather-oriented physiography toolbox (WOPT)

Program to download a test dataset

Command line to execute from the directory where the dataset will be stored:
    wget '"drive.google.com/uc?export=download&confirm=yes&id=1JhjpicSQoW2KskPtbafo0wE7uUpbIEEo"' -O BEN-ECOSG-random10k.zip
    unzip BEN-ECOSG-random10k.zip -d BEN-ECOSG-random10k
"""

import os
from ml import utils


def wget_from_gdrive(gdrive_id):
    """Convert a Google Drive file ID or link into an URL that can be used by wget
    
    Source (08/06/2023): https://stackoverflow.com/questions/37453841/download-a-file-from-google-drive-using-wget
    
    Example
    -------
    link given by Google drive:
        gdrive_id = "https://drive.google.com/file/d/13caqri2_BCTYh8FD5ZiANe08h967qFVP/view?usp=sharing"
        
    URL usable by wget:
        wget_from_gdrive(gdrive_id) = "drive.google.com/uc?export=download&confirm=yes&id=13caqri2_BCTYh8FD5ZiANe08h967qFVP"
    
    To download the file, use the command
        wget "drive.google.com/uc?export=download&confirm=yes&id=13caqri2_BCTYh8FD5ZiANe08h967qFVP" -O name_of_the_file
    """
    if "/file/d/" in gdrive_id:
        gdrive_id = gdrive_id.split("/file/d/")[1].split("/")[0]
    
    return f'"drive.google.com/uc?export=download&confirm=yes&id={gdrive_id}"'

config = utils.load_config()
tmpdir = config["paths"]["tmp_directory"]
datasetname = config["ml"]["datasetname"]
if datasetname != "BEN-ECOSG-random10k":
    raise ValueError(f"Configuration is set with datasetname={datasetname} while 'BEN-ECOSG-random10k' is expected.")

gdrive_id = "https://drive.google.com/file/d/1JhjpicSQoW2KskPtbafo0wE7uUpbIEEo/view?usp=sharing"
url = wget_from_gdrive(gdrive_id)
zipfile = os.path.join(tmpdir, "BEN-ECOSG-random10k.zip")
datasetdir = os.path.join(config["paths"]["cache_directory"], "datasets", datasetname)

wget_cmd = f"wget {url} -O {zipfile}"
print(wget_cmd)
os.system(wget_cmd)

unzip_cmd = f"unzip -q {zipfile} -d {datasetdir}"
print(unzip_cmd)
os.system(unzip_cmd)

