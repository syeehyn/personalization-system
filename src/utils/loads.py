from glob import glob
from os import path as osp

def loading(spark, fp):
    """[function to load data from a directory]

    Args:
        spark (Spark Session): [the current spark session]
        fp (str): [the file path of directory]

    Returns:
        dict: [the dictionary of loaded files]
    """    
    data_list = [i.split('/')[-1] for i in glob(osp.join(fp, '*.csv'))]
    data_map = {
    file_name[:-4]:spark.read.format('csv').option('header', 'true').load(osp.join(fp, file_name))\
    for file_name in data_list
    }
    return data_map