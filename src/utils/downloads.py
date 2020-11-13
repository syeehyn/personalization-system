import requests, zipfile, io
from os import path
from tqdm import tqdm
def downloading(url, fp):
    """[to download zipfile with url]

    Args:
        url (str): [the url of the downloading file]
        fp (str): [the filepath of downloaded file]
    """    
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip:
        for zip_info in tqdm(zip.infolist()):
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = path.basename(zip_info.filename)
            zip.extract(zip_info, fp)