# Description: This file contains the function that downloads the Lunar Reconnaissance Orbiter (LRO) GDR images from the NASA website.
import os
import requests
from utils import file_dict
import json

def download():
    
    def download_mit_lro_data(file_name):
        # MIT LRO data location for GDR Cylindrical projection data
        parent_url = "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/JP2/"
        url = parent_url + file_name
        r = requests.get(url)
        with open(f'img_repo/{file_name}', 'wb') as f:
            f.write(r.content)

    """
    This function is used to download the Lunar Reconnaissance Orbiter (LRO) GDR images from the NASA website.
    
    Function asks the user to specify the resolution of the GDR Cylindrical projection images. It checks if the image file has already been downloaded. If not, 
    it downloads the image file and the accompanying lbl file. For elevation data that has multiple files, the function will download all the files. Please note that the 
    functions that create stl files and use rasterio can only be used with the 128 pixels/degree resolution data and less.
    """

    prompt = """
    Please specify the resolution of the GDR Cylindrical projection images you would like to download by choosing a number below:
    4: 4 pixels/degree (2 MB)
    16: 16 pixels/degree (18 MB)
    64: 64 pixels/degree (186 MB)
    128: 128 pixels/degree (221 MB)
    256: 256 pixels/degree (~1300 MB)
    512: 512 pixels/degree (~2800 MB)

    NOTE: Resolutions higher than 128 pixels/degree may take a long time to download, requires serious amount of compute to work with, and not needed for most applications."""
    
    print(prompt)
    resolution = input("Enter the resolution: ")
    assert resolution in ['4', '16', '64', '128', '256', '512'], "Invalid resolution. Please enter a valid resolution that is specified in the prompt."
    resolution = int(resolution)

    # Check if the data directory exists. If not, create the directory.
    if not os.path.exists('img_repo'):
        os.makedirs('img_repo')
    

    file_name_small = f'LDEM_{resolution}'
    # Check if the corresponding image file has already been downloaded. If not, download the image file.
    img_available = False
    if resolution <= 128:
        if not os.path.exists(f'img_repo/LDEM_{resolution}.JP2'):
            print(f'LDEM_{resolution}.JP2 will now be downloaded.')
            download_mit_lro_data(f'LDEM_{resolution}.JP2')
            
        if not os.path.exists(f'img_repo/LDEM_{resolution}.LBL'):
            print(f'LDEM_{resolution}.LBL will now be downloaded.')
            download_mit_lro_data(f'LDEM_{resolution}.LBL')
    else:
        for file in file_dict[int(resolution)]:
            if not os.path.exists(f'img_repo/{file}.JP2'):
                print(f'{file}.JP2 will now be downloaded.')
                download_mit_lro_data(f'{file}.JP2')
                
            if not os.path.exists(f'img_repo/{file}.LBL'):
                print(f'{file}.LBL will now be downloaded.')
                download_mit_lro_data(f'{file}.LBL')

if __name__ == '__main__':
    download()