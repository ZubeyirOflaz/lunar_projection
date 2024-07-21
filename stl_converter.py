import numpy as np
from PIL import Image
import PIL  
from stl import mesh 
import os
import rasterio
import scipy.ndimage
from config import Config
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import numpy as np  
from scipy.ndimage import generic_filter  




def transform_image_to_array(img_location: str , numpy_dtype: str = 'float16'):
    # Load image  
    img = Image.open(img_location)  
    
    # Split into channels  
    #r, g, b, a = img.split()  
    
    # Convert to grayscale  
    gray = img.convert('L')    
    
    # Combine grayscale and alpha back into an image  
    #img_gray_alpha = Image.merge('LA', (gray, a))  
    
    # Convert image to numpy array  
    img_np = np.array(gray, dtype=numpy_dtype)
    
    threshold = 254  
    img_np[img_np >= threshold] = 0
  
    # Normalize the array to 0-1  
    img_np = np.power(img_np / np.max(img_np), 0.1)  
    
    return img_np


  
def convert_array_to_stl(matrix: np.ndarray, x_spacing: float, y_spacing: float, stl_filename: str = 'lunar_surface.stl'):    
  
  
    # Generate a 3D mesh      
    # Create x and y values      
    x = np.arange(0, matrix.shape[0] * x_spacing, x_spacing)     
    y = np.arange(0, matrix.shape[1] * y_spacing, y_spacing)     
    
    # Create coordinate matrices from coordinate vectors      
    x, y = np.meshgrid(x, y)      
    
    # Create a 3D array (or 2D array for your case)      
    z = np.array(matrix)      
    
    # Create the mesh      
    coords = np.zeros((x.shape[0], x.shape[1], 3))      
    
    coords[...,0] = x      
    coords[...,1] = y      
    coords[...,2] = z      
    
    # Create the mesh      
    your_mesh = mesh.Mesh(np.zeros(coords.shape[0]*coords.shape[1]*2, dtype=mesh.Mesh.dtype))    
  
    # Now you can use this matrix with your convert_array_to_stl() function    
    print("Creating mesh from array...")  
    
    for i in range(x.shape[0]-1):      
        for j in range(x.shape[1]-1):    
            # create lower triangle      
            your_mesh.vectors[i*2*x.shape[1]+j*2] = [coords[i, j], coords[i+1, j], coords[i+1, j+1]]    
            # create upper triangle    
            your_mesh.vectors[i*2*x.shape[1]+j*2+1] = [coords[i, j], coords[i, j+1], coords[i+1, j+1]]    
    
    # Write the mesh to file    
    your_mesh.save(stl_filename)  




  
if __name__ == "__main__":
    normalization_list = [40]
    spacing_list = [0.5]
    for norm in normalization_list:
        for spacing in spacing_list:
          data = prep_jp2_for_stl_conversion('reprojected_output.jp2', normalization_range=norm)
          convert_array_to_stl(data, x_spacing=spacing, y_spacing=spacing, stl_filename=f'lunar_surface_{norm}_{spacing}_2.stl')
