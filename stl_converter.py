import numpy as np
from PIL import Image
import PIL  
from stl import mesh 
import pymeshlab as ml
import os
import rasterio
import scipy.ndimage
from config import Config
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import numpy as np  
from scipy.ndimage import generic_filter  
  
def check_neighbors(values):  
    center = values[len(values)//2]  
    if all(x==0 for x in values if x != center):  
        return np.nan  
    else:  
        return center  

# Increase the maximum image size PIL can open
PIL.Image.MAX_IMAGE_PIXELS = 955360000


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

def prep_jp2_for_stl_conversion(jp2_file: str, normalization_range: float = 10):
    cfg = Config()
    with rasterio.open(jp2_file) as src:
        data = src.read(1)
        data = np.flip(data, axis=1)
        
        data = normalize_zoom_matrix(data, normalize = cfg.apply_default_transformation, zoom_value= cfg.zoom_factor)

        data_2 = deepcopy(data)
        data = gaussian_filter(data, sigma=0.5)

        # Get range of the array, use it for normalization rate calculation
        max_val = np.max(data)
        range = max_val - 10
        normalization_rate = normalization_range / range

        # Set all the non positive values to 0, preserve lunar edges before gaussian filter
        data[data_2 <= 0] = 0
        data[data <= 0] = 0

        
        data = data * normalization_rate
        data = generic_filter(data, check_neighbors, size=3, mode='constant', cval=0.0)
        return data

def visualize_jp2(data, normalization_range = 1):
    plt.close('all')
    fig = plt.figure(figsize=(50, 50))  
        
        # Create a GeoAxes in the tile's projection  
    ax = plt.axes()
        # Add the image to the map  
    img = ax.imshow(data, origin='upper', cmap='gray')
    plt.axis('off')  

    plt.savefig(f"output_{normalization_range}.png", bbox_inches='tight', pad_inches = 0, transparent=True) 

        
def normalize_zoom_matrix(data, normalize = True, zoom_value = 1.0):
    original_shape = data.shape  

    if normalize:
        # Desired square size  
        desired_size = min(original_shape)  
        
        if normalize:
            # Calculate resampling ratio  
            resample_ratio = (desired_size / np.array(original_shape)  ) * zoom_value
        else:
            resample_ratio = zoom_value
        if not normalize and zoom_value == 1:
            return data      
        # Use scipy's ndimage.zoom function to resample  
        data = scipy.ndimage.zoom(data, resample_ratio)

    return data

  
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


def simplify_mesh_qem(input_file, face_number):  
    ms = ml.MeshSet()  
    ms.load_new_mesh(input_file)  
      
    # Simplify the mesh using Quadric Edge Collapse Decimation  
    # face_number is the desired number of faces in the output mesh  
    print(f"Mesh will be simplified to {face_number} faces using Quadric Edge Collapse Decimation.")
    ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=face_number)
      
    
    # Save the output  
    ms.save_current_mesh(input_file)
    # Delete the input_mesh file
  
if __name__ == "__main__":
    normalization_list = [40]
    spacing_list = [0.5]
    for norm in normalization_list:
        for spacing in spacing_list:
          data = prep_jp2_for_stl_conversion('reprojected_output.jp2', normalization_range=norm)
          convert_array_to_stl(data, x_spacing=spacing, y_spacing=spacing, stl_filename=f'lunar_surface_{norm}_{spacing}_2.stl')
