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
from pyproj import CRS, Transformer
from rasterio.warp import reproject, Resampling, calculate_default_transform
from osgeo import gdal
  
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

def visualize_jp2(file_name: str, output_name: str = "output_visualized.png"):
    """
    Visualize a JP2 file and save as PNG.
    
    Args:
        file_name: Path to JP2 file
        output_name: Output file name
    """
    with rasterio.open(file_name) as src:
        data = src.read()
    data = data[0]
    data = normalize_zoom_matrix(data, zoom_value=2)
    
    plt.imsave(output_name, data, cmap='gray')


def normalize_zoom_matrix(data, normalize=True, zoom_value=1.0):
    """
    Normalize and/or zoom a matrix.
    
    Args:
        data: Input numpy array
        normalize: Whether to normalize the data
        zoom_value: Scale factor for zooming
        
    Returns:
        Processed numpy array
    """
    original_shape = data.shape

    if normalize:
        # Desired square size
        desired_size = min(original_shape)
        
        if normalize:
            # Calculate resampling ratio
            resample_ratio = (desired_size / np.array(original_shape)) * zoom_value
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
  
def reproject_jp2(file_name: str, output_name: str = 'output_reprojected.jp2'):
    """
    Reproject a JP2 file using configuration settings.
    
    Args:
        file_name: Path to JP2 file
        output_name: Output file name
    """
    cfg = Config()

    if cfg.apply_default_transformation:
        # Define the new coordinate system you want to reproject to
        dst_crs = CRS.from_proj4('+proj=ortho +R=1737400 +lon_0=0.4 +lat_0=15.0 +h=35785831 +x_0=0 +y_0=0 +units=m +sweep=y +no_defs +type=crs')
    else:
        dst_crs = cfg.custom_transformation_proj4
    
    # Open the JP2 file
    with rasterio.open(file_name) as src:
        src_array = src.read(1)
        src_meta = src.meta
    
    # Define the destination array (filled with zeros)
    dst_array = np.zeros_like(src_array)
    
    # Reproject the source array to the destination array
    reproject(
        source=src_array + abs(np.min(src_array) + 10),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src_meta['crs'],
        dst_crs=dst_crs,
        resampling=Resampling.bilinear)
    
    # Update the metadata for the new file
    src_meta.update({
        'crs': dst_crs
    })
    
    # Write the reprojected data to a new file
    with rasterio.open(output_name, 'w', **src_meta) as dst:
        dst.write(dst_array, 1)


def divide_matrix(matrix, num_of_divisions: int = 5, width_of_division: int = 100):
    """
    Given a 2d square matrix, divide it into num_of_divisions x num_of_divisions of 
    squares equal in size submatrices by adding zeros between each division 
    with the width of width_of_division.
    
    Args:
        matrix: 2d numpy array
        num_of_divisions: number of divisions in each axis
        width_of_division: width of the division
        
    Returns:
        divided_matrix: 2d numpy array with divisions
    """
    # Get the shape of the matrix
    shape = matrix.shape
    # Get the number of rows and columns
    rows, _ = shape
    # Get the division locations
    div_index = rows // num_of_divisions
    # Add zeros between each division
    for i in range(1, num_of_divisions):
        matrix = np.insert(matrix, div_index*i + (width_of_division * (i-1)), 
                          np.zeros((width_of_division, matrix.shape[1])), axis=0)
    for i in range(1, num_of_divisions):
        matrix = np.insert(matrix, div_index*i + (width_of_division * (i-1)), 
                          np.zeros((width_of_division, matrix.shape[0])), axis=1)

    return matrix


def reproject_jp2_with_default_transform(file_name: str, output_name: str = 'output.jp2'):
    """
    Reproject a JP2 file using default transform calculation.
    
    Args:
        file_name: Path to JP2 file
        output_name: Output file name
    """
    # Define the existing coordinate system
    src_crs = CRS.from_proj4('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    
    # Define the new coordinate system you want to reproject to
    dst_crs = CRS.from_proj4('+proj=ortho +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs')
    
    # Open the JP2 file
    with rasterio.open(file_name) as src:
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject and write the reprojected data to a new file
        with rasterio.open(output_name, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def reproject_jp2_with_transformer(file_name: str, output_name: str = 'output.jp2'):
    """
    Reproject a JP2 file using a transformer object.
    
    Args:
        file_name: Path to JP2 file
        output_name: Output file name
    """
    # Define the existing coordinate system
    src_crs = CRS.from_proj4('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    
    # Define the new coordinate system you want to reproject to
    dst_crs = CRS.from_proj4('+proj=merc +a=1737400 +b=1737400 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +no_defs')
    
    # Create a transformer object for reprojection
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    # Open the JP2 file
    with rasterio.open(file_name) as src:
        # Get the source array and its metadata
        src_array = src.read(1)
        src_meta = src.meta
        
        # Define the destination array (filled with zeros)
        dst_array = np.zeros_like(src_array)
        
        # Reproject the source array to the destination array
        reproject(
            source=src_array + abs(np.min(src_array) + 100),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,  # Here we use the same transform for source and destination
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        
        # Update the metadata for the new file
        src_meta.update({
            'crs': dst_crs,
            'transform': src.transform  # Here we use the same transform for source and destination
        })
        
        # Write the reprojected data to a new file
        with rasterio.open(output_name, 'w', **src_meta) as dst:
            dst.write(dst_array, 1)


def reproject_jp2_with_gdal(file_name: str, output_name: str = 'output.jp2'):
    """
    Reproject a JP2 file using GDAL.
    
    Args:
        file_name: Path to JP2 file
        output_name: Output file name
    """
    input_raster = gdal.Open(file_name)
    
    # Define target spatial reference with PROJ4 string
    proj4_string = '+proj=ortho +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs'
    
    warp = gdal.Warp(output_name,
                    input_raster,
                    dstSRS=proj4_string)
    
    # Clear the variable to close files
    warp = None

if __name__ == "__main__":
    normalization_list = [40]
    spacing_list = [0.5]
    for norm in normalization_list:
        for spacing in spacing_list:
          data = prep_jp2_for_stl_conversion('reprojected_output.jp2', normalization_range=norm)
          convert_array_to_stl(data, x_spacing=spacing, y_spacing=spacing, stl_filename=f'lunar_surface_{norm}_{spacing}_2.stl')
