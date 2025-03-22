import numpy as np
from PIL import Image
import PIL  
from stl import mesh 
import pymeshlab as ml
import os
import rasterio
import scipy.ndimage
from config_v4 import Config
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import numpy as np  
from scipy.ndimage import generic_filter 
  
  


# Increase the maximum image size PIL can open
PIL.Image.MAX_IMAGE_PIXELS = 955360000

def divide_matrix(matrix, num_of_divisions : int = 5, width_of_division: int = 20):
    """Given a 2d square matrix, divide it into num_of_divisions x num_of_divisions of squares equal in size submatrices by adding zeros between each division with the width of width_of_division
    Input:
    matrix : 2d numpy array
    num_of_divisions : int : number of divisions in each axis
    width_of_division : int : width of the division
    Output:
    divided_matrix : 2d numpy array : divided matrix
    """
    # Get the shape of the matrix
    shape = matrix.shape
    # Get the number of rows and columns
    rows, _ = shape
    # Get the division locations
    div_index = rows // num_of_divisions
    # Add zeros between each division
    for i in range(1, num_of_divisions):
        matrix = np.insert(matrix, div_index*i + (width_of_division * (i-1)), np.zeros((width_of_division, matrix.shape[1])), axis=0)
    for i in range(1, num_of_divisions):
        matrix = np.insert(matrix, div_index*i + (width_of_division * (i-1)), np.zeros((width_of_division, matrix.shape[0])), axis=1)

    return matrix

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

def prep_jp2_for_stl_conversion(jp2_file: str):
    cfg = Config()

    normalization_range = cfg.height_range
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
        data[data == np.nan] = 0

        
        data = data * normalization_rate
        data = divide_matrix(data, num_of_divisions=cfg.num_stl_divisions, width_of_division=20)
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
    stl_mesh = mesh.Mesh(np.zeros(coords.shape[0]*coords.shape[1]*2, dtype=mesh.Mesh.dtype))    
  
    # Now you can use this matrix with your convert_array_to_stl() function    
    print("Creating mesh from array...")  
    
    for i in range(x.shape[0]-1):        
        for j in range(x.shape[1]-1):  
            # Check if current cell and its neighbors are zero  
            if (matrix[i, j] == 0 and  
                matrix[i+1, j] == 0 and  
                matrix[i, j+1] == 0 and  
                matrix[i+1, j+1] == 0):  
                continue  
    
            # create lower triangle        
            stl_mesh.vectors[i*2*x.shape[1]+j*2] = [coords[i, j], coords[i+1, j], coords[i+1, j+1]]      
            # create upper triangle      
            stl_mesh.vectors[i*2*x.shape[1]+j*2+1] = [coords[i, j], coords[i, j+1], coords[i+1, j+1]]      

    
    # Write the mesh to file    
    stl_mesh.save(stl_filename)  


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


 
  
def numpy2stl(A, stl_filename, solid=True):  
    """  
    A: 2D numpy array (picture should be of dtype=ubyte)  
    scale: scale pixel intensity to physical dimensions for the stl mesh  
    mode: 0: Rectangular pixels 1: Hexagonal pixels  
    solid: If True: the function will add a base to the object to make it a 2-manifold  
    """  
    nx, ny = A.shape  
    vertices = np.zeros((nx*ny, 3))  
  
    # Create a grid of indices  
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))  
  
    # Set the vertices  
    vertices[:,0] = x.flatten()  
    vertices[:,1] = y.flatten()  
    vertices[:,2] = A.flatten()  
  
    # Define the faces  
    faces = []  
    for i in range(nx-1):  
        for j in range(ny-1):  
            if not (A[i, j] == A[i, j+1] == A[i+1, j] == A[i+1, j + 1] == 0):  
                # create face 1  
                faces.append([i*ny+j, i*ny+j+1, (i+1)*ny+j])  
  
                # create face 2  
                faces.append([(i+1)*ny+j, i*ny+j+1, (i+1)*ny+j+1])  
    faces = np.array(faces)  
  
    # Create the mesh  
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))  
  
    for i, f in enumerate(faces):  
        for j in range(3):  
            stl_mesh.vectors[i][j] = vertices[f[j],:]  
  
    if solid:  
        # Create the bottom face of the 3D object  
        zmin = vertices[:, 2].min()  
        bottom_face = []  
  
        for i in range(nx-1):  
            for j in range(ny-1):  
                if not (A[i, j] == A[i, j+1] == A[i+1, j] == A[i+1, j + 1] == 0):  
                    bottom_face.append([i*ny+j, i*ny+j+1, (i+1)*ny+j])  
                    bottom_face.append([(i+1)*ny+j, i*ny+j+1, (i+1)*ny+j+1])  
  
        bottom_face = np.array(bottom_face)  
        new_mesh = mesh.Mesh(np.zeros(bottom_face.shape[0], dtype=mesh.Mesh.dtype))  
          
        for i, f in enumerate(bottom_face):  
            for j in range(3):  
                new_mesh.vectors[i][j] = vertices[f[j],:]  
                new_mesh.vectors[i][j][2] = zmin  # Set z-coordinates to zmin  
  
        stl_mesh = mesh.Mesh(np.concatenate([stl_mesh.data, new_mesh.data]))  
  
    # Write the mesh to file  
    stl_mesh.save(stl_filename)  
   

  


def mesh_save_as_stl(matrix, output_file):
    array3d = np.dstack((matrix, np.zeros(matrix.shape)))
    stl_mesh = mesh.Mesh(array3d, remove_empty_areas=False)  

    # Write the mesh to file "output.stl"  
    stl_mesh.save(output_file)    

  
if __name__ == "__main__":
    cfg = Config()
    data = prep_jp2_for_stl_conversion('reprojected_output.jp2')
    data = np.pad(data, pad_width=2, mode='constant', constant_values=0.0)  

    for x in range(1, cfg.num_stl_divisions + 1):
        for y in range(1, cfg.num_stl_divisions + 1):
            div_range = data.shape[0] // cfg.num_stl_divisions
            x_start = div_range * (x-1)
            if x == cfg.num_stl_divisions:
                x_end = data.shape[0]
            else: 
                x_end = div_range * x
            if y == cfg.num_stl_divisions:
                y_end = data.shape[1]
            else:
                y_end = div_range * y
            y_start = div_range * (y-1)
            numpy2stl(data[x_start:x_end, y_start:y_end], f'stl_output/lunar_surface_{x}_{y}.stl')

            #convert_array_to_stl(data[x_start:x_end, y_start:y_end], 1, 1, f'stl_output/lunar_surface_{x}_{y}.stl')        
    