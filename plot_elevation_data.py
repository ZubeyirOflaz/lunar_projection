from osgeo import gdal
from utils import file_dict
import cartopy.crs as ccrs  
import matplotlib.pyplot as plt  


def read_img_data(resolution, scaling_factor):
    # List of image files to combine
    if resolution <= 128:
        file_list = [fr'img_repo\LDEM_{resolution}.JP2']
    else:
        file_list = [fr'img_repo\{file}.JP2' for file in file_dict[resolution]]

    # Build virtual raster  
    vrt = gdal.BuildVRT('combined.vrt', file_list)  
    
    # Open virtual raster  
    dataset = gdal.Open(file_list[0])  
    metadata = dataset.GetMetadata()  
    print("Metadata: ", metadata)
    # Print geospatial metadata  
    print("Projection: ", dataset.GetProjection())  
    print("GeoTransform: ", dataset.GetGeoTransform())   
    
    # Read and resize data  
    xoff = 0  
    yoff = 0  
    xcount = dataset.RasterXSize  
    ycount = dataset.RasterYSize  
    data = dataset.ReadAsArray(xoff, yoff, xcount, ycount,    
                            buf_xsize = int(xcount*scaling_factor), buf_ysize = int(ycount*scaling_factor),
                            buf_type=gdal.GDT_Float32)    
    return data


