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


def plot_elevation_data(resolution,
                        scaling_factor: float = 0.5,
                        show_plot: bool = False, 
                        interpolation: str = 'antialiased',
                        target_proj: ccrs.Projection = ccrs.Geostationary(30)):
    """
    This function is used to plot the elevation data on a map using the specified projection.
    Input:
    resolution: int - The resolution of the image data to be plotted. The resolution can be 4, 16, 64, 128, 256, or 512 pixels/degree.
    show_plot: bool - If True, the plot will be displayed and then saved as PNG file. If False, the plot will only be saved as a PNG file.
    target_proj: ccrs.Projection - The projection to be used for the plot. Default is Orthographic projection.
    """
    # Read the image data
    data = read_img_data(resolution, scaling_factor)

    # Original projection of the data  
    source_proj = ccrs.LambertCylindrical()  

    # Create a new figure  
    fig = plt.figure(figsize=(20, 20))  
    
    # Create a GeoAxes in the tile's projection  
    ax = plt.axes(projection=target_proj)  
    
    # Add the image to the map  
    img = ax.imshow(data, origin='upper', transform=source_proj, extent=[-180, 180, -90, 90], cmap='gray', interpolation=interpolation)

    if show_plot:
        plt.show()
    else:
        # Save the figure  
        plt.savefig(f'output_{interpolation}.png', transparent=True)
    plt.close(fig)    
    

if __name__ == "__main__":
    plot_elevation_data(128, 0.5, show_plot=False)

