from typing import NamedTuple

class Config(NamedTuple):

    input_resolution = 16 # The resolution of the image data to be converted to STL. The resolution can be 4, 16, 64, 128, 256, or 512 pixels/degree.
                          # Please note that the higher resolutions create minor artifacts in the output STL file (cause to be investigated).

    apply_default_transformation = True # Whether to apply the default transformation to the image data. Default transformation creates an orthographic projection that centers 
                                        # the image in Sea of Tranquility in latitude and lunar equator in longitude. 
    custom_transformation_proj4 = None # Custom proj4 string for the transformation. The apply_default_transformation parameter should be set to False for this to take effect.
    

    simplify_mesh : bool = True
    target_num_faces : int = 10000
    
    
    
    # Phsical properties of the output STL file(s)
    num_stl_divisions : int = 5 # Number of divisions in each axis for the STL file. Creates nXn divisions.
    height_range : int = 60 # Height range of the STL file in millimeters.
    zoom_factor : float = 1.0 # Zoom factor for the image data
