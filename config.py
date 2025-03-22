from typing import NamedTuple

class Config(NamedTuple):

    apply_default_transformation = True
    custom_transformation_proj4 = None
    

    simplify_mesh : bool = True
    target_num_faces : int = 10000
    
    
    
    # Phsical properties of the output STL file(s)
    num_stl_divisions : int = 5 # Number of divisions in each axis for the STL file. Creates nXn divisions.
    total_diameter : int = 1000 # Total diameter of the STL file in millimeters.
    height : int = 100 # Height range of the STL file in millimeters.
    zoom_factor : float = 0.5