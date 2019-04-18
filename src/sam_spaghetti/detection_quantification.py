import numpy as np
import pandas as pd

from vplants.tissue_nukem_3d.microscopy_images import imread
from vplants.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_lsm_image, read_tiff_image

from vplants.tissue_nukem_3d.utils.matplotlib_tools import view_image_projection

import vplants.tissue_nukem_3d.nuclei_image_topomesh
reload(vplants.tissue_nukem_3d.nuclei_image_topomesh)
from vplants.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh

from timagetk.components import SpatialImage
from timagetk.io import imsave

from vplants.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh
from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

import sys
import os
import logging
from time import time as current_time

import matplotlib.pyplot as plt

# channel_compute_ratios = ['DIIV']
channel_compute_ratios = []

def detect_and_quantify(img_dict, reference_name='TagBFP', signal_names=None, compute_ratios=None, save_files=True, image_dirname=None, nomenclature_name=None, microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_name = nomenclature_name[:-4]
    
    if signal_names is None:
        signal_names = img_dict.keys()

    if compute_ratios is None:
        compute_ratios = [signal_name in channel_compute_ratios for signal_name in signal_names]

    logging.info("".join(["  " for l in xrange(loglevel)])+"--> Detecting and quantifying")
    # topomesh = nuclei_image_topomesh(nomenclature_names[filename],dirname=image_dirname,reference_name=reference_name,signal_names=signal_names,compute_ratios=compute_ratios,redetect=redetect, recompute=recompute,subsampling=4)
    topomesh, surface_topomesh = nuclei_image_topomesh(img_dict,reference_name=reference_name,signal_names=signal_names,compute_ratios=compute_ratios, microscope_orientation=microscope_orientation, radius_range=(0.8,1.4), threshold=3000, surface_mode='image', return_surface=True)

    if save_files:
        topomesh_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_nuclei_signal_curvature_topomesh.ply"
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving detected nuclei")
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer','mean_curvature','gaussian_curvature']),(1,[]),(2,[]),(3,[])]),color_faces=False) 

        surface_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_surface_topomesh.ply"
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving surface mesh")
        save_ply_property_topomesh(surface_topomesh,surface_file,properties_to_save=dict([(0,['mean_curvature','gaussian_curvature']),(1,[]),(2,[]),(3,[])])) 

    df = topomesh_to_dataframe(topomesh,0)
    if save_files:
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving nuclei data")
        if ('DIIV' in df.columns)&('TagBFP' in df.columns):
            df['qDII'] = df['DIIV'].values/df['TagBFP'].values
        if ('RGAV' in df.columns)&('TagBFP' in df.columns):
            df['qRGA'] = df['RGAV'].values/df['TagBFP'].values

        df['label'] = df.index.values
        df.to_csv(image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_signal_data.csv",index=False)  

    results = (df, topomesh, img_dict, surface_topomesh)

    return results
                      

