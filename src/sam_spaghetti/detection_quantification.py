import numpy as np
import pandas as pd

from vplants.tissue_nukem_3d.microscopy_images import imread
from vplants.tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_tiff_image

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

def detect_from_czi(czi_file, no_organ_file=None, reference_name='TagBFP', channel_names=None, save_files=True, save_images=True, image_dirname=None, nomenclature_name=None, microscope_orientation=-1, verbose=True, debug=False,loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    filename = os.path.split(czi_file)[1]

    if image_dirname is None:
        image_dirname = os.path.split(czi_file)[0]+"/../nuclei_images/"

    if channel_names is None:
        if "qDII-CLV3-DR5-PIN1-PI" in filename:
            channel_names = ['DIIV','DR5','PIN1','PI','TagBFP','CLV3']
        elif "qDII-CLV3-DR5-PI" in filename:
            channel_names = ['DIIV','PI','DR5','TagBFP','CLV3']
        elif "qDII-CLV3-DR5" in filename:
            channel_names = ['DIIV','DR5','TagBFP','CLV3']
        elif "qDII-CLV-pAHP6" in filename:
            channel_names = ['DIIV','AHP6','TagBFP','CLV3']
        elif "qDII-CLV3-PIN1-PI" in filename:
            channel_names = ['DIIV','PIN1','PI','TagBFP','CLV3']
        else:
            channel_names = ['DIIV','TagBFP','CLV3']

    compute_ratios = [True if signal_name in channel_compute_ratios else False for signal_name in channel_names]
            
    start_time = current_time()
    logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading microscopy image")
    img_dict = read_czi_image(czi_file,channel_names=channel_names)
    logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Loading microscopy image ["+str(current_time()-start_time)+" s]")

    if nomenclature_name is None:
        nomenclature_name = filename
    sequence_name = nomenclature_name[:-4]

    reference_img = img_dict[reference_name]
            
    if no_organ_file is None:
        no_organ_file = os.path.split(czi_file)[0]+"/../TIF-No-organs/"+filename[:-4]+"-No-organs.tif"
            
    if os.path.exists(no_organ_file):
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Loading cropped image")
        no_organ_dict = read_tiff_image(no_organ_file,channel_names=channel_names)
        voxelsize = img_dict[reference_name].voxelsize
        for channel in channel_names:
            no_organ_dict[channel] = SpatialImage(no_organ_dict[channel],voxelsize=voxelsize)
    else:
        no_organ_dict = {}

    n_channels = len(img_dict)
            
    if save_images:
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Saving image channels")
    for i_channel, channel_name in enumerate(channel_names):

        if save_files:
            raw_img_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_"+channel_name+"_raw.inr.gz"        
            img_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_"+channel_name+".inr.gz"
            no_organ_img_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/"+nomenclature_name+"_"+channel_name+"_no_organ.inr.gz"
        
        # if channel_name == reference_name:
        if (channel_name != 'DIIV') or (not 'PIN1' in channel_names):
        # if True:
            if channel_name in no_organ_dict:
                no_organ_img = no_organ_dict[channel_name]

                if save_images:
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving raw "+channel_name+" image")
                    imsave(raw_img_file,img_dict[channel_name])
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Saving raw "+channel_name+" image ["+str(current_time()-start_time)+" s]")
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving cropped "+channel_name+" image")
                    imsave(img_file,no_organ_img)
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Saving cropped "+channel_name+" image ["+str(current_time()-start_time)+" s]")
                
                img_dict[channel_name] = no_organ_img
            else:
                if save_images:
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving "+channel_name+" image")
                    imsave(img_file,img_dict[channel_name])
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Saving "+channel_name+" image ["+str(current_time()-start_time)+" s]")
        elif channel_name == 'DIIV':
            voxelsize = img_dict[channel_name].voxelsize
                
            if channel_name in no_organ_dict:
                no_organ_img = no_organ_dict[channel_name]
                substracted_img = SpatialImage(np.maximum(0,(no_organ_dict['DIIV'].astype(np.int32) - no_organ_dict['PIN1'].astype(np.int32))).astype(np.uint16),voxelsize=voxelsize)
            
                if save_images:
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving raw "+channel_name+" image")
                    imsave(raw_img_file,img_dict[channel_name])
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving cropped "+channel_name+" image")
                    imsave(no_organ_img_file,no_organ_img)
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving cropped substracted "+channel_name+" image")
                    imsave(img_file,substracted_img)
            else:
                substracted_img = SpatialImage(np.maximum(0,(img_dict['DIIV'].get_array().astype(np.int32) - img_dict['PIN1'].get_array().astype(np.int32))).astype(np.uint16),voxelsize=voxelsize)
            
                if save_images:
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving raw "+channel_name+" image")
                    imsave(raw_img_file,img_dict[channel_name])
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving substracted "+channel_name+" image")
                    imsave(img_file,substracted_img)

            img_dict[channel_name] = substracted_img          
        else:
            if save_images:
                imsave(img_file,img_dict[channel_name])

    detect_and_quantify(img_dict,reference_name=reference_name,signal_names=channel_names,compute_ratios=compute_ratios,save_files=save_files,image_dirname=image_dirname,nomenclature_name=nomenclature_name,microscope_orientation=microscope_orientation,verbose=verbose,debug=debug,loglevel=loglevel)
            

def detect_and_quantify(img_dict, reference_name='TagBFP', signal_names=None, compute_ratios=None, save_files=True, image_dirname=None, nomenclature_name=None, microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_name = nomenclature_name[:-4]
    
    if signal_names is None:
        signal_names = img_dict.keys()

    if compute_ratios is None:
        compute_ratios = [False for signal_name in signal_names]

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
                      

