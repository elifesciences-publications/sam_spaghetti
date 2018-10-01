import numpy as np 
import pandas as pd
import scipy.ndimage as nd

from scipy.misc import imsave as imsave2d

from vplants.image.serial.all import imread, imsave
from vplants.image.spatial_image import SpatialImage
from vplants.image.registration import pts2transfo

from timagetk.components import SpatialImage as TissueImage
from timagetk.algorithms.trsf import BalTransformation, allocate_c_bal_matrix, apply_trsf, create_trsf

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data

from time import time as current_time
from copy import deepcopy

import logging


def sequence_signal_image_slices(sequence_name, image_dirname, save_files=True, signal_names=None, filenames=None, aligned=False, filtering=False, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=120., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel+1)    
    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=verbose, debug=debug, loglevel=loglevel+1)  
               
    if signal_names is None:
        signal_names = signal_images.keys()

    logging.info("".join(["  " for l in xrange(loglevel)])+"--> Computing 2D signal images "+str(signal_names))

    assert reference_name in signal_names

    if filenames is None:
        filenames = np.sort(signal_images[reference_name].keys())
    

    if len(filenames)>0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        filtered_signal_images = {}
        for signal_name in signal_names:
            filtered_signal_images[signal_name] = {}

        for filename in filenames:
            for signal_name in signal_names:
                signal_img = signal_images[signal_name][filename].astype(float)
                filtered_img = signal_img
                if filtering:
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Filtering : "+filename+" "+signal_name)
                    filtered_img = gaussian_filter(filtered_img,sigma=nuclei_sigma/np.array(signal_img.voxelsize),order=0)
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Filtering : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")
                filtered_signal_images[signal_name][filename] = filtered_img.astype(signal_images[signal_name][filename].dtype)    

        if aligned:
            reflections = {}
            alignment_transformations = {}    
            image_centers = {}    

            aligned_images = {}
            for signal_name in signal_names:
                aligned_images[signal_name] = {}

            for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
                file_data = signal_data[filename]
                file_data = file_data[file_data['layer']==1]

                X = file_data['aligned_x'].values
                Y = file_data['aligned_y'].values
                Z = file_data['aligned_z'].values

                img_X = file_data['center_x'].values
                img_Y = file_data['center_y'].values
                img_Z = file_data['center_z'].values

                img_points = np.transpose([img_X,img_Y,img_Z])
                aligned_points = np.transpose([X,Y,Z])

                reference_img = signal_images[reference_name][filename]
                img_center = (np.array(reference_img.shape)*np.array(reference_img.voxelsize))/2.
                img_center[2] = reference_img.shape[2]*reference_img.voxelsize[2]/8.
                image_centers[filename] = img_center
                                
                alignment_transformation = pts2transfo(img_center+microscope_orientation*aligned_points,microscope_orientation*img_points)

                rotation_angle = ((180.*np.arctan2(alignment_transformation[1,0],alignment_transformation[0,0])/np.pi)+180)%360 - 180
                reflection = np.sign(alignment_transformation[0,0]*alignment_transformation[1,1])==-1
                
                if reflection:
                    img_points = np.transpose([img_X,microscope_orientation*reference_img.shape[1]*reference_img.voxelsize[1]-img_Y,img_Z])
                    alignment_transformation = pts2transfo(img_center+microscope_orientation*aligned_points,microscope_orientation*img_points)

                reflections[filename] = reflection
                alignment_transformations[filename] = alignment_transformation
        
                alignment_trsf = create_trsf(param_str_2='-identity', trsf_type=BalTransformation.RIGID_3D, trsf_unit=BalTransformation.REAL_UNIT)
                allocate_c_bal_matrix(alignment_trsf.mat.c_struct, alignment_transformations[filename])

                for i_signal, signal_name in enumerate(signal_names):
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Aligning : "+filename+" "+signal_name)
                    
                    if signal_images[signal_name].has_key(filename):
                        if reflections[filename]:
                            reflected_image = deepcopy(filtered_signal_images[signal_name][filename])
                            reflected_image[:,:] = filtered_signal_images[signal_name][filename][:,::-1,:]
                            aligned_images[signal_name][filename] = apply_trsf(TissueImage(reflected_image.astype(reference_img.dtype),voxelsize=reference_img.voxelsize),alignment_trsf,param_str_2 = '-interpolation nearest')
                        else:
                            aligned_images[signal_name][filename] = apply_trsf(TissueImage(deepcopy(filtered_signal_images[signal_name][filename]).astype(reference_img.dtype),voxelsize=reference_img.voxelsize),alignment_trsf,param_str_2 = '-interpolation nearest')
                
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Aligning : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")


        slice_coords = {}
        image_slices = {}
        image_views = {}
        for signal_name in signal_names:
            image_slices[signal_name] = {}
            image_views[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer']==1]

            if aligned:
                X = file_data['aligned_x'].values
                Y = file_data['aligned_y'].values
                Z = file_data['aligned_z'].values
            else:
                X = file_data['center_x'].values
                Y = file_data['center_y'].values
                Z = file_data['center_z'].values

            reference_img = signal_images[reference_name][filename]
            size = np.array(reference_img.shape)
            voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

            if resolution is None:
                resolution = np.abs(voxelsize)[0]

            if aligned:
                xx,yy = np.meshgrid(np.linspace(-r_max,r_max,(2*r_max)/resolution+1),np.linspace(-r_max,r_max,(2*r_max)/resolution+1))
            else:
                xx,yy = np.meshgrid(np.linspace(0,((size-1)*voxelsize)[0],((size-1)*np.abs(voxelsize))[0]/resolution+1),np.linspace(0,((size-1)*voxelsize)[1],((size-1)*np.abs(voxelsize))[0]/resolution+1))
            # extent = xx.max(),xx.min(),yy.min(),yy.max()
            extent = xx.min(),xx.max(),yy.max(),yy.min()

            if projection_type == "L1_slice":
                start_time = current_time()
                logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing Z-map : "+filename)

                zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),Z)
                if aligned:
                    img_center = image_centers[filename]
                    coords = (img_center+microscope_orientation*np.transpose([xx,yy,zz],(1,2,0)))/np.array(reference_img.voxelsize)
                else:
                    coords = (microscope_orientation*np.transpose([xx,yy,zz],(1,2,0)))/np.array(reference_img.voxelsize)

                for k in xrange(3):
                    coords[:,:,k] = np.maximum(np.minimum(coords[:,:,k],reference_img.shape[k]-1),0)
                coords[np.isnan(coords)]=0
                coords = coords.astype(int)
                coords = tuple(np.transpose(np.concatenate(coords)))

                slice_coords[filename] = coords

                logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Computing Z-map : "+filename+"   ["+str(current_time() - start_time)+" s]")

                for i_signal, signal_name in enumerate(signal_names):
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Slicing : "+filename+" "+signal_name)

                    if aligned:
                        image_slices[signal_name][filename] = aligned_images[signal_name][filename][slice_coords[filename]].reshape(xx.shape)
                    else:
                        image_slices[signal_name][filename] = filtered_signal_images[signal_name][filename][slice_coords[filename]].reshape(xx.shape)

                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Slicing : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")

            elif projection_type == "max_intensity":
                if aligned:
                    img_center = image_centers[filename]
                    coords = img_center + (microscope_orientation*np.transpose([xx,yy,np.zeros_like(xx)],(1,2,0)))/np.array(reference_img.voxelsize)
                else:
                    coords = (microscope_orientation*np.transpose([xx,yy,np.zeros_like(xx)],(1,2,0)))/np.array(reference_img.voxelsize)
                
                for k in xrange(3):
                    coords[:,:,k] = np.maximum(np.minimum(coords[:,:,k],reference_img.shape[k]-1),0)
                coords[np.isnan(coords)]=0
                coords = coords.astype(int)
                coords = tuple(np.transpose(np.concatenate(coords)))

                for i_signal, signal_name in enumerate(signal_names):
                    start_time = current_time()
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Projecting : "+filename+" "+signal_name)
                    depth = (np.arange(size[2])/float(size[2]-1))[np.newaxis,np.newaxis]*np.ones_like(xx)[:,:,np.newaxis]
                    depth = np.ones_like(depth)
                    
                    if aligned:
                        max_projection = (depth*(aligned_images[signal_name][filename][coords[:2]].reshape(xx.shape + (reference_img.shape[2],)))).max(axis=2)
                        max_projection = np.transpose(max_projection)[::-1,::-1]
                    else:
                        max_projection = (depth*(filtered_signal_images[signal_name][filename][coords[:2]].reshape(xx.shape + (reference_img.shape[2],)))).max(axis=2)
                    
                    image_slices[signal_name][filename] = max_projection
                    logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Projecting : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")

            if save_files:
                logging.info("".join(["  " for l in xrange(loglevel)])+"--> Saving 2D signal images : "+filename+" "+str(signal_names))
                for i_signal, signal_name in enumerate(signal_names):
                    image_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+("_aligned_" if aligned else "_")+projection_type+"_"+signal_name+"_projection.tif"
                    imsave2d(image_filename,image_slices[signal_name][filename])
        return image_slices    

