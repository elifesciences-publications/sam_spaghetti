import numpy as np
from scipy import ndimage as nd
import pandas as pd

from vplants.image.serial.all import imread, imsave
from vplants.image.spatial_image import SpatialImage

from vplants.container import array_dict

from timagetk.algorithms.blockmatching import blockmatching
#from timagetk.plugins.registration import registration
from timagetk.components import SpatialImage as TissueImage

import os
import logging
from time import time as current_time

max_time = 100

def sequence_rigid_vectorfield_registration(sequence_name, save_files=True, image_dirname=None, reference_name='TagBFP', microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if save_files and (image_dirname is None):
        logging.error("No output directory provided, results will NOT be saved!")
        save_files = False


    filenames = []
    reference_images = {}
    
    for time in xrange(max_time):
        filename = sequence_name + "_t" + str(time).zfill(2)
        
        reference_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+reference_name+".inr.gz"

        if os.path.exists(reference_image_file):
            reference_img = imread(reference_image_file)
            reference_images[filename] = TissueImage(reference_img,voxelsize=reference_img.voxelsize)
            filenames += [filename]


    transformed_images = {}
    transformed_images[filenames[0]] = reference_images[filenames[0]]

    rigid_transformations = {}
    rigid_transformations[filenames[0]] = np.diag(np.ones(4,float))

    vectorfield_transformations = {}
    vectorfield_transformations[filenames[0]] = np.zeros(reference_images[filenames[0]].shape+(3,))

    
    for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):

        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Computing transformations "+reference_filename+" <--> "+floating_filename)
        # print "-------------------------------"
        # print reference_filename," --> ",floating_filename
        # print "-------------------------------"
        
        reference_image_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_"+reference_name+".inr.gz"
        floating_image_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_"+reference_name+".inr.gz"
        
        reference_img = reference_images[reference_filename] if i_file==0 else transformed_images[reference_filename] 
        floating_img = reference_images[floating_filename]
        
        size = np.array(reference_img.shape)
        resolution = microscope_orientation*np.array(reference_img.voxelsize)
        
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing rigid transformation "+reference_filename[-3:]+" <-- "+floating_filename[-3:])
        transform, transform_img = blockmatching(floating_img, reference_img, param_str_1 ='-trsf-type rigid')
        #transform, transform_img = registration(floating_img, reference_img, transformation_type='rigid')
        
        transformed_images[floating_filename] = floating_img

        rigid_matrix = transform.mat.to_np_array()
        rigid_transformations[floating_filename] = rigid_matrix

        invert_rigid_matrix = np.linalg.inv(rigid_matrix)

        logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Computing rigid transformation "+reference_filename[-3:]+" <-- "+floating_filename[-3:]+" ["+str(current_time()-start_time)+" s]")

        if save_files:
            transformed_image_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+filenames[0][-3:]+"_"+reference_name+".inr.gz"
            imsave(transformed_image_file,SpatialImage(transform_img,voxelsize=reference_img.voxelsize))
        
            reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(reference_to_floating_transform_file, rigid_matrix, delimiter=";")
        
            floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(floating_to_reference_transform_file, invert_rigid_matrix, delimiter=";")
        

        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing vectorfield transformation "+reference_filename[-3:]+" <-- "+floating_filename[-3:])
        blockmatching_transform, blockmatching_img = blockmatching(transform_img, reference_img,  param_str_1 ='-trsf-type vectorfield')
        #blockmatching_transform, blockmatching_img = registration(transform_img, reference_img,  transformation_type='vectorfield')
        vector_field = np.transpose([blockmatching_transform.vx.to_spatial_image(),blockmatching_transform.vy.to_spatial_image(),blockmatching_transform.vz.to_spatial_image()],(1,2,3,0))
        logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Computing vectorfield transformation "+reference_filename[-3:]+" <-- "+floating_filename[-3:]+" ["+str(current_time()-start_time)+" s]")
        
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing vectorfield transformation "+reference_filename[-3:]+" --> "+floating_filename[-3:])
        invert_blockmatching_transform, invert_blockmatching_img = blockmatching(reference_img, transform_img, param_str_1 ='-trsf-type vectorfield')
        #invert_blockmatching_transform, invert_blockmatching_img = registration(reference_img, transform_img,  transformation_type='vectorfield')
        invert_vector_field = np.transpose([invert_blockmatching_transform.vx.to_spatial_image(),invert_blockmatching_transform.vy.to_spatial_image(),invert_blockmatching_transform.vz.to_spatial_image()],(1,2,3,0))
        logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Computing vectorfield transformation "+reference_filename[-3:]+" --> "+floating_filename[-3:]+" ["+str(current_time()-start_time)+" s]")
        
        vectorfield_transformations[floating_filename] = invert_vector_field

        if save_files:
            start_time = current_time()
            logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving vectorfield transformations")
            vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
            imsave(vector_field_file,SpatialImage(vector_field,voxelsize=reference_img.voxelsize))
            
            invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
            imsave(invert_vector_field_file,SpatialImage(invert_vector_field,voxelsize=reference_img.voxelsize))
            logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Saving vectorfield transformations ["+str(current_time()-start_time)+" s]")

    result = (transformed_images, rigid_transformations, vectorfield_transformations)
    return result
            