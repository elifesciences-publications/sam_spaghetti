import numpy as np
from scipy import ndimage as nd
import pandas as pd
import matplotlib.pyplot as plt

from vplants.image.serial.all import imread, imsave
from vplants.image.spatial_image import SpatialImage

from vplants.cellcomplex.property_topomesh.utils.delaunay_tools import delaunay_triangulation
from vplants.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh, triangle_topomesh
from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces

from vplants.cellcomplex.property_topomesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components, cut_surface_topomesh, clean_topomesh
from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

from vplants.tissue_nukem_3d.epidermal_maps import nuclei_density_function, compute_local_2d_signal

from vplants.container import array_dict

from sam_spaghetti.signal_image_plot import load_sequence_signal_images, load_sequence_signal_data
from sam_spaghetti.utils.signal_luts import quantified_signals

from copy import deepcopy
from time import time as current_time

import os
import logging

max_time = 99


def load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    rigid_transformations = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading rigid transforms "+reference_filename+" <--> "+floating_filename)
        reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
        floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            
        rigid_transformations[(reference_filename,floating_filename)] = np.loadtxt(reference_to_floating_transform_file,delimiter=";")
        rigid_transformations[(floating_filename,reference_filename)] = np.loadtxt(floating_to_reference_transform_file,delimiter=";")    

    return rigid_transformations


def load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    vectorfield_transformations = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" <-- "+floating_filename)
        vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(floating_filename,reference_filename)] = imread(vector_field_file)
        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" <-- "+floating_filename+" ["+str(current_time()-start_time)+" s]")

        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" --> "+floating_filename)
        invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(reference_filename,floating_filename)] = imread(invert_vector_field_file)
        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" --> "+floating_filename+" ["+str(current_time()-start_time)+" s]")

    return vectorfield_transformations


def spherical_2d_projection(positions, center_offset=3):
    center = positions.values().mean(axis=0) 
    center[2] -= center_offset*(positions.values()-positions.values().mean(axis=0))[:,2].max()
    
    point_vectors = positions.values() - center
    point_r = np.linalg.norm(point_vectors,axis=1)
    point_rx = np.linalg.norm(point_vectors[:,np.array([0,2])],axis=1)
    point_ry = np.linalg.norm(point_vectors[:,np.array([1,2])],axis=1)
    
    point_phi = np.sign(point_vectors[:,0])*np.arccos(point_vectors[:,2]/point_rx)
    point_psi = np.sign(point_vectors[:,1])*np.arccos(point_vectors[:,2]/point_ry)
    
    spherical_positions = deepcopy(positions)
    for i,c in enumerate(positions.keys()):
        spherical_positions[c][0] = point_phi[i]
        spherical_positions[c][1] = point_psi[i]
        spherical_positions[c][2] = 0.

    return spherical_positions
            

def compute_surfacic_growth(sequence_name, image_dirname, save_files=True, maximal_length=15., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_vectorfield_transforms = load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    filenames = np.sort(sequence_data.keys())

    previous_transform = np.diag(np.ones(4))
        
    for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):
        
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Computing next surfacic growth "+reference_filename+" --> "+floating_filename)      

        rigid_matrix = sequence_rigid_transforms[(reference_filename,floating_filename)]
        invert_rigid_matrix = sequence_rigid_transforms[(floating_filename,reference_filename)]
        vector_field = sequence_vectorfield_transforms[(floating_filename,reference_filename)]
        invert_vector_field = sequence_vectorfield_transforms[(reference_filename,floating_filename)]
     
        size = np.array(vector_field.shape)[:3]
        resolution = microscope_orientation*np.array(vector_field.voxelsize)
            
        reference_data = sequence_data[reference_filename]    
        floating_data = sequence_data[floating_filename]    

        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing triangulation on "+reference_filename[-3:])      

        X = reference_data['center_x']
        Y = reference_data['center_y']
        Z = reference_data['center_z']
        
        if i_file == 0:
            reference_data['registered_x'] = X
            reference_data['registered_y'] = Y
            reference_data['registered_z'] = Z
    
        reference_points = np.transpose([X,Y,Z])

        cell_barycenters = array_dict(np.transpose([X,Y,Z]),reference_data.index.values)
        cell_flat_barycenters = spherical_2d_projection(cell_barycenters)
            
        triangles = np.array(cell_barycenters.keys())[delaunay_triangulation(np.array([cell_flat_barycenters[c] for c in cell_barycenters.keys()]))]
        reference_triangulation_topomesh = triangle_topomesh(triangles, cell_barycenters)
        
        compute_topomesh_property(reference_triangulation_topomesh,'length',1)
        compute_topomesh_property(reference_triangulation_topomesh,'triangles',1)
        
        boundary_edges = np.array(map(len,reference_triangulation_topomesh.wisp_property('triangles',1).values()))==1
        distant_edges = reference_triangulation_topomesh.wisp_property('length',1).values() > maximal_length
        edges_to_remove = np.array(list(reference_triangulation_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
        while len(edges_to_remove) > 0:
            triangles_to_remove = np.concatenate(reference_triangulation_topomesh.wisp_property('triangles',1).values(edges_to_remove))
            for t in triangles_to_remove:
                reference_triangulation_topomesh.remove_wisp(2,t)
            
            clean_topomesh(reference_triangulation_topomesh)
            
            compute_topomesh_property(reference_triangulation_topomesh,'triangles',1)
        
            boundary_edges = np.array(map(len,reference_triangulation_topomesh.wisp_property('triangles',1).values()))==1
            distant_edges = reference_triangulation_topomesh.wisp_property('length',1).values() > maximal_length
            edges_to_remove = np.array(list(reference_triangulation_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Transforming triangulation to "+floating_filename[-3:])     

        invert_registered_triangulation_topomesh = deepcopy(reference_triangulation_topomesh)
        
        image_coords = tuple(np.transpose(np.minimum(size-1,np.maximum(0,(reference_points/resolution).astype(int)))))
        point_displacement = invert_vector_field[image_coords]
        
        for i_dim,dim in enumerate(['x','y','z']):
            if "next_motion_"+dim in reference_data.columns:
                del reference_data['next_motion_'+dim]
            reference_data['next_motion_'+dim] = point_displacement[:,i_dim]
            if i_file == 0:
                if "previous_motion_"+dim in reference_data.columns:
                    del reference_data['previous_motion_'+dim]
                reference_data['previous_motion_'+dim]=0
        
        invert_registered_points = reference_points + point_displacement
        invert_registered_barycenters = array_dict(invert_registered_points,reference_data.index.values)

        invert_registered_triangulation_topomesh.update_wisp_property('barycenter',0,invert_registered_barycenters)
        
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing triangle area ratio "+floating_filename[-3:]+"/"+reference_filename[-3:])

        compute_topomesh_property(reference_triangulation_topomesh,'area',2)

        compute_topomesh_property(invert_registered_triangulation_topomesh,'length',1)
        compute_topomesh_property(invert_registered_triangulation_topomesh,'area',2)

        area_growth = invert_registered_triangulation_topomesh.wisp_property('area',2).values() - reference_triangulation_topomesh.wisp_property('area',2).values() 
        relative_area_growth = invert_registered_triangulation_topomesh.wisp_property('area',2).values()/reference_triangulation_topomesh.wisp_property('area',2).values() 

        reference_triangulation_topomesh.update_wisp_property('relative_surfacic_growth',2,array_dict(relative_area_growth,reference_triangulation_topomesh.wisp_property('area',2).keys()))
        compute_topomesh_vertex_property_from_faces(reference_triangulation_topomesh,'relative_surfacic_growth',neighborhood=3,adjacency_sigma=1.2)

        relative_future_surfacic_growth = reference_triangulation_topomesh.wisp_property('relative_surfacic_growth',0).values(reference_data.index.values)
        
        reference_data['next_relative_surfacic_growth'] = relative_future_surfacic_growth
        if i_file == 0:
            reference_data['previous_relative_surfacic_growth'] = np.nan

        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Computing next surfacic growth "+reference_filename+" --> "+floating_filename+" ["+str(current_time()-start_time)+" s]")
        
        
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Computing previous surfacic growth "+reference_filename+" <-- "+floating_filename)

        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing triangulation on "+floating_filename[-3:])    

        X = floating_data['center_x']
        Y = floating_data['center_y']
        Z = floating_data['center_z']
        
        cell_barycenters = array_dict(np.transpose([X,Y,Z]),floating_data.index.values)

        homogeneous_points = np.concatenate([microscope_orientation*np.transpose([X,Y,Z]), np.ones((len(X),1))],axis=1)
        registered_points = np.einsum("...ij,...j->...i",invert_rigid_matrix, homogeneous_points)
        registered_points = microscope_orientation*registered_points[:,:3]
        
        previous_transform = np.dot(invert_rigid_matrix,previous_transform)
        sequence_registered_points = np.einsum("...ij,...j->...i",previous_transform,homogeneous_points)
        sequence_registered_points = microscope_orientation*sequence_registered_points[:,:3]

        floating_data['registered_x']=registered_points[:,0]
        floating_data['registered_y']=registered_points[:,1]
        floating_data['registered_z']=registered_points[:,2]
        
        registered_barycenters = array_dict(registered_points,floating_data.index.values)
            
        cell_flat_barycenters = spherical_2d_projection(cell_barycenters)
        
        triangles = np.array(cell_barycenters.keys())[delaunay_triangulation(np.array([cell_flat_barycenters[c] for c in cell_barycenters.keys()]))]
        transform_triangulation_topomesh = triangle_topomesh(triangles, registered_barycenters)

        compute_topomesh_property(transform_triangulation_topomesh,'length',1)
        compute_topomesh_property(transform_triangulation_topomesh,'triangles',1)
        
        boundary_edges = np.array(map(len,transform_triangulation_topomesh.wisp_property('triangles',1).values()))==1
        distant_edges = transform_triangulation_topomesh.wisp_property('length',1).values() > maximal_length
        edges_to_remove = np.array(list(transform_triangulation_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
        while len(edges_to_remove) > 0:
            triangles_to_remove = np.concatenate(transform_triangulation_topomesh.wisp_property('triangles',1).values(edges_to_remove))
            for t in triangles_to_remove:
                transform_triangulation_topomesh.remove_wisp(2,t)
            
            clean_topomesh(transform_triangulation_topomesh)
            
            compute_topomesh_property(transform_triangulation_topomesh,'triangles',1)
        
            boundary_edges = np.array(map(len,transform_triangulation_topomesh.wisp_property('triangles',1).values()))==1
            distant_edges = transform_triangulation_topomesh.wisp_property('length',1).values() > maximal_length
            edges_to_remove = np.array(list(transform_triangulation_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Transforming triangulation to "+reference_filename[-3:])   
        
        registered_triangulation_topomesh = deepcopy(transform_triangulation_topomesh)

        image_coords = tuple(np.transpose(np.minimum(size-1,np.maximum(0,(registered_points/resolution).astype(int)))))
        point_displacement = vector_field[image_coords]
        
        for i_dim,dim in enumerate(['x','y','z']):
            if "previous_motion_"+dim in floating_data.columns:
                del floating_data['previous_motion_'+dim]
            floating_data['previous_motion_'+dim] = point_displacement[:,i_dim]
            if i_file == len(filenames)-2:
                if 'next_motion_'+dim in floating_data.columns:
                    del floating_data['next_motion_'+dim]
                floating_data['next_motion_'+dim]=0
                
        floating_data['previous_motion_x'] = point_displacement[:,0]
        floating_data['previous_motion_y'] = point_displacement[:,1]
        floating_data['previous_motion_z'] = point_displacement[:,2]
        
        if i_file == len(filenames)-2:
            floating_data['next_motion_x'] = 0
            floating_data['next_motion_y'] = 0
            floating_data['next_motion_z'] = 0
        
        registered_points = registered_points + point_displacement
        registered_barycenters = array_dict(registered_points,floating_data.index.values)
        
        registered_triangulation_topomesh.update_wisp_property('barycenter',0,registered_barycenters)
        
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing triangle area ratio "+floating_filename[-3:]+"/"+reference_filename[-3:])

        compute_topomesh_property(transform_triangulation_topomesh,'area',2)

        compute_topomesh_property(registered_triangulation_topomesh,'length',1)
        compute_topomesh_property(registered_triangulation_topomesh,'area',2)
        
        area_growth = transform_triangulation_topomesh.wisp_property('area',2).values() - registered_triangulation_topomesh.wisp_property('area',2).values() 
        relative_area_growth = transform_triangulation_topomesh.wisp_property('area',2).values()/registered_triangulation_topomesh.wisp_property('area',2).values() 

        registered_triangulation_topomesh.update_wisp_property('relative_surfacic_growth',2,array_dict(relative_area_growth,transform_triangulation_topomesh.wisp_property('area',2).keys()))
        compute_topomesh_vertex_property_from_faces(registered_triangulation_topomesh,'relative_surfacic_growth',neighborhood=3,adjacency_sigma=1.2)
        relative_surfacic_growth = registered_triangulation_topomesh.wisp_property('relative_surfacic_growth',0).values(floating_data.index.values)

        # transform_triangulation_topomesh.update_wisp_property('relative_surfacic_growth',2,array_dict(relative_area_growth,transform_triangulation_topomesh.wisp_property('area',2).keys()))
        # compute_topomesh_vertex_property_from_faces(transform_triangulation_topomesh,'relative_surfacic_growth',neighborhood=3,adjacency_sigma=1.2)
        # relative_surfacic_growth = transform_triangulation_topomesh.wisp_property('relative_surfacic_growth',0).values(floating_data.index.values)
        
        floating_data['previous_relative_surfacic_growth'] = relative_surfacic_growth
        if i_file == len(filenames)-2:
            floating_data['next_relative_surfacic_growth'] = np.nan
        
        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Computing previous surfacic growth "+reference_filename+" <-- "+floating_filename+" ["+str(current_time()-start_time)+" s]")


        # variable_names = ['DIIV','Normalized_DIIV','Auxin','Normalized Auxin','DR5','Normalized_DR5','CLV3','TagBFP','Normalized_TagBFP','mean_curvature','gaussian_curvature']
        variable_names = [c for c in reference_data.columns if c in quantified_signals]

        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Estimating next signal values "+reference_filename[-3:]+" --> "+floating_filename[-3:]+" "+str(variable_names))

        floating_X = floating_data['registered_x'].values
        floating_Y = floating_data['registered_y'].values
        
        reference_next_X = invert_registered_points[:,0]
        reference_next_Y = invert_registered_points[:,1]

        for var in variable_names:
            if (var in reference_data.columns) and (var in floating_data.columns):
                floating_var = floating_data[var].values
                reference_next_var = compute_local_2d_signal(np.transpose([floating_X,floating_Y]),np.transpose([reference_next_X,reference_next_Y]),floating_var)
                reference_data['next_'+var] = reference_next_var
            if i_file == len(filenames)-2:
                floating_data['next_'+var] = np.nan
        

        logging.info("".join(["  " for l in xrange(loglevel)])+"<--> Estimating next signal values "+reference_filename[-3:]+" --> "+floating_filename[-3:]+" ["+str(current_time()-start_time)+" s]")

        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Estimating previous signal values "+reference_filename[-3:]+" <-- "+floating_filename[-3:]+" "+str(variable_names))

        reference_X = reference_data['center_x'].values
        reference_Y = reference_data['center_y'].values
        
        floating_previous_X = registered_points[:,0]
        floating_previous_Y = registered_points[:,1]
        
        for var in variable_names:
            if (var in reference_data.columns) and (var in floating_data.columns):
                reference_var = reference_data[var].values
                floating_previous_var = compute_local_2d_signal(np.transpose([reference_X,reference_Y]),np.transpose([floating_previous_X,floating_previous_Y]),reference_var)
                floating_data['previous_'+var] = floating_previous_var
            if i_file == 0:
                reference_data['previous_'+var] = np.nan

        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Estimating previous signal values "+reference_filename[-3:]+" <-- "+floating_filename[-3:]+" ["+str(current_time()-start_time)+" s]")

    if save_files:
        for filename in sequence_data:
            normalized=True
            aligned=False
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            sequence_data[filename].to_csv(data_filename,index=False)

    return sequence_data
        
            