import numpy as np
from scipy import ndimage as nd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patch

# from vplants.image.serial.all import imread, imsave
# from vplants.image.spatial_image import SpatialImage

# from vplants.cellcomplex.property_topomesh.utils.delaunay_tools import delaunay_triangulation
# from vplants.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh, triangle_topomesh
# from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces

# from vplants.cellcomplex.property_topomesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components, cut_surface_topomesh, clean_topomesh
# from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function

from sam_spaghetti.signal_image_plot import load_sequence_signal_data
from sam_spaghetti.sequence_growth_estimation import load_sequence_rigid_transformations
from sam_spaghetti.utils.signal_luts import signal_colormaps, signal_ranges, signal_lut_ranges, primordia_colors

from vplants.container import array_dict

import os
import logging

def extract_clv3_circle(positions, clv3_values, clv3_threshold=0.4, cell_radius=5, density_k=0.33):
    
    X = positions[:,0]
    Y = positions[:,1]
    r_max = 200

    xx, yy = np.meshgrid(np.linspace(-r_max,r_max,101),np.linspace(-r_max,r_max,101))
    zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),clv3_values,cell_radius=cell_radius,density_k=density_k)    

    
    resolution = np.array([xx[0,1]-xx[0,0],yy[1,0]-yy[0,0]])

    zz[np.isnan(zz)] = 0
                
    clv3_regions = nd.label((zz>clv3_threshold).astype(int))[0]
    components = np.unique(clv3_regions)[1:]
    component_centers = np.transpose([nd.sum(xx,clv3_regions,index=components),nd.sum(yy,clv3_regions,index=components)])/nd.sum(np.ones_like(xx),clv3_regions,index=components)[:,np.newaxis]
    component_areas = np.array([(clv3_regions==c).sum() * np.prod(resolution) for c in components])

    #for c, a in zip(component_centers, component_areas):
        #print c," (",np.sqrt(a/np.pi),")"

    if len(component_centers)>0:
        # component_matching = vq(np.array([[0,0]]),component_centers)
        # clv3_center = component_centers[component_matching[0][0]]
        # clv3_area = (clv3_regions==component_matching[0]+1).sum() * np.prod(resolution)

        clv3_center = component_centers[np.argmax(component_areas)]
        clv3_area = np.max(component_areas)
        clv3_radius = np.sqrt(clv3_area/np.pi)


    else:
        clv3_center = clv3_radius = None
    return clv3_center, clv3_radius



            

def align_sam_sequence(sequence_name, image_dirname, save_files=True, sam_orientation=1, r_max=80, cell_radius=5., density_k=0.25, microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    filenames = np.sort(sequence_data.keys())
    for i_file, filename in enumerate(filenames):
        sequence_data[filename] = sequence_data[filename][sequence_data[filename]['layer']==1]

    previous_transform = np.diag(np.ones(4))
        
    for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):

        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Computing sequence registered points "+reference_filename+" --> "+floating_filename)      

        rigid_matrix = sequence_rigid_transforms[(reference_filename,floating_filename)]
        invert_rigid_matrix = sequence_rigid_transforms[(floating_filename,reference_filename)]

        reference_data = sequence_data[reference_filename]
        floating_data = sequence_data[floating_filename]
        
        X = reference_data['center_x'].values
        Y = reference_data['center_y'].values
        Z = reference_data['center_z'].values
        
        if i_file == 0:
            reference_data['registered_x'] = X
            reference_data['registered_y'] = Y
            reference_data['registered_z'] = Z
            
            reference_data['sequence_registered_x'] = X
            reference_data['sequence_registered_y'] = Y
            reference_data['sequence_registered_z'] = Z
            
    
        reference_points = np.transpose([X,Y,Z])

        cell_barycenters = array_dict(np.transpose([X,Y,Z]),reference_data.index.values)

        X = floating_data['center_x'].values
        Y = floating_data['center_y'].values
        Z = floating_data['center_z'].values
        
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
        
        floating_data['sequence_registered_x']=sequence_registered_points[:,0]
        floating_data['sequence_registered_y']=sequence_registered_points[:,1]
        floating_data['sequence_registered_z']=sequence_registered_points[:,2]

    figure = plt.figure(1)
    figure.clf()
    figure.patch.set_facecolor('w')
    
    for i_file, filename in enumerate(filenames):
        X = sequence_data[filename]['sequence_registered_x'].values
        Y = sequence_data[filename]['sequence_registered_y'].values
        Z = sequence_data[filename]['sequence_registered_z'].values
        
        figure.gca().scatter(X,Y,c=i_file*np.ones_like(X),cmap='jet',linewidth=0,alpha=0.6,vmin=-1,vmax=5)
        figure.gca().axis('equal')
    figure.set_size_inches(10,10)
    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_sequence_registered_nuclei.png")

    #registration_filenames = [f for f in filenames]
    registration_filenames = [f for f in filenames if (not 't04' in f) and (int(f[-2:]) <= 10)]   
    logging.info("".join(["  " for l in xrange(loglevel)])+"--> Aligning SAM sequence using "+str([f[-3:] for f in registration_filenames]))


    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Computing optimal vertical orientation")                
    X = np.concatenate([sequence_data[f]['sequence_registered_x'].values for f in registration_filenames])
    Y = np.concatenate([sequence_data[f]['sequence_registered_y'].values for f in registration_filenames])
    Z = np.concatenate([sequence_data[f]['sequence_registered_z'].values for f in registration_filenames])
    
    clv3 = np.concatenate([sequence_data[f]['Normalized_CLV3'].values for f in registration_filenames])

    centers = []
    radii = []
    for clv3_threshold in np.linspace(1.2,1.8,7):
        clv3_center, clv3_radius = extract_clv3_circle(np.transpose([X,Y]),clv3,clv3_threshold=clv3_threshold)
        centers += [clv3_center]
        radii += [clv3_radius]
    clv3_center = np.mean(centers,axis=0)
    clv3_radius = np.mean(radii)
    logging.info("".join(["  " for l in xrange(loglevel)])+"    --> CZ Circle : "+str(clv3_center)+" ("+str(clv3_radius)+")")
    
    center_altitude = compute_local_2d_signal(np.transpose([X,Y]),clv3_center,Z)[0]
    centered_positions = np.transpose([X,Y,Z])-np.array(list(clv3_center)+[center_altitude])


    figure = plt.figure(2)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.add_subplot(2,1,1)
    figure.gca().scatter(np.linalg.norm(centered_positions[:,:2],axis=1),centered_positions[:,2],linewidth=0,alpha=0.6)
    figure.gca().set_xlim(0,120)
    figure.gca().set_ylim(-50,10)

    angle_resolution = 0.01
    angle_max = 0.2
    angles = np.linspace(-angle_max,angle_max,2*(angle_max/angle_resolution)+1)
    
    psis, phis = np.meshgrid(angles,angles)
    
    rotation_mse = []
    
    for dome_phi in angles:
        phi_mse = []
        for dome_psi in angles:
            rotation_matrix_psi = np.array([[1,0,0],[0,np.cos(dome_psi),-np.sin(dome_psi)],[0,np.sin(dome_psi),np.cos(dome_psi)]])
            rotation_matrix_phi = np.array([[np.cos(dome_phi),0,-np.sin(dome_phi)],[0,1,0],[np.sin(dome_phi),0,np.cos(dome_phi)]])
            rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,centered_positions)
            rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)
            
            rotated_r = np.linalg.norm(rotated_positions[:,:2],axis=1)
            rotated_z = rotated_positions[:,2]
            r_weights = np.exp(-np.power(rotated_r,2)/np.power(20,2))
            p = np.polyfit(rotated_r,rotated_z,deg=2,w=r_weights)
        
            r = np.linspace(0,r_max,r_max+1)
            mse = (r_weights*np.power(rotated_z-np.polyval(p,rotated_r),2)).sum()/(r_weights.sum())
            phi_mse += [mse]
        rotation_mse += [phi_mse]
    
    optimal_rotation = np.where(rotation_mse==np.array(rotation_mse).min())
    optimal_phi = (phis[optimal_rotation]).mean()
    optimal_psi = (psis[optimal_rotation]).mean()

    logging.info("".join(["  " for l in xrange(loglevel)])+"    --> Optimal angles : ("+str(optimal_phi)+", "+str(optimal_psi)+")")
    
    rotation_matrix_psi = np.array([[1,0,0],[0,np.cos(optimal_psi),-np.sin(optimal_psi)],[0,np.sin(optimal_psi),np.cos(optimal_psi)]])
    rotation_matrix_phi = np.array([[np.cos(optimal_phi),0,-np.sin(optimal_phi)],[0,1,0],[np.sin(optimal_phi),0,np.cos(optimal_phi)]])
    rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,centered_positions)
    rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)

    figure.add_subplot(2,1,2)
    figure.gca().scatter(np.linalg.norm(rotated_positions[:,:2],axis=1),rotated_positions[:,2],linewidth=0,alpha=0.6)
    figure.gca().set_xlim(0,120)
    figure.gca().set_ylim(-50,10)

    figure.set_size_inches(10,10)
    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_vertical_axis_optimization.png")


    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Detecting global qDII minimum")  

    X = rotated_positions[:,0]
    Y = rotated_positions[:,1]
    Z = rotated_positions[:,2]  

    normalized_qDII = np.concatenate([sequence_data[f]['Normalized_qDII'].values for f in registration_filenames])

    radial_thetas = np.linspace(-180,180,361)*np.pi/180.
    radial_radii = np.linspace(0,r_max+10,2*r_max+21)
    
    T,R = np.meshgrid(radial_thetas,radial_radii)
    xx = R*np.cos(T)
    yy = R*np.sin(T)
    
    radial_qDII = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),normalized_qDII,cell_radius=cell_radius,density_k=density_k)
    
    nuclei_positions = dict(zip(range(len(X)),np.transpose([X,Y,np.zeros_like(X)])))
    nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(xx,yy,np.zeros_like(xx))
    
    confidence_map = nuclei_density  + np.maximum(1-np.linalg.norm([xx,yy],axis=0)/60.,0)
    confidence_map = nd.gaussian_filter(confidence_map,sigma=1.0)
        
    figure = plt.figure(2)
    figure.clf()
    figure.patch.set_facecolor('w')
    
    figure.gca().contourf(xx,yy,radial_qDII,np.linspace(signal_ranges['Normalized_qDII'][0],signal_ranges['Normalized_qDII'][1],51),cmap=signal_colormaps['Normalized_qDII'],alpha=1,antialiased=True,vmin=signal_lut_ranges['Normalized_qDII'][0],vmax=signal_lut_ranges['Normalized_qDII'][1])
    figure.gca().contour(xx,yy,radial_qDII,np.linspace(signal_ranges['Normalized_qDII'][0],signal_ranges['Normalized_qDII'][1],51),cmap='gray',alpha=0.2,linewidths=1,antialiased=True,vmin=-1,vmax=0)
        
    for a in xrange(16):
        figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)
        
    figure.gca().axis('equal')
    

    cz_ring = np.where((R/clv3_radius>0.9)&(R/clv3_radius<1.4))
    ring_theta_min = T[cz_ring][np.where(radial_qDII[cz_ring] == radial_qDII[cz_ring].min())][0]
    ring_radius_min = R[cz_ring][np.where(radial_qDII[cz_ring] == radial_qDII[cz_ring].min())][0]
    ring_theta_min = 180.*ring_theta_min/np.pi
    
    absolute_min = ring_radius_min*np.array([np.cos(np.pi*ring_theta_min/180.),np.sin(np.pi*ring_theta_min/180.)])
    logging.info("".join(["  " for l in xrange(loglevel)])+"    --> qDII minimum : "+str(absolute_min)+" ("+str(ring_theta_min)+")")
    
    
    c = patch.RegularPolygon(xy=absolute_min,numVertices=3,radius=3,orientation=-np.pi,fc=primordia_colors[0],ec='w',lw=3,alpha=1)
    figure.gca().add_artist(c)
    
    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_Normalized_qDII_map.jpg")
        

    logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Applying alignment transform "+sequence_name) 

    theta_min = np.pi*ring_theta_min/180.
    theta_matrix = np.array([[np.cos(theta_min),-np.sin(theta_min)],[np.sin(theta_min),np.cos(theta_min)]])
    orientation_matrix = np.array([[1,0],[0,sam_orientation]])


    #theta_positions = np.einsum('...ij,...i->...j', theta_matrix,np.einsum('...ij,...i->...j',orientation_matrix,np.transpose([X,Y])))
    theta_positions = np.einsum('...ij,...i->...j', orientation_matrix,np.einsum('...ij,...i->...j',theta_matrix,np.transpose([X,Y])))
    #offset = 0

    aligned_centers = []
    aligned_radii = []
    for clv3_threshold in np.linspace(1.2,1.8,7):
        aligned_clv3_center, aligned_clv3_radius = extract_clv3_circle(theta_positions,clv3,clv3_threshold=clv3_threshold)
        aligned_centers += [aligned_clv3_center]
        aligned_radii += [aligned_clv3_radius]
    aligned_clv3_center = np.mean(aligned_centers,axis=0)
    aligned_clv3_radius = np.mean(aligned_radii)
    logging.info("".join(["  " for l in xrange(loglevel)])+"    --> Aligned CZ Circle : "+str(aligned_clv3_center)+" ("+str(aligned_clv3_radius)+")")
    
    
    for i_file, filename in enumerate(filenames):
        logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Applying alignment transform on "+filename) 

        data = sequence_data[filename]
        
        X = data['sequence_registered_x'].values
        Y = data['sequence_registered_y'].values
        Z = data['sequence_registered_z'].values
        
        centered_positions = np.transpose([X,Y,Z])-np.array(list(clv3_center)+[center_altitude])            
        rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,centered_positions)
        rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)
        theta_positions = np.einsum('...ij,...i->...j', orientation_matrix,np.einsum('...ij,...i->...j',theta_matrix,rotated_positions[:,:2]))
       
        sequence_data[filename]['aligned_x'] = theta_positions[:,0]
        sequence_data[filename]['aligned_y'] = theta_positions[:,1]
        sequence_data[filename]['aligned_z'] = rotated_positions[:,2]
    
        if save_files:
            normalized=True
            aligned=True
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            sequence_data[filename].to_csv(data_filename,index=False)

    absolute_min = ring_radius_min*np.array([1,0])

    return sequence_data







