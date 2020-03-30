import numpy as np
from scipy import ndimage as nd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.patheffects as path_effects
import matplotlib as mpl

from tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function
from tissue_nukem_3d.signal_map import SignalMap
from tissue_nukem_3d.signal_map_visualization import plot_signal_map
from tissue_nukem_3d.signal_map_analysis import compute_signal_map_landscape, signal_map_landscape_analysis, signal_map_regions

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_data, load_sequence_rigid_transformations
from sam_spaghetti.utils.signal_luts import signal_colormaps, signal_ranges, signal_lut_ranges, primordia_colors

from cellcomplex.utils import array_dict

import os
import logging


golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)
golden_angle = np.degrees(golden_angle)


def extract_clv3_circle(signal_data, position_name='center', clv3_threshold_range=np.linspace(1.2,1.8,7), cell_radius=5, density_k=0.33, verbose=False, debug=False, loglevel=0):
    if not "Normalized_CLV3" in signal_data.columns:
        if not "CLV3" in signal_data.columns:
            raise KeyError("CLV3 signal is not defined! Unable to estimate CZ position!")
        else:
            signal_data['Normalized_CLV3'] = signal_data['CLV3']/signal_data[signal_data['layer']==1]['CLV3'].mean()

    signal_map = SignalMap(signal_data, extent=200, position_name=position_name, resolution=2, polar=False, radius=cell_radius, density_k=density_k)


    # clv3_map = signal_map.signal_map("Normalized_CLV3")

    centers = []
    radii = []
    for clv3_threshold in clv3_threshold_range:

        clv3_regions = signal_map_regions(signal_map,'Normalized_CLV3',threshold=clv3_threshold)
        if len(clv3_regions)>0:
            clv3_region = clv3_regions.iloc[np.argmax(clv3_regions['area'])]
            clv3_radius = np.sqrt(clv3_region['area']/np.pi)
            clv3_center = clv3_region[['center_x','center_y']].values
            centers += [clv3_center]
            radii += [clv3_radius]
        
        # clv3_regions = nd.label((clv3_map>clv3_threshold).astype(int))[0]
        # components = np.unique(clv3_regions)[1:]
        # component_centers = np.transpose([nd.sum(xx,clv3_regions,index=components),nd.sum(yy,clv3_regions,index=components)])/nd.sum(np.ones_like(xx),clv3_regions,index=components)[:,np.newaxis]
        # component_areas = np.array([(clv3_regions==c).sum() * np.prod(resolution) for c in components])

        # if len(component_centers)>0:
        #     clv3_center = component_centers[np.argmax(component_areas)]
        #     clv3_area = np.max(component_areas)
        #     clv3_radius = np.sqrt(clv3_area/np.pi)

        #     centers += [clv3_center]
        #     radii += [clv3_radius]

    clv3_center = np.mean(centers,axis=0)
    clv3_radius = np.mean(radii)
    
    return clv3_center, clv3_radius


def optimize_vertical_axis(positions, angle_max=0.2, angle_resolution=0.01, r_max=80, verbose=False, debug=False, loglevel=0):
    """
    """

    angles = np.linspace(-angle_max,angle_max,int(np.ceil(2*(angle_max/angle_resolution)+1)))
    
    psis, phis = np.meshgrid(angles,angles)
    
    rotation_mse = []
    
    for dome_phi in angles:
        phi_mse = []
        for dome_psi in angles:
            rotation_matrix_psi = np.array([[1,0,0],[0,np.cos(dome_psi),-np.sin(dome_psi)],[0,np.sin(dome_psi),np.cos(dome_psi)]])
            rotation_matrix_phi = np.array([[np.cos(dome_phi),0,-np.sin(dome_phi)],[0,1,0],[np.sin(dome_phi),0,np.cos(dome_phi)]])
            rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,positions)
            rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)
            
            rotated_r = np.linalg.norm(rotated_positions[:,:2],axis=1)
            rotated_z = rotated_positions[:,2]
            r_weights = np.exp(-np.power(rotated_r,2)/np.power(20,2))
            p = np.polyfit(rotated_r,rotated_z,deg=2,w=r_weights)
        
            r = np.linspace(0,r_max,int(r_max+1))
            mse = (r_weights*np.power(rotated_z-np.polyval(p,rotated_r),2)).sum()/(r_weights.sum())
            phi_mse += [mse]
        rotation_mse += [phi_mse]
    
    optimal_rotation = np.where(rotation_mse==np.array(rotation_mse).min())
    optimal_phi = (phis[optimal_rotation]).mean()
    optimal_psi = (psis[optimal_rotation]).mean()

    logging.info("".join(["  " for l in range(loglevel)])+"--> Optimal angles : ("+str(optimal_phi)+", "+str(optimal_psi)+")")
    
    rotation_matrix_psi = np.array([[1,0,0],[0,np.cos(optimal_psi),-np.sin(optimal_psi)],[0,np.sin(optimal_psi),np.cos(optimal_psi)]])
    rotation_matrix_phi = np.array([[np.cos(optimal_phi),0,-np.sin(optimal_phi)],[0,1,0],[np.sin(optimal_phi),0,np.cos(optimal_phi)]])
    rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,positions)
    rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)


    return rotated_positions, np.dot(rotation_matrix_phi,rotation_matrix_psi)


def label_primordia_extrema(extrema_data, signal_name=None, clv3_radius=28., primordium_range=range(-3,6), opening_angle=75., r_max=100.):

    minima_data = extrema_data[extrema_data['extremum_type']=='minimum']
    maxima_data = extrema_data[extrema_data['extremum_type']=='maximum']
    saddle_data = extrema_data[extrema_data['extremum_type']=='saddle']
    
    extrema_fields = [c for c in extrema_data.columns if not c in ['point','area']]         

    primordia_extrema = {}
    for field in extrema_fields+['primordium']:
        primordia_extrema[field]= []

    for primordium in primordium_range:
        primordium_theta = (primordium*golden_angle + 180)%360 - 180

        max_angle = (opening_angle+10*(np.maximum(0,1-(primordium))))

        radius_min = 0.8 + 0.2*(primordium)
        radius_max = 1.7 + 0.25*(primordium)
            
        primordium_minima_data = minima_data.copy(deep=True)
        
        primordium_minima_data['score'] *= np.cos(np.minimum(np.pi*np.abs((primordium_minima_data['aligned_theta']-primordium_theta+180)%360-180)/max_angle,np.pi))
        primordium_minima_data['score'] *= np.minimum(np.power(radius_max/(primordium_minima_data['radial_distance']/clv3_radius),2),1)
        primordium_minima_data['score'] *= np.minimum(np.power((primordium_minima_data['radial_distance']/clv3_radius)/(radius_min),2),1)
        primordium_minima_data = primordium_minima_data[primordium_minima_data['score']>0]
        if primordium_minima_data['score'].max() > 0.2:
            primordium_minimum = primordium_minima_data.index[np.argmax(primordium_minima_data['score'])]
        else:
            primordium_minimum = None
                
        if primordium_minimum is not None:
            extremum_data = primordium_minima_data[primordium_minima_data.index==primordium_minimum]
            primordia_extrema['primordium'] += [primordium]
            for field in extrema_fields:
                primordia_extrema[field] += list(extremum_data[field])
        primordium_maxima_data = maxima_data.copy(deep=True)
        
        primordium_maxima_data['score'] *= np.cos(np.minimum(np.pi*np.abs((primordium_maxima_data['aligned_theta']-primordium_theta+180)%360-180)/max_angle,np.pi))
        primordium_maxima_data['score'] *= np.minimum(np.power((radius_max-0.3)/(primordium_maxima_data['radial_distance']/clv3_radius),2),1)
        primordium_maxima_data['score'] *= np.minimum(np.power(np.maximum((primordium_maxima_data['radial_distance']/clv3_radius)/(radius_min-0.3),0),2),1)
        primordium_maxima_data = primordium_maxima_data[primordium_maxima_data['score']>0]
        if primordium_minimum is not None:
            primordium_maxima_data['score'] *= primordium_minima_data['radial_distance'][primordium_minimum]/primordium_maxima_data['radial_distance'] > 1.05
            if signal_name is not None:
                primordium_maxima_data['score'] *= np.maximum(np.minimum(1,(primordium_maxima_data[signal_name]-primordium_minima_data[signal_name][primordium_minimum])/0.2),0)
        if primordium_maxima_data['score'].max() > 0.2:
            primordium_maximum = primordium_maxima_data.index[np.argmax(primordium_maxima_data['score'])]
        else:
            primordium_maximum = None

        primordium_saddle_data = saddle_data.copy(deep=True)
        primordium_saddle_data['score'] *= np.cos(np.minimum(np.pi*np.abs((primordium_saddle_data['aligned_theta']-primordium_theta+180)%360-180)/max_angle,np.pi))
        primordium_saddle_data['score'] *= np.minimum(np.power((radius_max-0.3)/(primordium_saddle_data['radial_distance']/clv3_radius),2),1)
        primordium_saddle_data['score'] *= np.minimum(np.power(np.maximum((primordium_saddle_data['radial_distance']/clv3_radius)/(radius_min-0.3),0),2),1)
        primordium_saddle_data = primordium_saddle_data[primordium_saddle_data['score']>0]
        if primordium_minimum is not None:
            primordium_saddle_data['score'] *= primordium_minima_data['radial_distance'][primordium_minimum]/primordium_saddle_data['radial_distance'] > 1.05 
            if signal_name is not None:
                primordium_saddle_data['score'] *= np.maximum(np.minimum(1,(primordium_saddle_data[signal_name]-primordium_minima_data[signal_name][primordium_minimum])/0.2),0)
        if (primordium_maximum is None) or (primordium_maxima_data['score'].max()<primordium_saddle_data['score'].max()):
            if primordium_saddle_data['score'].max() > 0.2:
                primordium_maximum = None
                primordium_saddle = primordium_saddle_data.index[np.argmax(primordium_saddle_data['score'])]
            else:
                primordium_maximum = None
                primordium_saddle = None
        else:
            primordium_saddle = None

        if primordium_maximum is not None:
            extremum_data = primordium_maxima_data[primordium_maxima_data.index==primordium_maximum]
            primordia_extrema['primordium'] += [primordium]
            for field in extrema_fields:
                primordia_extrema[field] += list(extremum_data[field])
                   
        if primordium_saddle is not None:
            extremum_data = primordium_saddle_data[primordium_saddle_data.index==primordium_saddle]
            primordia_extrema['primordium'] += [primordium]
            for field in extrema_fields:
                primordia_extrema[field] += list(extremum_data[field])
        
    primordia_extrema_data = pd.DataFrame().from_dict(primordia_extrema)

    return primordia_extrema_data


def align_sam_sequence(sequence_name, image_dirname, save_files=True, sam_orientation=1, r_max=120, cell_radius=5., density_k=0.25, microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    filenames = np.sort(list(sequence_data.keys()))
    for i_file, filename in enumerate(filenames):
        sequence_data[filename] = sequence_data[filename][sequence_data[filename]['layer']==1]

        file_data = sequence_data[filename]

        X = file_data['center_x'].values
        Y = file_data['center_y'].values
        Z = file_data['center_z'].values
        
        if i_file == 0:
            file_data['registered_x'] = X
            file_data['registered_y'] = Y
            file_data['registered_z'] = Z
            
            file_data['sequence_registered_x'] = X
            file_data['sequence_registered_y'] = Y
            file_data['sequence_registered_z'] = Z

    previous_transform = np.diag(np.ones(4))
        
    for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):

        logging.info("".join(["  " for l in range(loglevel)])+"--> Computing sequence registered points "+reference_filename+" --> "+floating_filename)

        rigid_matrix = sequence_rigid_transforms[(reference_filename,floating_filename)]
        invert_rigid_matrix = sequence_rigid_transforms[(floating_filename,reference_filename)]

        reference_data = sequence_data[reference_filename]
        floating_data = sequence_data[floating_filename]
        
        X = reference_data['center_x'].values
        Y = reference_data['center_y'].values
        Z = reference_data['center_z'].values
            
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

    for f in filenames:
        if not 'qDII' in sequence_data[f].columns:
            sequence_data[f]['qDII'] = sequence_data[f]['DIIV']
            sequence_data[f]['Normalized_qDII'] = sequence_data[f]['Normalized_DIIV']
            sequence_data[f]['DIIV'] = sequence_data[f]['DIIV']*sequence_data[f]['TagBFP']


    #registration_filenames = [f for f in filenames]
    registration_filenames = [f for f in filenames if (not 't04' in f) and (int(f[-2:]) <= 10)]   
    logging.info("".join(["  " for l in range(loglevel)])+"--> Aligning SAM sequence using "+str([f[-3:] for f in registration_filenames]))


    logging.info("".join(["  " for l in range(loglevel)])+"  --> Computing optimal vertical orientation")

    registration_data = pd.concat([sequence_data[f] for f in registration_filenames])
    X = np.concatenate([sequence_data[f]['sequence_registered_x'].values for f in registration_filenames])
    Y = np.concatenate([sequence_data[f]['sequence_registered_y'].values for f in registration_filenames])
    Z = np.concatenate([sequence_data[f]['sequence_registered_z'].values for f in registration_filenames])

    clv3_center, clv3_radius = extract_clv3_circle(registration_data,position_name='sequence_registered')
    logging.info("".join(["  " for l in range(loglevel)])+"    --> CZ Circle : "+str(clv3_center)+" ("+str(clv3_radius)+")")
    
    center_altitude = compute_local_2d_signal(np.transpose([X,Y]),clv3_center,Z)[0]
    centered_positions = np.transpose([X,Y,Z])-np.array(list(clv3_center)+[center_altitude])

    rotated_positions, rotation_matrix = optimize_vertical_axis(centered_positions,r_max=r_max,verbose=verbose,debug=debug,loglevel=loglevel+2) 

    figure = plt.figure(1)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.add_subplot(2,1,1)
    figure.gca().scatter(np.linalg.norm(centered_positions[:,:2],axis=1),centered_positions[:,2],linewidth=0,alpha=0.6)
    figure.gca().set_xlim(0,120)
    figure.gca().set_ylim(-50,10)

    figure.add_subplot(2,1,2)
    figure.gca().scatter(np.linalg.norm(rotated_positions[:,:2],axis=1),rotated_positions[:,2],linewidth=0,alpha=0.6)
    figure.gca().set_xlim(0,120)
    figure.gca().set_ylim(-50,10)

    figure.set_size_inches(10,10)
    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_vertical_axis_optimization.png")
    
    logging.info("".join(["  " for l in range(loglevel)])+"  --> Detecting global qDII minimum")

    X = rotated_positions[:,0]
    Y = rotated_positions[:,1]
    Z = rotated_positions[:,2]  

    normalized_qDII = np.concatenate([sequence_data[f]['Normalized_qDII'].values for f in registration_filenames])
    normalized_clv3 = np.concatenate([sequence_data[f]['Normalized_CLV3'].values for f in registration_filenames])
    
    radial_thetas = np.linspace(-180,180,361)*np.pi/180.
    radial_radii = np.linspace(0,r_max+10,int(2*r_max+21))
    
    T,R = np.meshgrid(radial_thetas,radial_radii)
    xx = R*np.cos(T)
    yy = R*np.sin(T)

    radial_clv3 = compute_local_2d_signal(np.transpose([X, Y]), np.transpose([xx, yy], (1, 2, 0)), normalized_clv3, cell_radius=cell_radius, density_k=density_k)
    radial_qDII = compute_local_2d_signal(np.transpose([X, Y]), np.transpose([xx, yy], (1, 2, 0)), normalized_qDII, cell_radius=cell_radius, density_k=density_k)

    nuclei_positions = dict(zip(range(len(X)),np.transpose([X,Y,np.zeros_like(X)])))
    nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(xx,yy,np.zeros_like(xx))
    
    confidence_map = nuclei_density  + np.maximum(1-np.linalg.norm([xx,yy],axis=0)/60.,0)
    confidence_map = nd.gaussian_filter(confidence_map,sigma=1.0)

    figure = plt.figure(1)
    figure.clf()
    figure.patch.set_facecolor('w')

    figure.add_subplot(1,2,1)

    figure.gca().contourf(xx,yy,radial_clv3,np.linspace(0,5,51),cmap=signal_colormaps['CLV3'],alpha=1,antialiased=True,vmin=0,vmax=5)
    figure.gca().contour(xx,yy,radial_clv3,np.linspace(0,5,51),cmap='gray',alpha=0.2,linewidths=1,antialiased=True,vmin=-1,vmax=0)
        
    for a in range(16):
        figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)
        
    figure.gca().axis('equal')

    c = patch.Circle(xy=[0, 0], radius=clv3_radius, ec="k", fc='None', lw=7, alpha=1)
    figure.gca().add_artist(c)
    c = patch.Circle(xy=[0, 0], radius=clv3_radius, ec="#c94389", fc='None', lw=5, alpha=1)
    figure.gca().add_artist(c)

    figure.gca().set_xlim(microscope_orientation*(-r_max - 10), microscope_orientation*(r_max + 10))
    figure.gca().set_ylim(microscope_orientation*(-r_max - 10), microscope_orientation*(r_max + 10))
    figure.gca().axis('off')

    figure.add_subplot(1,2,2)

    
    figure.gca().contourf(xx,yy,radial_qDII,np.linspace(signal_ranges['Normalized_qDII'][0],signal_ranges['Normalized_qDII'][1],51),cmap=signal_colormaps['Normalized_qDII'],alpha=1,antialiased=True,vmin=signal_lut_ranges['Normalized_qDII'][0],vmax=signal_lut_ranges['Normalized_qDII'][1])
    figure.gca().contour(xx,yy,radial_qDII,np.linspace(signal_ranges['Normalized_qDII'][0],signal_ranges['Normalized_qDII'][1],51),cmap='gray',alpha=0.2,linewidths=1,antialiased=True,vmin=-1,vmax=0)
        
    for a in range(16):
        figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)
        
    figure.gca().axis('equal')

    cz_bounds = [0.9,1.6]

    cz_ring = np.where((R/clv3_radius>cz_bounds[0])&(R/clv3_radius<cz_bounds[1]))
    ring_theta_min = T[cz_ring][np.where(radial_qDII[cz_ring] == radial_qDII[cz_ring].min())][0]
    ring_radius_min = R[cz_ring][np.where(radial_qDII[cz_ring] == radial_qDII[cz_ring].min())][0]
    ring_theta_min = 180.*ring_theta_min/np.pi
    
    absolute_min = ring_radius_min*np.array([np.cos(np.pi*ring_theta_min/180.),np.sin(np.pi*ring_theta_min/180.)])
    logging.info("".join(["  " for l in range(loglevel)])+"    --> qDII minimum : "+str(absolute_min)+" ("+str(ring_theta_min)+")")

    c = patch.Circle(xy=[0, 0], radius=clv3_radius, ec="#c94389", fc='None', lw=5, alpha=0.5)
    figure.gca().add_artist(c)
    c = patch.Circle(xy=[0, 0], radius=cz_bounds[0]*clv3_radius, ec="#c94389", fc='None', lw=2, alpha=0.25)
    figure.gca().add_artist(c)
    c = patch.Circle(xy=[0, 0], radius=cz_bounds[1]*clv3_radius, ec="#c94389", fc='None', lw=2, alpha=0.25)
    figure.gca().add_artist(c)
    
    c = patch.RegularPolygon(xy=absolute_min,numVertices=3,radius=5,orientation=-np.pi,fc=primordia_colors[0],ec='w',lw=3,alpha=1)
    figure.gca().add_artist(c)

    figure.gca().set_xlim(microscope_orientation*(-r_max - 10), microscope_orientation*(r_max + 10))
    figure.gca().set_ylim(microscope_orientation*(-r_max - 10), microscope_orientation*(r_max + 10))
    figure.gca().axis('off')

    figure.set_size_inches(18*2,18)
    figure.tight_layout()

    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_Normalized_CLV3_qDII_maps.jpg")

    logging.info("".join(["  " for l in range(loglevel)])+"  --> Applying alignment transform "+sequence_name)

    theta_min = np.pi*ring_theta_min/180.
    theta_matrix = np.array([[np.cos(theta_min),-np.sin(theta_min)],[np.sin(theta_min),np.cos(theta_min)]])
    orientation_matrix = np.array([[1,0],[0,sam_orientation]])

    #theta_positions = np.einsum('...ij,...i->...j', theta_matrix,np.einsum('...ij,...i->...j',orientation_matrix,np.transpose([X,Y])))
    theta_positions = np.einsum('...ij,...i->...j', orientation_matrix,np.einsum('...ij,...i->...j',theta_matrix,np.transpose([X,Y])))
    #offset = 0
    
    for i_file, filename in enumerate(filenames):
        logging.info("".join(["  " for l in range(loglevel)])+"  --> Applying alignment transform on "+filename)

        data = sequence_data[filename]
        
        X = data['sequence_registered_x'].values
        Y = data['sequence_registered_y'].values
        Z = data['sequence_registered_z'].values
        
        centered_positions = np.transpose([X,Y,Z])-np.array(list(clv3_center)+[center_altitude])            
        # rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_psi,centered_positions)
        # rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix_phi,rotated_positions)
        rotated_positions = np.einsum('...ij,...j->...i',rotation_matrix,centered_positions)
        theta_positions = np.einsum('...ij,...i->...j', orientation_matrix,np.einsum('...ij,...i->...j',theta_matrix,rotated_positions[:,:2]))
       
        sequence_data[filename]['aligned_x'] = theta_positions[:,0]
        sequence_data[filename]['aligned_y'] = theta_positions[:,1]
        sequence_data[filename]['aligned_z'] = rotated_positions[:,2]

        sequence_data[filename]['clv3_radius'] = [clv3_radius for i in range(len(sequence_data[filename]))]
    
        if save_files:
            normalized=True
            aligned=True
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            sequence_data[filename].to_csv(data_filename,index=False)

    return sequence_data


def detect_organ_primordia(sequence_name, image_dirname, save_files=True, r_max=80, cell_radius=5., density_k=0.25, microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=verbose, debug=debug, loglevel=loglevel)
    
    filenames = np.sort(list(sequence_data.keys()))
    for i_file, filename in enumerate(filenames):
        sequence_data[filename] = sequence_data[filename][sequence_data[filename]['layer']==1]

    radial_thetas = np.linspace(-180,180,361)*np.pi/180.
    radial_radii = np.linspace(0,r_max+10,int(2*r_max+21))

    figure = plt.figure(0)
    figure.clf()
    figure.patch.set_facecolor('w')

    for i_file, filename in enumerate(filenames):
        data = sequence_data[filename]
        
        X = data['aligned_x'].values
        Y = data['aligned_y'].values
        Z = data['aligned_z'].values

        clv3_radius = data['clv3_radius'].mean()

        logging.info("".join(["  " for l in range(loglevel)])+"--> Computing qDII landscape "+filename)
        

        # file_map = SignalMap(data, extent=r_max+10, position_name='aligned', resolution=0.5, polar=True, radius=cell_radius, density_k=density_k)
        file_map = SignalMap(data, extent=r_max+10, position_name='aligned', resolution=0.5, polar=False, radius=cell_radius, density_k=density_k)
        file_map.compute_signal_map('Normalized_qDII')

        xx = file_map.xx
        yy = file_map.yy
        R = file_map.rr
        T = file_map.tt

        signal_name = "Normalized_qDII"

        compute_signal_map_landscape(file_map,[signal_name])

        qDII_gradient_valleys = file_map.signal_map(signal_name+"_valleys")
        qDII_gradient_ridges = file_map.signal_map(signal_name+"_ridges")

        figure.add_subplot(3,len(filenames),i_file+1)

        figure.gca().pcolormesh(xx,yy,(qDII_gradient_ridges-qDII_gradient_valleys),cmap='PuOr_r',antialiased=True,shading='gouraud',vmin=-0.25,vmax=0.25)
        figure.gca().pcolormesh(xx,yy,4.*np.sqrt(qDII_gradient_ridges*qDII_gradient_valleys),cmap='Greens',antialiased=True,shading='gouraud',vmin=0,vmax=0.25,alpha=0.2)

        qdII_gradient_saddles = nd.binary_dilation(qDII_gradient_ridges*qDII_gradient_valleys>0.01,iterations=2)
        figure.gca().contour(xx,yy,(qDII_gradient_ridges>0.03)&(file_map.confidence_map>0.5)&(np.logical_not(qdII_gradient_saddles)),[0.5],cmap='Oranges',antialiased=True,vmin=-1,vmax=2)
        figure.gca().contour(xx,yy,(qDII_gradient_valleys>0.03)&(file_map.confidence_map>0.5)&(np.logical_not(qdII_gradient_saddles)),[0.5],cmap='Purples',antialiased=True,vmin=-1,vmax=2)
        figure.gca().contour(xx,yy,(qdII_gradient_saddles)&(file_map.confidence_map>0.5),[0.5],cmap='Greens',antialiased=True,vmin=-1,vmax=2)
 
        for a in range(16):
            figure.gca().contourf(xx,yy,file_map.confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)
        
        # CS = figure.gca().contour(xx, yy, R,np.linspace(0,80,17),cmap='Greys',vmin=-1,vmax=0,alpha=0.1)
        # figure.gca().clabel(CS, inline=1, fontsize=10,alpha=0.1)
        
        # c = patch.Circle(xy=[0,0],radius=clv3_radius,ec="#c94389",fc='None',lw=5,alpha=0.5)
        # figure.gca().add_artist(c)

        figure.gca().set_xlim(-r_max-10,r_max+10)
        figure.gca().set_ylim(-r_max-10,r_max+10)
        figure.gca().axis('off')
        
        # landscape_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_qDII_raw_landscape.png"])
        # landscape_figure.set_size_inches(18,18)
        # landscape_figure.savefig(landscape_filename)

        logging.info("".join(["  " for l in range(loglevel)])+"--> Detecting landscape extremal points "+filename)

        extrema_data = signal_map_landscape_analysis(file_map,signal_name,threshold=0.03)

        extrema_data['score'] = 1
        extrema_data['score'] *= [(1-s) if t=='minimum' else 0.2+s if t=='saddle' else s for s,t in extrema_data[[signal_name,'extremum_type']].values]
        # extrema_data['score'][extrema_data['extremum_type']=='minimum'] *= 1. - extrema_data[extrema_data['extremum_type']=='minimum'][signal_name]
        # extrema_data['score'][extrema_data['extremum_type']=='maximum'] *= extrema_data[extrema_data['extremum_type']=='maximum'][signal_name]
        # extrema_data['score'][extrema_data['extremum_type']=='saddle'] *= 0.2 + extrema_data[extrema_data['extremum_type']=='saddle'][signal_name]
        extrema_data['score'] *= np.minimum(1,0.5+extrema_data['extremality'])
        # extrema_data['score'][extrema_data['extremum_type']!='saddle'] *= 1 - 1./extrema_data[extrema_data['extremum_type']!='saddle']['area']
        extrema_data['score'] *= [1 - 1./a if t != 'saddle' else 1 for a,t in extrema_data[['area','extremum_type']].values]
        extrema_data['score'] *= (extrema_data['radial_distance']/clv3_radius)>0.5
        extrema_data = extrema_data[extrema_data['score']>0]

        minima_data = extrema_data[extrema_data['extremum_type']=='minimum']
        maxima_data = extrema_data[extrema_data['extremum_type']=='maximum']
        saddle_data = extrema_data[extrema_data['extremum_type']=='saddle']

        figure.add_subplot(3,len(filenames),len(filenames)+i_file+1)

        signal_name = "Normalized_qDII"
        plot_signal_map(file_map, signal_name, figure, colormap='Greys', signal_range=signal_ranges[signal_name], signal_lut_range=(signal_lut_ranges[signal_name][0],signal_lut_ranges[signal_name][0]), distance_rings=False)

        figure.gca().pcolormesh(xx,yy,(qDII_gradient_ridges-qDII_gradient_valleys),cmap='PuOr_r',antialiased=True,shading='gouraud',vmin=-0.25,vmax=0.25,alpha=0.5)
        figure.gca().pcolormesh(xx,yy,4.*np.sqrt(qDII_gradient_ridges*qDII_gradient_valleys),cmap='Greens',antialiased=True,shading='gouraud',vmin=0,vmax=0.25,alpha=0.1)


        for a in range(16):
            figure.gca().contourf(xx, yy, file_map.confidence_map, [-100, 0.1 + a / 24.], cmap='gray_r', alpha=1 - a / 15., vmin=1, vmax=2)

        figure.gca().set_xlim(-r_max - 10, r_max + 10)
        figure.gca().set_ylim(-r_max - 10, r_max + 10)
        figure.gca().axis('off')
        
        figure.gca().scatter(minima_data['aligned_x'],minima_data['aligned_y'],c=np.ones(len(minima_data)),cmap='Purples',s=300,marker="v",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        for l in minima_data.index:
            # if minima_data['score'][l]>0.2:
            if minima_data['score'][l]>0.:
                #figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],"$_{"+str(np.round(minima_data['score'][l],2))+"}$",color='b',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                #figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],"$_{"+str(np.round(minima_data['score'][l],2))+" ("+str(np.round(minima_data['extremality'][l],2))+")}$",color='indigo',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],"$_{"+str(np.round(minima_data['score'][l],2))+"}$",color='indigo',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],str(l),color='indigo',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])
        
        figure.gca().scatter(maxima_data['aligned_x'],maxima_data['aligned_y'],c=np.ones(len(maxima_data)),cmap='Oranges',s=300,marker="^",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        for l in maxima_data.index:
            # if maxima_data['score'][l]>0.2:
            if maxima_data['score'][l]>0.:
                #figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],"$_{"+str(np.round(maxima_data['score'][l],2))+"}$",color='r',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                #figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],"$_{"+str(np.round(maxima_data['score'][l],2))+" ("+str(np.round(maxima_data['extremality'][l],2))+")}$",color='darkorange',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],"$_{"+str(np.round(maxima_data['score'][l],2))+"}$",color='darkorange',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],str(l),color='darkorange',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])

        figure.gca().scatter(saddle_data['aligned_x'],saddle_data['aligned_y'],c=np.ones(len(saddle_data)),cmap='Greens',s=300,marker="o",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        for l in saddle_data.index:
            if saddle_data['score'][l]>0:
                #figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],"$_{"+str(np.round(saddle_data['score'][l],2))+"}$",color='m',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                #figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],"$_{"+str(np.round(saddle_data['score'][l],2))+" ("+str(np.round(saddle_data['extremality'][l],2))+")}$",color='forestgreen',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],"$_{"+str(np.round(saddle_data['score'][l],2))+"}$",color='forestgreen',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
                figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],str(l),color='forestgreen',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])
        
        # landscape_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_qDII_landscape.png"])
        # landscape_figure.set_size_inches(18,18)
        # landscape_figure.savefig(landscape_filename)

        # extrema_data = pd.concat([minima_data,maxima_data,saddle_data])
        extrema_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_all_extrema.csv"])    
        extrema_data.to_csv(extrema_filename,index=False)

        logging.info("".join(["  " for l in range(loglevel)])+"--> Labelling extremal points by primordia "+filename)

        opening_angle = 75.
        primordia_extrema_data = label_primordia_extrema(extrema_data, signal_name, clv3_radius, opening_angle=opening_angle)

        minima_domains = np.ones_like(T)
        primordia_domains = {}   
                    
        for primordium in range(-3,6):
        # for primordium in [1]:
        # for primordium in [0,1,2]:
            primordium_theta = (primordium*golden_angle + 180)%360 - 180
                           
            primordia_domains[primordium] = np.copy(minima_domains)
            
            max_angle = (opening_angle+10*(np.maximum(0,1-(primordium))))
            
            primordia_domains[primordium] *= np.cos(np.minimum(np.pi*np.abs(((180*T/np.pi)-primordium_theta+180)%360-180)/max_angle,np.pi))
            
            radius_min = 0.9 + 0.2*(primordium)
            radius_max = 1.9 + 0.25*(primordium)
            
            primordia_domains[primordium] *= np.minimum(np.power(radius_max/(R/clv3_radius),2),1)
            primordia_domains[primordium] *= np.minimum(np.power((R/clv3_radius)/radius_min,2),1)
            primordia_domains[primordium] *= (R/clv3_radius)>0.5
            primordia_domains[primordium] = np.maximum(0,np.minimum(primordia_domains[primordium],1))
            
            color = np.array([int(primordia_colors[primordium][1+2*k:3+2*k],16) for k in range(3)])
            color_dict = dict(red=[],green=[],blue=[])
            for k,c in enumerate(['red','green','blue']):
                color_dict[c] += [(0,1.,1.),(1,color[k]/255.,color[k]/255.)]
            for c in ['red','green','blue']:
                color_dict[c] = tuple(color_dict[c])
            mpl_cmap = mpl.colors.LinearSegmentedColormap("primordium_"+str(primordium), color_dict)
            figure.gca().contour(xx,yy,primordia_domains[primordium],np.linspace(0.1,1,11),cmap=mpl_cmap)


        figure.gca().scatter(minima_data['aligned_x'],minima_data['aligned_y'],c=np.ones(len(minima_data)),cmap='Greys',s=200,marker="v",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        # for l in minima_data.index:
        #     if minima_data['score'][l]>0.2:
        #         #primordium_figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],"$_{"+str(np.round(minima_data['score'][l],2))+"}$",color='b',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],"$_{"+str(np.round(minima_data['score'][l],2))+" ("+str(np.round(minima_data['extremality'][l],2))+")}$",color='seagreen',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(minima_data['aligned_x'][l],minima_data['aligned_y'][l],str(l),color='seagreen',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])
        
        figure.gca().scatter(maxima_data['aligned_x'],maxima_data['aligned_y'],c=np.ones(len(maxima_data)),cmap='Greys',s=200,marker="^",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        # for l in maxima_data.index:
        #     if maxima_data['score'][l]>0.2:
        #         #primordium_figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],"$_{"+str(np.round(maxima_data['score'][l],2))+"}$",color='r',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],"$_{"+str(np.round(maxima_data['score'][l],2))+" ("+str(np.round(maxima_data['extremality'][l],2))+")}$",color='chocolate',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(maxima_data['aligned_x'][l],maxima_data['aligned_y'][l],str(l),color='chocolate',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])
        
        figure.gca().scatter(saddle_data['aligned_x'],saddle_data['aligned_y'],c=np.ones(len(saddle_data)),cmap='Greys',s=200,marker="o",edgecolor='w',linewidth=1,vmin=0,vmax=2)
        # for l in saddle_data.index:
        #     if saddle_data['score'][l]>0:
        #         #primordium_figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],"$_{"+str(np.round(saddle_data['score'][l],2))+"}$",color='m',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],"$_{"+str(np.round(saddle_data['score'][l],2))+" ("+str(np.round(saddle_data['extremality'][l],2))+")}$",color='limegreen',size=10,verticalalignment='top',path_effects=[mpl.patheffects.withStroke(linewidth=2,foreground="w")])
        #         primordium_figure.gca().text(saddle_data['aligned_x'][l],saddle_data['aligned_y'][l],str(l),color='limegreen',size=11,path_effects=[mpl.patheffects.withStroke(linewidth=3,foreground="w")])

        figure.add_subplot(3,len(filenames),2*len(filenames)+i_file+1)

        signal_name = "Normalized_qDII"
        plot_signal_map(file_map, signal_name, figure, colormap=signal_colormaps[signal_name], signal_range=signal_ranges[signal_name], signal_lut_range=signal_lut_ranges[signal_name], distance_rings=False)

        for primordium in range(-3,6):
        #for primordium in [0,1,2]:
            primordium_minima_data = primordia_extrema_data[(primordia_extrema_data['primordium']==primordium)&(primordia_extrema_data['extremum_type']=='minimum')]
            primordium_maxima_data = primordia_extrema_data[(primordia_extrema_data['primordium']==primordium)&(primordia_extrema_data['extremum_type']=='maximum')]
            primordium_saddle_data = primordia_extrema_data[(primordia_extrema_data['primordium']==primordium)&(primordia_extrema_data['extremum_type']=='saddle')]
            
            figure.gca().scatter(primordium_minima_data['aligned_x'],primordium_minima_data['aligned_y'],s=480,marker="v",facecolor=primordia_colors[primordium],edgecolor='w',linewidth=3)
            figure.gca().scatter(primordium_maxima_data['aligned_x'],primordium_maxima_data['aligned_y'],s=480,marker="^",facecolor=primordia_colors[primordium],edgecolor='w',linewidth=3)
            figure.gca().scatter(primordium_saddle_data['aligned_x'],primordium_saddle_data['aligned_y'],s=480,marker="o",facecolor=primordia_colors[primordium],edgecolor='w',linewidth=3)


        primordia_extrema_data['clv3_radius'] = [clv3_radius for i in range(len(primordia_extrema_data))]
        primordia_extrema_data['filename'] = [filename for i in range(len(primordia_extrema_data))]
        primordia_extrema_data['experiment'] = [data['experiment'].values[0] for i in range(len(primordia_extrema_data))]
        primordia_extrema_data['sam_id'] = [data['sam_id'].values[0] for i in range(len(primordia_extrema_data))]
        primordia_extrema_data['hour_time'] = [data['hour_time'].values[0] for i in range(len(primordia_extrema_data))]
        primordia_extrema_data['growth_condition'] = ['LD' if 'LD' in filename else 'SD' for i in range(len(primordia_extrema_data))]
        
        for field in ['aligned_z','CLV3','Normalized_CLV3','DIIV','Normalized_DIIV','qDII','Auxin']:
            if field in data.columns:
                primordia_extrema_data[field] = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([primordia_extrema_data['aligned_x'],primordia_extrema_data['aligned_y']]),data[field].values)

        primordia_extrema_data[['primordium','extremum_type','label']].to_csv(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_detected_extrema.csv",index=False)

        primordia_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_primordia_extrema.csv"]) 
        primordia_extrema_data.to_csv(primordia_data_filename,index=False)

    figure.set_size_inches(18*len(filenames),18*3)
    figure.tight_layout()

    global_primordia_filename = "".join([image_dirname+"/"+sequence_name+"/"+sequence_name,"_Normalized_qDII_primordia_detection.png"])
    figure.savefig(global_primordia_filename)

    figure.set_size_inches(10*len(filenames),10*3)
    figure.tight_layout()

