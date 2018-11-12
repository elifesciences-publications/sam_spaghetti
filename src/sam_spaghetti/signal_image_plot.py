import numpy as np 
import pandas as pd
import scipy.ndimage as nd

from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.cluster.vq import vq

import matplotlib.pyplot as plt
# import matplotlib.patches as patch
# import matplotlib as mpl
from matplotlib.colors import Normalize
# import matplotlib.patheffects as patheffect

# from vplants.container import array_dict

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function
from vplants.tissue_nukem_3d.signal_map import SignalMap, plot_signal_map
# from vplants.tissue_nukem_3d.nuclei_detection import compute_fluorescence_ratios



import sam_spaghetti.utils.signal_luts
reload(sam_spaghetti.utils.signal_luts)
from sam_spaghetti.utils.signal_luts import *


from time import time as current_time
from copy import deepcopy

import logging



def signal_image_plot(image_slices, figure=None, signal_names=None, filenames=None, aligned=False, filtering=False, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=120., microscope_orientation=-1, save_image_views=True, verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if signal_names is None:
        signal_names = image_slices.keys()
    logging.info("".join(["  " for l in xrange(loglevel)])+"--> Plotting signal images "+str(signal_names))

    assert reference_name in signal_names

    if filenames is None:
        filenames = np.sort(image_slices[reference_name].keys())
    

    if len(filenames)>0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        image_views = {}
        for signal_name in signal_names:
            image_views[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):

            # size = np.array(image_slices[reference_name][filename].shape)
            # voxelsize = microscope_orientation*np.array(image_slices[reference_name][filename].voxelsize)

            # if resolution is None:
            #     resolution = np.abs(voxelsize)[0]

            if aligned:
                # xx,yy = np.meshgrid(np.linspace(-r_max,r_max,(2*r_max)/resolution+1),np.linspace(-r_max,r_max,(2*r_max)/resolution+1))
                extent = -r_max,r_max,r_max,-r_max
            else:
                # xx,yy = np.meshgrid(np.linspace(0,((size-1)*voxelsize)[0],((size-1)*np.abs(voxelsize))[0]/resolution+1),np.linspace(0,((size-1)*voxelsize)[1],((size-1)*np.abs(voxelsize))[0]/resolution+1))
                extent = 0, 2*r_max, 2*r_max, 0
            # extent = xx.max(),xx.min(),yy.min(),yy.max()
            # extent = xx.min(),xx.max(),yy.max(),yy.min()

            logging.info("".join(["  " for l in xrange(loglevel)])+"--> Creating 2D Views : "+filename+" "+str(signal_names))
            for i_signal, signal_name in enumerate(signal_names):

                if projection_type == "L1_slice":
                    norm = Normalize(vmin=channel_ranges[signal_name][0],vmax=channel_ranges[signal_name][1])
                elif projection_type == "max_intensity":
                    norm = Normalize(vmin=signal_ranges[signal_name][0],vmax=signal_ranges[signal_name][1])

                image_views[signal_name][filename] = cm.ScalarMappable(cmap=signal_colormaps[signal_name],norm=norm).to_rgba(image_slices[signal_name][filename])


        signal_display_list = []
        signal_display_list += [[s] for s in signal_names]
        for s1 in signal_names:
            if s1 != reference_name:
                signal_display_list += [[s1,s2] for s2 in signal_names if (s2 < s1) and (s2 != reference_name)]
        signal_display_list += [[s for s in signal_names if s != reference_name]]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):

            # blend = np.zeros_like(image_views[reference_name][filename])

            for i_signal, signals_to_display in enumerate(signal_display_list):

                figure.add_subplot(len(signal_display_list),len(filenames),i_signal*len(filenames)+i_time+1)

                # blend = deepcopy(image_views[reference_name][filename])
                blend = np.zeros_like(image_views[reference_name][filename])

                # if signal_name in ['PIN1','PI','PIN1-PI']:
                #     # blend = deepcopy(image_views[membrane_name][filename]) 
                #     blend = np.zeros_like(image_views[membrane_name][filename])
                # else:
                #     blend = deepcopy(image_views[reference_name][filename])
                #     # blend = deepcopy(1.-image_views[reference_name][filename])
                #     # blend = np.ones_like(image_views[reference_name][filename])

                for signal_name in signals_to_display:

                    if signal_name in ['PIN1','PI','PIN1-PI']:
                        # blend = 0.6*blend + 1.0*image_views[signal_name][filename]
                        blend += image_views[signal_name][filename]
                    else:
                        # blend = np.maximum(blend,image_views[signal_name][filename])
                        blend = np.maximum(blend,0.2+1.*image_views[signal_name][filename])
                        # blend *= (0.2+1.*image_views[signal_name][filename])
                        # blend *= (1-image_views[signal_name][filename])

                        # blend *= image_views[signal_name][filename]
                        # blend = np.maximum(blend,0.4+0.8*image_views[signal_name][filename])
                        # blend = np.maximum(blend,image_views[signal_name][filename])

                # blend = np.maximum(blend,image_views[signal_name][filename])

                blend *= image_views[reference_name][filename]
                # blend = 1-blend
                # blend = 0.8*blend+0.2*image_views[reference_name][filename]
                blend = np.maximum(np.minimum(blend,1),0)


                # figure.gca().imshow(np.transpose(blend,(1,0,2))[:,::-1],extent=extent)
                figure.gca().imshow(blend,extent=extent)
                # figure.gca().set_xlim(xx.min(),xx.max())
                figure.gca().set_xlim(extent[0],extent[1])
                # figure.gca().set_ylim(yy.min(),yy.max())
                figure.gca().set_ylim(extent[3],extent[2])
                figure.gca().axis('off')

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

        figure.set_size_inches(10*len(filenames),10*(len(signal_display_list)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)
        figure.tight_layout()

        return figure


def signal_nuclei_plot(signal_data, figure=None, signal_names=None, filenames=None, registered=False, aligned=False, reference_name='TagBFP', r_max=110., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)
    
    if filenames is None:
        filenames = np.sort(signal_data.keys())

    if len(filenames)>0:
        
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals]
            signal_names.remove(reference_name)

        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer']==1]
            logging.info("".join(["  " for l in xrange(loglevel)])+"--> Plotting detected nuclei for "+filename)

            if aligned:
                X = file_data['aligned_x'].values
                Y = file_data['aligned_y'].values
                Z = file_data['aligned_z'].values
            elif registered:
                X = microscope_orientation*file_data['registered_x'].values
                Y = microscope_orientation*file_data['registered_y'].values
                Z = microscope_orientation*file_data['registered_z'].values
            else:
                X = microscope_orientation*file_data['center_x'].values
                Y = microscope_orientation*file_data['center_y'].values
                Z = microscope_orientation*file_data['center_z'].values

            # reference_img = signal_images[reference_name][filename]
            # size = np.array(reference_img.shape)
            # voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

            # if resolution is None:
            #     resolution = np.abs(voxelsize)[0]

            # xx,yy = np.meshgrid(np.linspace(0,((size-1)*voxelsize)[0],((size-1)*np.abs(voxelsize))[0]/resolution+1),np.linspace(0,((size-1)*voxelsize)[1],((size-1)*np.abs(voxelsize))[0]/resolution+1))
            # extent = xx.max(),xx.min(),yy.min(),yy.max()

            for i_signal, signal_name in enumerate(signal_names):

                figure.add_subplot(len(signal_names),len(filenames),i_signal*len(filenames)+i_time+1)

                # logging.info("".join(["  " for l in xrange(loglevel+1)])+"--> Plotting nuclei signal "+signal_name)
                figure.gca().scatter(X,Y,c=file_data[signal_name].values,s=320,linewidth=0,cmap=signal_colormaps[signal_name],vmin=signal_lut_ranges[signal_name][0],vmax=signal_lut_ranges[signal_name][1])

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

                if i_time == 0:
                    figure.gca().set_ylabel(signal_name,size=28)

                # figure.gca().set_xlim(xx.min(),xx.max())
                # figure.gca().set_ylim(yy.min(),yy.max())
                figure.gca().axis('equal')

                # figure.add_subplot(len(signal_names)+2,len(filenames),(len(signal_names)+1)*len(filenames)+i_time+1)
                    
                # nuclei_positions = dict(zip(range(len(X)),np.transpose([X,Y,np.zeros_like(X)])))
                
                # nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(xx,yy,np.zeros_like(xx))
                # confidence_map = nuclei_density  + np.maximum(1-np.linalg.norm([xx,yy],axis=0)/60.,0)
                # confidence_map = nd.gaussian_filter(confidence_map,sigma=1.0)
        
                # signal_field = "Auxin"
                
                # signal_maps = {}
                # for i_signal, signal_name in enumerate(signal_names+[signal_field]):
                #     signal_maps[signal_name] = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),file_data[signal_name].values,cell_radius=cell_radius,density_k=density_k)
                
                # figure.gca().contourf(xx,yy,signal_maps[signal_field],np.linspace(signal_ranges[signal_field][0],signal_ranges[signal_field][1],51),cmap=signal_colormaps[signal_field],alpha=1,antialiased=True,vmin=signal_lut_ranges[signal_field][0],vmax=signal_lut_ranges[signal_field][1])
                # figure.gca().contour(xx,yy,signal_maps[signal_field],np.linspace(signal_ranges[signal_field][0],signal_ranges[signal_field][1],101),cmap='gray',alpha=0.1,linewidths=1,antialiased=True,vmin=-1,vmax=0)

                # for a in xrange(16):
                #     figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)

                # # CS = figure.gca().contour(xx, yy, np.linalg.norm([xx,yy],axis=0),np.linspace(0,80,9),cmap='Greys',vmin=-1,vmax=0,alpha=0.1)
                # # figure.gca().clabel(CS, inline=1, fontsize=10,alpha=0.1)

                if aligned:
                    figure.gca().set_xlim(-r_max,r_max)
                    figure.gca().set_ylim(-r_max,r_max)
                elif registered:
                    figure.gca().set_xlim(0,2*r_max)
                    figure.gca().set_ylim(0,2*r_max)
                else:
                    figure.gca().set_xlim(0,2*r_max)
                    figure.gca().set_ylim(0,2*r_max)

                # figure.gca().axis('off')

                # figure.add_subplot(len(signal_names)+2,len(filenames),(len(signal_names)+1)*len(filenames)+i_time+1)
                # figure.gca().axis('off')

        figure.set_size_inches(10*len(filenames),10*(len(signal_names)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        return figure



def signal_map_plot(signal_data, figure=None, signal_names=None, filenames=None, registered=False, aligned=False, reference_name='TagBFP', cell_radius=7.5, density_k=0.55, r_max=110., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)
    
    if filenames is None:
        filenames = np.sort(signal_data.keys())

    if figure is None:
        figure = plt.figure(0)
        figure.clf()
        figure.patch.set_facecolor('w')

    if len(filenames)>0:
        
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals and (not 'Normalized' in c)]
            signal_names.remove(reference_name)


        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer']==1]
            logging.info("".join(["  " for l in xrange(loglevel)])+"--> Plotting signal maps for "+filename)

            if aligned:
                # X = file_data['aligned_x'].values
                # Y = file_data['aligned_y'].values
                # Z = file_data['aligned_z'].values
                position_name = 'aligned'
                center = np.array([0,0])
            elif registered:
                # X = microscope_orientation*file_data['registered_x'].values
                # Y = microscope_orientation*file_data['registered_y'].values
                # Z = microscope_orientation*file_data['registered_z'].values
                position_name = 'registered'
                center = np.array([-r_max,-r_max])
            else:
                # X = microscope_orientation*file_data['center_x'].values
                # Y = microscope_orientation*file_data['center_y'].values
                # Z = microscope_orientation*file_data['center_z'].values
                position_name = 'center'
                center = np.array([-r_max,-r_max])

            # reference_img = signal_images[reference_name][filename]
            # size = np.array(reference_img.shape)
            # voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

            # if resolution is None:
            #     resolution = np.abs(voxelsize)[0]

            # if aligned:
            #     xx,yy = np.meshgrid(np.linspace(-r_max,r_max,2*r_max+1),np.linspace(-r_max,r_max,2*r_max+1))
            # else:
            #     xx,yy = np.meshgrid(np.linspace(0,2*r_max,2*r_max+1),np.linspace(0,2*r_max,2*r_max+1))


            # xx,yy = np.meshgrid(np.linspace(0,((size-1)*voxelsize)[0],((size-1)*np.abs(voxelsize))[0]/resolution+1),np.linspace(0,((size-1)*voxelsize)[1],((size-1)*np.abs(voxelsize))[0]/resolution+1))
            # extent = xx.max(),xx.min(),yy.min(),yy.max()
            signal_map = SignalMap(file_data, position_name=position_name, extent=r_max, origin=center, polar=False, radius=cell_radius, density_k=density_k)
            
            # nuclei_positions = dict(zip(range(len(X)),np.transpose([X,Y,np.zeros_like(X)])))
            
            # nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(xx,yy,np.zeros_like(xx))
            # confidence_map = nuclei_density  + np.maximum(1-np.linalg.norm([xx,yy],axis=0)/60.,0)
            # confidence_map = nd.gaussian_filter(confidence_map,sigma=1.0)
    
            # # signal_field = "Auxin"
            
            # signal_maps = {}
            # for i_signal, signal_name in enumerate(signal_names):
            #     signal_maps[signal_name] = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),file_data[signal_name].values,cell_radius=cell_radius,density_k=density_k)
            

            for i_signal, signal_name in enumerate(signal_names):

                figure.add_subplot(len(signal_names),len(filenames),i_signal*len(filenames)+i_time+1)

                plot_signal_map(signal_map, signal_name, figure, distance_rings=aligned, colormap=signal_colormaps[signal_name], signal_range=signal_ranges[signal_name], signal_lut_range=signal_lut_ranges[signal_name])
                # logging.info("".join(["  " for l in xrange(loglevel+1)])+"--> Plotting nuclei signal "+signal_name)
                # figure.gca().scatter(X,Y,c=file_data[signal_name].values,s=320,linewidth=0,cmap=signal_colormaps[signal_name],vmin=signal_lut_ranges[signal_name][0],vmax=signal_lut_ranges[signal_name][1])

                # figure.gca().contourf(xx,yy,signal_maps[signal_name],np.linspace(signal_ranges[signal_name][0],signal_ranges[signal_name][1],101),cmap=signal_colormaps[signal_name],alpha=1,antialiased=True,vmin=signal_lut_ranges[signal_name][0],vmax=signal_lut_ranges[signal_name][1])
                # figure.gca().contour(xx,yy,signal_maps[signal_name],np.linspace(signal_ranges[signal_name][0],signal_ranges[signal_name][1],101),cmap='gray',alpha=0.1,linewidths=1,antialiased=True,vmin=-1,vmax=0)

                # for a in xrange(16):
                #     figure.gca().contourf(xx,yy,confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)

                # if aligned:
                #     CS = figure.gca().contour(xx, yy, np.linalg.norm([xx,yy],axis=0),np.linspace(0,80,9),cmap='Greys',vmin=-1,vmax=0,alpha=0.1)
                #     figure.gca().clabel(CS, inline=1, fontsize=10,alpha=0.1)

                if not aligned:
                    figure.gca().set_xlim(-2*r_max,0)
                    figure.gca().set_ylim(-2*r_max,0)

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

                if i_time == 0:
                    figure.gca().set_ylabel(signal_name,size=28)

                # figure.gca().axis('equal')

                # figure.add_subplot(len(signal_names)+2,len(filenames),(len(signal_names)+1)*len(filenames)+i_time+1)


                # # CS = figure.gca().contour(xx, yy, np.linalg.norm([xx,yy],axis=0),np.linspace(0,80,9),cmap='Greys',vmin=-1,vmax=0,alpha=0.1)
                # # figure.gca().clabel(CS, inline=1, fontsize=10,alpha=0.1)

                # figure.gca().set_xlim(xx.min(),xx.max())
                # figure.gca().set_ylim(yy.min(),yy.max())

                # figure.gca().axis('off')

                # figure.add_subplot(len(signal_names)+2,len(filenames),(len(signal_names)+1)*len(filenames)+i_time+1)
                # figure.gca().axis('off')

        figure.set_size_inches(10*len(filenames),10*(len(signal_names)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

    return figure



