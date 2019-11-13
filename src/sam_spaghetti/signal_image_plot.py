import numpy as np
import pandas as pd
import scipy.ndimage as nd

import matplotlib.pyplot as plt
# import matplotlib.patches as patch
# import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize
# import matplotlib.patheffects as patheffect
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from cellcomplex.property_topomesh.utils.matplotlib_tools import mpl_draw_topomesh

from tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function
from tissue_nukem_3d.signal_map_visualization import plot_signal_map, plot_tensor_data, plot_vector_data

from tissue_paredes.utils.matplotlib_tools import mpl_draw_wall_lines

from sam_spaghetti.utils.signal_luts import signal_colormaps, signal_ranges, signal_lut_ranges, channel_colormaps, channel_ranges, quantified_signals, vector_signals, tensor_signals, vector_signal_colors

from time import time as current_time
from copy import deepcopy

import logging

def plot_image_view_blend(image_views, filename, signals_to_display, figure, extent, reference_name='TagBFP', membrane_name='PI'):
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
        if signal_name in ['PIN1', 'PI', 'PIN1-PI']:
            # blend = 0.6*blend + 1.0*image_views[signal_name][filename]
            # blend += image_views[signal_name][filename]
            blend = np.maximum(blend, 0.6 * blend + 1. * image_views[signal_name][filename])
        else:
            # blend = np.maximum(blend,image_views[signal_name][filename])
            blend = np.maximum(blend, 0.2 + 1. * image_views[signal_name][filename])
            # blend *= (0.2+1.*image_views[signal_name][filename])
            # blend *= (1-image_views[signal_name][filename])

            # blend *= image_views[signal_name][filename]
            # blend = np.maximum(blend,0.4+0.8*image_views[signal_name][filename])
            # blend = np.maximum(blend,image_views[signal_name][filename])

    # blend = np.maximum(blend,image_views[signal_name][filename])

    if np.any([signal_name in ['PIN1', 'PI', 'PIN1-PI'] for signal_name in signals_to_display]):
        blend *= 0.6 + 1.0*image_views[membrane_name][filename]
    elif np.any(["_seg" in signal_name for signal_name in signals_to_display]):
        blend *= 1
    else:
        blend *= image_views[reference_name][filename]

    # blend = 1-blend
    # blend = 0.8*blend+0.2*image_views[reference_name][filename]
    blend = np.maximum(np.minimum(blend, 1), 0)

    figure.gca().imshow(np.transpose(blend, (1, 0, 2))[::-1], extent=extent)
    # figure.gca().imshow(blend, extent=extent)
    figure.gca().axis('off')


def signal_image_plot(image_slices, figure=None, signal_names=None, filenames=None, aligned=False, filtering=False, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=110., microscope_orientation=-1, save_image_views=True, verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if signal_names is None:
        signal_names = image_slices.keys()
    logging.info("".join(["  " for l in range(loglevel)])+"--> Plotting signal images "+str(signal_names))

    assert reference_name in signal_names

    if filenames is None:
        filenames = np.sort(list(image_slices[reference_name].keys()))
    

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

            logging.info("".join(["  " for l in range(loglevel)])+"--> Creating 2D Views : "+filename+" "+str(signal_names))
            for i_signal, signal_name in enumerate(signal_names):

                if not "_seg" in  signal_name:
                    if projection_type == "L1_slice":
                        norm = Normalize(vmin=channel_ranges[signal_name][0],vmax=channel_ranges[signal_name][1])
                    elif projection_type == "max_intensity":
                        norm = Normalize(vmin=signal_ranges[signal_name][0],vmax=signal_ranges[signal_name][1])
                    image_views[signal_name][filename] = cm.ScalarMappable(cmap=signal_colormaps[signal_name], norm=norm).to_rgba(image_slices[signal_name][filename])
                else:
                    norm = Normalize(0,255)
                    image_views[signal_name][filename] = cm.ScalarMappable(cmap='glasbey', norm=norm).to_rgba(image_slices[signal_name][filename]%256)

        signal_display_list = []
        signal_display_list += [[s] for s in signal_names if not "_seg" in s]
        for s1 in signal_names:
            if s1 != reference_name and not "_seg" in s1:
                signal_display_list += [[s1,s2] for s2 in signal_names if (s2 < s1) and (s2 != reference_name and not "_seg" in s2)]
        all_signal_list = [s for s in signal_names if (s != reference_name and not "_seg" in s)]
        if len(all_signal_list)>2:
            signal_display_list += [all_signal_list]
        signal_display_list += [[s] for s in signal_names if "_seg" in s]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):

            # blend = np.zeros_like(image_views[reference_name][filename])

            for i_signal, signals_to_display in enumerate(signal_display_list):

                figure.add_subplot(len(signal_display_list),len(filenames),i_signal*len(filenames)+i_time+1)

                plot_image_view_blend(image_views, filename, signals_to_display, figure, extent, reference_name=reference_name)

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

        figure.set_size_inches(10*len(filenames),10*(len(signal_display_list)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)
        figure.tight_layout()

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            for i_signal, signals_to_display in enumerate(signal_display_list):
                figure.add_subplot(len(signal_display_list),len(filenames),i_signal*len(filenames)+i_time+1)
                # figure.gca().set_xlim(xx.min(),xx.max())
                figure.gca().set_xlim(extent[0],extent[1])
                # figure.gca().set_ylim(yy.min(),yy.max())
                figure.gca().set_ylim(extent[3],extent[2])

        return figure


def signal_image_primordium_plot(image_primordia_slices, figure=None, signal_names=None, filenames=None, primordium=0, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=120., microscope_orientation=-1, save_image_views=True, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if signal_names is None:
        signal_names = image_primordia_slices.keys()
    logging.info("".join(["  " for l in range(loglevel)]) + "--> Plotting signal images " + str(signal_names))

    assert reference_name in signal_names
    assert primordium in image_primordia_slices[reference_name].keys()

    if filenames is None:
        filenames = np.sort(list(image_primordia_slices[reference_name][primordium].keys()))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        image_views = {}
        for signal_name in signal_names:
            image_views[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            reference_img = image_primordia_slices[reference_name][primordium][filename]
            size = np.array(reference_img.shape)
            voxelsize = np.array(reference_img.voxelsize)
            img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.

            if resolution is None:
                resolution = np.abs(voxelsize)[0]

            img_z = np.arange(size[1]) * voxelsize[1] - img_center[1]
            img_r = np.arange(r_max/resolution) * resolution
            rr, zz = map(np.transpose,np.meshgrid(img_r,img_z))
            extent = rr.min(), rr.max(), zz.max(), zz.min()

            print(rr.shape,zz.shape,reference_img.shape)

            logging.info("".join(["  " for l in range(loglevel)]) + "--> Creating 2D Views : " + filename + " " + str(signal_names))
            for i_signal, signal_name in enumerate(signal_names):
                if signal_name in ['PI','PIN1']:
                    norm = Normalize(vmin=channel_ranges[signal_name][0],vmax=channel_ranges[signal_name][1])
                else:
                    norm = Normalize(vmin=signal_ranges[signal_name][0], vmax=signal_ranges[signal_name][1])
                image_views[signal_name][filename] = cm.ScalarMappable(cmap=signal_colormaps[signal_name], norm=norm).to_rgba(image_primordia_slices[signal_name][primordium][filename])


        signal_display_list = []
        signal_display_list += [[s] for s in signal_names]
        for s1 in signal_names:
            if s1 != reference_name:
                signal_display_list += [[s1, s2] for s2 in signal_names if (s2 < s1) and (s2 != reference_name)]
        signal_display_list += [[s for s in signal_names if s != reference_name]]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            # blend = np.zeros_like(image_views[reference_name][filename])

            for i_signal, signals_to_display in enumerate(signal_display_list):

                figure.add_subplot(len(signal_display_list), len(filenames), i_signal * len(filenames) + i_time + 1)

                plot_image_view_blend(image_views, filename, signals_to_display, figure, extent, reference_name=reference_name)

                if i_signal == 0:
                    figure.gca().set_title("t=" + str(time) + "h", size=28)

        figure.set_size_inches(10 * len(filenames), 6 * (len(signal_display_list)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0, hspace=0)
        figure.tight_layout()

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):
            for i_signal, signals_to_display in enumerate(signal_display_list):
                figure.add_subplot(len(signal_display_list), len(filenames), i_signal * len(filenames) + i_time + 1)
                # figure.gca().set_xlim(xx.min(),xx.max())
                figure.gca().set_xlim(extent[0], extent[1])
                # figure.gca().set_ylim(yy.min(),yy.max())
                figure.gca().set_ylim(extent[3], extent[2])

        return figure


def signal_image_all_primordia_plot(image_primordia_slices, figure=None, signal_names=None, filenames=None, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=120., microscope_orientation=-1, save_image_views=True, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if signal_names is None:
        signal_names = image_primordia_slices.keys()
    logging.info("".join(["  " for l in range(loglevel)]) + "--> Plotting signal images " + str(signal_names))

    assert reference_name in signal_names

    if filenames is None:
        filenames = np.sort(np.unique(np.concatenate([image_primordia_slices[reference_name][p].keys() for p in image_primordia_slices[reference_name].keys()])))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        file_primordia = [[(p,f) for f in np.sort(list(image_primordia_slices[reference_name][p].keys()))] for p in np.sort(list(image_primordia_slices[reference_name].keys()))]
        file_primordia = np.concatenate([p for p in file_primordia if len(p)>0])

        image_views = {}
        for signal_name in signal_names:
            image_views[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            for primordium in image_primordia_slices[reference_name].keys():

                if filename in image_primordia_slices[reference_name][primordium].keys():

                    reference_img = image_primordia_slices[reference_name][primordium][filename]
                    size = np.array(reference_img.shape)
                    voxelsize = np.array(reference_img.voxelsize)
                    img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.

                    if resolution is None:
                        resolution = np.abs(voxelsize)[0]

                    img_z = np.arange(size[1]) * voxelsize[1] - img_center[1]
                    img_r = np.arange(r_max/resolution) * resolution
                    rr, zz = map(np.transpose,np.meshgrid(img_r,img_z))
                    extent = rr.min(), rr.max(), zz.max(), zz.min()

                    print(rr.shape,zz.shape,reference_img.shape)

                    logging.info("".join(["  " for l in range(loglevel)]) + "--> Creating 2D Views : " + filename + " P" + str(primordium) + " " + str(signal_names))
                    for i_signal, signal_name in enumerate(signal_names):
                        if signal_name in ['PI', 'PIN1']:
                            norm = Normalize(vmin=channel_ranges[signal_name][0], vmax=channel_ranges[signal_name][1])
                        else:
                            norm = Normalize(vmin=signal_ranges[signal_name][0], vmax=signal_ranges[signal_name][1])
                        image_views[signal_name][(str(primordium),filename)] = cm.ScalarMappable(cmap=signal_colormaps[signal_name], norm=norm).to_rgba(image_primordia_slices[signal_name][primordium][filename])

        signal_display_list = []
        signal_display_list += [[s] for s in signal_names]
        for s1 in signal_names:
            if s1 != reference_name:
                signal_display_list += [[s1, s2] for s2 in signal_names if (s2 < s1) and (s2 != reference_name)]
        signal_display_list += [[s for s in signal_names if s != reference_name]]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_p, (primordium, filename) in enumerate(file_primordia):

            time = file_times[filenames==filename][0]

            # blend = np.zeros_like(image_views[reference_name][filename])

            for i_signal, signals_to_display in enumerate(signal_display_list):

                figure.add_subplot(len(signal_display_list), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)

                plot_image_view_blend(image_views, (primordium,filename), signals_to_display, figure, extent, reference_name='TagBFP')

                if i_signal == 0:
                    figure.gca().set_title("P"+str(primordium)+" t=" + str(time) + "h", size=28)

        figure.set_size_inches(10 * len(file_primordia), 6 * (len(signal_display_list)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0, hspace=0)
        figure.tight_layout()

        for i_p, (primordium, filename) in enumerate(file_primordia):
            for i_signal, signals_to_display in enumerate(signal_display_list):
                figure.add_subplot(len(signal_display_list), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)
                # figure.gca().set_xlim(xx.min(),xx.max())
                figure.gca().set_xlim(extent[0], extent[1])
                # figure.gca().set_ylim(yy.min(),yy.max())
                figure.gca().set_ylim(extent[3], extent[2])

        return figure


def signal_nuclei_plot(signal_data, figure=None, signal_names=None, filenames=None, normalized=True, registered=False, aligned=False, reference_name='TagBFP', r_max=110., microscope_orientation=-1, markersize=320., alpha=1., verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)
    
    if filenames is None:
        filenames = np.sort(list(signal_data.keys()))

    if len(filenames)>0:
        
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            if normalized:
                signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals]
            else:
                signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals and (not 'Normalized' in c)]
            # signal_names.remove(reference_name)

        vector_names = [c for c in signal_names if c in vector_signals]
        vector_names = [c for c in vector_names if (c in signal_data[filenames[0]].columns)
                                                or np.all([c+"_"+dim in signal_data[filenames[0]].columns for dim in ['x','y']])]

        tensor_names = [c for c in signal_names if c in tensor_signals]

        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        signal_names += vector_names
        signal_names += tensor_names

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer']==1]
            logging.info("".join(["  " for l in range(loglevel)])+"--> Plotting detected nuclei for "+filename)

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

                figure.add_subplot(len(signal_names), len(filenames),i_signal*len(filenames)+i_time+1)

                if signal_name in tensor_names:
                    plot_tensor_data(figure,X,Y,file_data[signal_name].values, np.ones_like(X), tensor_style='crosshair',colormap='gray',value_range=(0,0),scale=10.)
                    plot_tensor_data(figure,X,Y,file_data[signal_name].values, np.ones_like(X), tensor_style='ellipse',colormap='gray',value_range=(0,0),scale=10.,alpha=0.2)
                    col = None
                elif signal_name in vector_names:
                    if signal_name in file_data.columns:
                        vectors = file_data[signal_name].values
                    else:
                        vectors = file_data[[signal_name+"_"+dim for dim in ['x','y','z']]].values
                        figure.gca().quiver(X, Y, vectors[:, 0], vectors[:, 1], color=vector_signal_colors[signal_name], units='xy', scale=1.)
                    col = None
                else:
                    # logging.info("".join(["  " for l in range(loglevel+1)])+"--> Plotting nuclei signal "+signal_name)
                    col = figure.gca().scatter(X,Y,c=file_data[signal_name].values,s=markersize,linewidth=0,alpha=alpha,cmap=signal_colormaps[signal_name],vmin=signal_lut_ranges[signal_name][0],vmax=signal_lut_ranges[signal_name][1])

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

                if i_time == 0:
                    figure.gca().set_ylabel(signal_name,size=28)
                else:
                    figure.gca().set_yticklabels([])

                figure.gca().axis('equal')

                if i_time == len(filenames)-1:
                    if col is not None:
                        cax = inset_axes(figure.gca(),width="3%", height="25%", loc='lower right')
                        cbar = figure.colorbar(col, cax=cax, pad=0.)
                        cax.yaxis.set_ticks_position('left')
                        cbar.set_clim(*signal_lut_ranges[signal_name])

        figure.set_size_inches(10*len(filenames),10*(len(signal_names)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            for i_signal, signal_name in enumerate(signal_names):
                figure.add_subplot(len(signal_names), len(filenames), i_signal * len(filenames) + i_time + 1)


                if aligned:
                    figure.gca().set_xlim(-r_max, r_max)
                    figure.gca().set_ylim(-r_max, r_max)
                elif registered:
                    figure.gca().set_xlim(0, 2 * r_max)
                    figure.gca().set_ylim(0, 2 * r_max)
                else:
                    figure.gca().set_xlim(0, 2 * r_max)
                    figure.gca().set_ylim(0, 2 * r_max)

        return figure


def signal_nuclei_all_primordia_plot(primordia_signal_data, figure=None, signal_names=None, filenames=None, normalized=True, reference_name='TagBFP', r_max=60., microscope_orientation=-1, markersize=480., alpha=1., verbose=False, debug=False, loglevel=0):

    primordia_range = np.sort(list(primordia_signal_data.keys()))

    if filenames is None:
        filenames = np.sort(np.unique(np.concatenate([primordia_signal_data[p].keys() for p in primordia_range])))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        signal_names = ["Auxin"]
        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            if normalized:
                signal_names = [c for c in primordia_signal_data.values()[0].values()[0].columns if c in quantified_signals]
            else:
                signal_names = [c for c in primordia_signal_data.values()[0].values()[0].columns if c in quantified_signals and (not 'Normalized' in c)]
            # signal_names.remove(reference_name)

        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        file_primordia = [[(p, f) for f in np.sort(list(primordia_signal_data[p].keys())) if (f in filenames) and ("t00" in f) and (p>-2)] for p in np.sort(list(primordia_signal_data.keys()))]
        # file_primordia = [[(str(p), f) for f in np.sort([f for primordium,f in primordia_signal_data.keys() if (primordium==str(p)) and (f in filenames) and ("t00" in f) and (p>-2)]) ] for p in primordia_range]
        file_primordia = np.concatenate([p for p in file_primordia if len(p) > 0])

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_p, (primordium, filename) in enumerate(file_primordia):
            if filename in primordia_signal_data[int(primordium)].keys():
                file_primordium_data = primordia_signal_data[int(primordium)][filename]
                time = file_times[np.array(filenames) == filename][0]
                i_time = np.arange(len(filenames))[np.array(filenames) == filename][0]

                for i_signal, signal_name in enumerate(signal_names):

                    figure.add_subplot(len(signal_names), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)
                    col = figure.gca().scatter(file_primordium_data['radial_distance'].values, file_primordium_data['aligned_z'].values, c=file_primordium_data[signal_name].values, s=markersize, linewidth=0, alpha=alpha, cmap=signal_colormaps[signal_name], vmin=signal_lut_ranges[signal_name][0], vmax=signal_lut_ranges[signal_name][1])

                    if i_signal == 0:
                        figure.gca().set_title("P"+str(primordium)+" t=" + str(time) + "h", size=28)

                    if i_p == 0:
                        figure.gca().set_ylabel(signal_name, size=28)

                    figure.gca().axis('equal')

                    if False:
                    # if i_time == len(filenames) - 1:
                        cax = inset_axes(figure.gca(),width="3%", height="25%", loc='lower right')
                        cbar = figure.colorbar(col, cax=cax, pad=0.)
                        cax.yaxis.set_ticks_position('left')
                        cbar.set_clim(*signal_lut_ranges[signal_name])


        figure.set_size_inches(8*len(file_primordia),6*len(signal_names))
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        for i_p, (primordium, filename) in enumerate(file_primordia):
            for i_signal, signal_name in enumerate(signal_names):
                figure.add_subplot(len(signal_names), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)
                figure.gca().set_xlim(0, r_max)
                figure.gca().set_ylim(-0.5*r_max, 0.1*r_max)
                figure.gca().set_xticklabels([str(int(r)) for r in figure.gca().get_xticks()],size=20)
                if i_p != len(file_primordia)-1:
                    figure.gca().axis('off')

        return figure


def signal_wall_plot(wall_topomeshes, figure=None, signal_names=None, filenames=None, normalized=True, registered=False, aligned=False, reference_name='PI', r_max=110., microscope_orientation=-1, linewidth=1., alpha=1., verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)
    
    if filenames is None:
        filenames = np.sort(list(wall_topomeshes.keys()))

    if len(filenames)>0:
        
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            if normalized:
                signal_names = [c for c in wall_topomeshes[filenames[0]].wisp_property_names(3) if c in quantified_signals]
            else:
                signal_names = [c for c in wall_topomeshes[filenames[0]].wisp_property_names(3) if c in quantified_signals and (not 'Normalized' in c)]
            # signal_names.remove(reference_name)

        vector_names = [c for c in signal_names if c in vector_signals]
        tensor_names = [c for c in signal_names if c in tensor_signals]

        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        signal_names += vector_names
        signal_names += tensor_names

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            wall_topomesh = wall_topomeshes[filename]

            for i_signal, signal_name in enumerate(signal_names):

                figure.add_subplot(len(signal_names), len(filenames),i_signal*len(filenames)+i_time+1)

                if signal_name in vector_names:
                    mpl_draw_wall_lines(wall_topomesh, figure=figure, colormap='gray', value_range=(0,0), wall_scale=1/12., linewidth=1)
                    mpl_draw_topomesh(wall_topomesh, figure, 3, property_name=signal_name, linewidth=linewidth, color=vector_signal_colors[signal_name])
                else:
                    mpl_draw_wall_lines(wall_topomesh, figure=figure, property_name=signal_name, colormap=signal_colormaps[signal_name], value_range=signal_lut_ranges[signal_name], wall_scale=1/12., linewidth=linewidth)

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

                if i_time == 0:
                    figure.gca().set_ylabel(signal_name,size=28)
                else:
                    figure.gca().set_yticklabels([])

                figure.gca().axis('equal')

        figure.set_size_inches(10*len(filenames),10*(len(signal_names)))
        # figure.set_size_inches(5*len(filenames),5)
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            for i_signal, signal_name in enumerate(signal_names):
                figure.add_subplot(len(signal_names), len(filenames), i_signal * len(filenames) + i_time + 1)

                if aligned:
                    figure.gca().set_xlim(-r_max, r_max)
                    figure.gca().set_ylim(-r_max, r_max)
                elif registered:
                    figure.gca().set_xlim(0, 2 * r_max)
                    figure.gca().set_ylim(0, 2 * r_max)
                else:
                    figure.gca().set_xlim(0, 2 * r_max)
                    figure.gca().set_ylim(0, 2 * r_max)

        return figure
    

def signal_map_plot(signal_maps, figure=None, signal_names=None, filenames=None, registered=False, aligned=False, r_max=110., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)
    
    if filenames is None:
        filenames = np.sort(list(signal_maps.keys()))

    if figure is None:
        figure = plt.figure(0)
        figure.clf()
        figure.patch.set_facecolor('w')

    if len(filenames)>0:
        
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            signal_names = list(signal_maps.values())[0].signal_names()
        vector_names = [c for c in signal_names if c in vector_signals]
        tensor_names = [c for c in signal_names if c in tensor_signals]

        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        signal_names += tensor_names


        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            logging.info("".join(["  " for l in range(loglevel)]) + "--> Plotting signal maps for " + filename)
            signal_map = signal_maps[filename]

            for i_signal, signal_name in enumerate(signal_names):

                figure.add_subplot(len(signal_names),len(filenames),i_signal*len(filenames)+i_time+1)

                if signal_name in tensor_names:
                    plot_signal_map(signal_map, signal_name, figure, distance_rings=aligned, colormap='gray', signal_range=(0,0), signal_lut_range=(0,0))
                    col = None
                else:
                    col = plot_signal_map(signal_map, signal_name, figure, distance_rings=aligned, colormap=signal_colormaps[signal_name], signal_range=signal_ranges[signal_name], signal_lut_range=signal_lut_ranges[signal_name])

                if i_signal == 0:
                    figure.gca().set_title("t="+str(time)+"h",size=28)

                if i_time == 0:
                    figure.gca().set_ylabel(signal_name,size=28)
                else:
                    figure.gca().set_yticklabels([])

                figure.gca().axis('on')

                if i_time == len(filenames)-1:
                    if col is not None:
                        cax = inset_axes(figure.gca(),width="3%", height="25%", loc='lower right')
                        cbar = figure.colorbar(col, cax=cax, pad=0.)
                        cax.yaxis.set_ticks_position('left')
                        cbar.set_clim(*signal_lut_ranges[signal_name])


        figure.set_size_inches(10*len(filenames),10*(len(signal_names)))
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            for i_signal, signal_name in enumerate(signal_names):
                figure.add_subplot(len(signal_names), len(filenames), i_signal * len(filenames) + i_time + 1)

                if not aligned:
                    figure.gca().set_xlim(0, microscope_orientation*2 * r_max)
                    figure.gca().set_ylim(0, microscope_orientation*2 * r_max)

    return figure


def signal_map_all_primordia_plot(primordia_signal_maps, figure=None, signal_names=None, filenames=None, normalized=True, reference_name='TagBFP', cell_radius=7.5, density_k=0.55, r_max=60., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    primordia_range = np.sort(np.unique([int(p) for p,_ in primordia_signal_maps.keys()]))

    if filenames is None:
        filenames = np.sort(np.unique([f for _,f in primordia_signal_maps.keys()]))

    file_primordia = [[(str(p), f) for f in np.sort([f for primordium,f in primordia_signal_maps.keys() if (primordium==str(p)) and (f in filenames) and ("t00" in f) and (p>-2)]) ] for p in primordia_range]
    file_primordia = np.concatenate([p for p in file_primordia if len(p) > 0])

    if len(file_primordia) > 0:
        filenames = np.sort(np.unique([f for _,f in file_primordia]))
        file_times = np.array([int(f[-2:]) for f in filenames])

        signal_names = ["Auxin"]
        if signal_names is None:
            signal_names = primordia_signal_maps.values()[0].signal_names()
        signal_names = [c for c in signal_names if c in signal_colormaps]
        signal_names = [c for c in signal_names if c in signal_lut_ranges]

        if figure is None:
            figure = plt.figure(0)
            figure.clf()
            figure.patch.set_facecolor('w')

        for i_p, (primordium, filename) in enumerate(file_primordia):
            logging.info("".join(["  " for l in range(loglevel)]) + "--> Plotting P"+str(primordium)+" signal maps for " + filename)

            time = file_times[filenames == filename][0]
            i_time = np.arange(len(filenames))[filenames == filename][0]

            signal_map = primordia_signal_maps[(primordium, filename)]

            for i_signal, signal_name in enumerate(signal_names):

                figure.add_subplot(len(signal_names), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)

                col = plot_signal_map(signal_map, signal_name, figure, distance_rings=False, colormap=signal_colormaps[signal_name], signal_range=signal_ranges[signal_name],signal_lut_range=signal_lut_ranges[signal_name])

                if i_signal == 0:
                    figure.gca().set_title("P"+str(primordium)+" t=" + str(time) + "h", size=28)

                if i_p == 0:
                    figure.gca().set_ylabel(signal_name, size=28)

                figure.gca().axis('on')

                # if i_time == len(filenames)-1:
                if False:
                    cax = inset_axes(figure.gca(),width="3%", height="25%", loc='lower right')
                    cbar = figure.colorbar(col, cax=cax, pad=0.)
                    cax.yaxis.set_ticks_position('left')
                    cbar.set_clim(*signal_lut_ranges[signal_name])

        figure.set_size_inches(8*len(file_primordia),6*len(signal_names))
        figure.tight_layout()
        figure.subplots_adjust(wspace=0,hspace=0)

        for i_p, (primordium, filename) in enumerate(file_primordia):
            for i_signal, signal_name in enumerate(signal_names):
                figure.add_subplot(len(signal_names), len(file_primordia), i_signal * len(file_primordia) + i_p + 1)
                figure.gca().set_xlim(0, r_max)
                figure.gca().set_ylim(-0.5*r_max, 0.1*r_max)
                figure.gca().axis('off')

        return figure


