import numpy as np
import pandas as pd

from vplants.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal

from sam_spaghetti.sam_sequence_info import get_experiment_name
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_data, load_sequence_primordia_data

import os
import logging

max_sam_id = 100
max_time = 100


def compile_signal_data(experiments, image_dirname, data_dirname=None, save_files=True, normalize_data=True, aligned=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if data_dirname is None:
        data_dirname = image_dirname+"/../../"

    data_list = []

    for exp in experiments:
        for sam_id in xrange(max_sam_id):

            sequence_name = get_experiment_name(exp, data_dirname) + "_sam" + str(sam_id).zfill(2)
            sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=aligned, aligned=aligned, verbose=verbose, debug=debug, loglevel=loglevel)

            for filename in sequence_data:
                data = sequence_data[filename]
                time = int(filename[-2:])
                # data = pd.read_csv(data_file)
            
                if not aligned:
                    if 'qDII' in data.columns:
                        data['Auxin'] = 1 - data['qDII']
                    if 'qRGA' in data.columns:
                        data['Gibberelins'] = 1 - data['qRGA']

                    if normalize_data:
                        for signal in ['CLV3']:
                            if signal in data.columns:
                                #signal_data  = data[signal][data['model_coords_r']<2.*r_max/3.]
                                signal_data  = data[signal][data['layer']==1]
                                data['Normalized_'+signal] = data[signal]/signal_data.mean()
                                clv3_threshold = 1.2
                                #data['Normalized_'+signal] = data[signal]/max_clv3

                        for signal in ['DIIV','qDII','Auxin','RGAV','qRGA','Gibberelins','DR5','TagBFP']:
                            if signal in data.columns:
                                signal_data  = data[signal][data['layer']==1]
                                data['Normalized_'+signal] = 0.5 + 0.2*(data[signal]-signal_data.mean())/(signal_data.std())
                    
                    data['filename'] = [filename for i in xrange(len(data))]
                    data['experiment'] = [exp for i in xrange(len(data))]
                    data['sam_id'] = [sam_id for i in xrange(len(data))]
                    data['hour_time'] = [time for i in xrange(len(data))]
                    data['growth_condition'] = ["LD" if "LD" in filename else "SD" for i in xrange(len(data))]
                    
                    data['short_name'] = [exp+'_Sam'+str(sam_id) for i in xrange(len(data))]
                    
                    if save_files:
                        data.to_csv(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_normalized_signal_data.csv",index=False) 
                else:
                    data['radial_distance'] = np.linalg.norm([data['aligned_x'],data['aligned_y']],axis=0)
                    data['aligned_theta'] = 180./np.pi*np.sign(data['aligned_y'])*np.arccos(data['aligned_x']/data['radial_distance'])
                    
                    distance_rank = dict(zip(np.sort(data['radial_distance']),np.arange(len(data))))
                    data['nuclei_count'] = [distance_rank[d] for d in data['radial_distance']]
                    data['nuclei_distance'] = np.sqrt(data['nuclei_count']/np.pi)
                
                    data['filename'] = [filename for i in xrange(len(data))]
                    data['experiment'] = [exp for i in xrange(len(data))]
                    data['sam_id'] = [sam_id for i in xrange(len(data))]
                    data['hour_time'] = [time for i in xrange(len(data))]
                    data['growth_condition'] = ["LD" if "LD" in filename else "SD" for i in xrange(len(data))]

                    if save_files:
                        data.to_csv(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_aligned_L1_normalized_signal_data.csv",index=False) 
                
                data_list += [data]

    if len(data_list)>0:
        complete_data = pd.concat(data_list)
        experiment_string = "".join([experiments[0]]+["_"+exp for exp in experiments[1:]])
        
        if save_files:
            if not aligned:
                complete_data.to_csv(data_dirname+"/"+experiment_string+"_normalized_signal_data.csv",index=False) 
            else:
                complete_data.to_csv(data_dirname+"/"+experiment_string+"_aligned_L1_normalized_signal_data.csv",index=False) 
    else:
        complete_data = pd.DataFrame()
        logging.warning("No data to compile! Nothing will be saved!")

    return complete_data

def compile_primordia_data(experiments, image_dirname, data_dirname=None, save_files=True, compute_surface_distance=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if data_dirname is None:
        data_dirname = image_dirname+"/../../"

    data_list = []

    for exp in experiments:
        for sam_id in xrange(max_sam_id):

            sequence_name = get_experiment_name(exp, data_dirname) + "_sam" + str(sam_id).zfill(2)
            sequence_primordia_data = load_sequence_primordia_data(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)
            sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=verbose, debug=debug, loglevel=loglevel)

            for filename in sequence_primordia_data:
                primordia_data = sequence_primordia_data[filename]
                time = int(filename[-2:])
                    
                primordia_data['filename'] = [filename for i in xrange(len(primordia_data))]
                primordia_data['experiment'] = [exp for i in xrange(len(primordia_data))]
                primordia_data['sam_id'] = [sam_id for i in xrange(len(primordia_data))]
                primordia_data['hour_time'] = [time for i in xrange(len(primordia_data))]
                primordia_data['growth_condition'] = ["LD" if "LD" in filename else "SD" for i in xrange(len(primordia_data))]
                primordia_data['short_name'] = [exp+'_Sam'+str(sam_id) for i in xrange(len(primordia_data))]

                if filename in sequence_data:
                    data = sequence_data[filename]

                    distance_rank = dict(zip(np.sort(data['radial_distance']),np.arange(len(data))))
                    data['nuclei_count'] = [distance_rank[d] for d in data['radial_distance']]
                    data['nuclei_distance'] = np.sqrt(data['nuclei_count']/np.pi)
                    
                    X = data['aligned_x'].values
                    Y = data['aligned_y'].values
                    Z = data['aligned_z'].values
                                    
                    for field in ['nuclei_count','nuclei_distance']:
                        primordia_data[field] = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([primordia_data['aligned_x'],primordia_data['aligned_y']]),data[field].values)
                    
                    if compute_surface_distance:
                        surface_distances = []
                        for x,y in zip(X,Y):
                            print x,y
                            radial_x = np.linspace(0,1,101)*x
                            radial_y = np.linspace(0,1,101)*y
                            radial_z = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([radial_x,radial_y]),Z)
                            radial_dz = radial_z[1:]-radial_z[:-1]
                            radial_r = np.linalg.norm([radial_x,radial_y],axis=0)
                            radial_dr = radial_r[1:]-radial_r[:-1]
                            surface_distance = np.linalg.norm([radial_dr,radial_dz],axis=0).sum()
                            surface_distances += [surface_distance]
                        
                        data['surface_distance'] = surface_distances
                        
                        data['surface_x'] = data['surface_distance']*np.cos(np.pi*data['aligned_theta']/180.)
                        data['surface_y'] = data['surface_distance']*np.sin(np.pi*data['aligned_theta']/180.)
                    
                        for field in ['surface_distance','surface_x','surface_y']:
                            primordia_data[field] = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([primordia_data['aligned_x'],primordia_data['aligned_y']]),data[field].values)
                    
                data_list += [primordia_data]

    if len(data_list)>0:
        complete_data = pd.concat(data_list)
        experiment_string = "".join([experiments[0]]+["_"+exp for exp in experiments[1:]])
        
        if save_files:
            complete_data.to_csv(data_dirname+"/"+experiment_string+"_aligned_primordia_extrema_data.csv",index=False) 
    else:
        complete_data = pd.DataFrame()
        logging.warning("No data to compile! Nothing will be saved!")

    return complete_data
