import numpy as np
import pandas as pd

from vplants.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal

from sam_spaghetti.sam_sequence_info import get_experiment_name
from sam_spaghetti.signal_image_plot import load_sequence_signal_data

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

            for time in xrange(max_time):
                
                filename = sequence_name + "_t" + str(time).zfill(2)
                
                if not aligned:
                    data_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"
                else:
                    data_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_aligned_L1_normalized_signal_data.csv"

                if os.path.exists(data_file):
                    data = pd.read_csv(data_file)
                
                    if not aligned:
                        data['Auxin'] = 1 - data['qDII']

                        if normalize_data:
                            for signal in ['CLV3']:
                                #signal_data  = data[signal][data['model_coords_r']<2.*r_max/3.]
                                signal_data  = data[signal][data['layer']==1]
                                data['Normalized_'+signal] = data[signal]/signal_data.mean()
                                clv3_threshold = 1.2
                                #data['Normalized_'+signal] = data[signal]/max_clv3
                        
                            for signal in ['DIIV','qDII','Auxin','DR5','TagBFP']:
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
                            data.to_csv(data_file,index=False) 
                    
                    data_list += [data]


    complete_data = pd.concat(data_list)
    
    if save_files:
        if not aligned:
            complete_data.to_csv(data_dirname+"/normalized_signal_data.csv",index=False) 
        else:
            complete_data.to_csv(data_dirname+"/aligned_L1_normalized_signal_data.csv",index=False) 

    return complete_data
