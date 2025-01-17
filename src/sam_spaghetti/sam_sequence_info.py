import numpy as np
import pandas as pd
import os

import sam_spaghetti

from sam_spaghetti.utils.signal_luts import signal_colormaps, signal_ranges, signal_lut_ranges, channel_ranges

sam_spaghetti_dirname = sam_spaghetti.__path__[0]+"/../../share/data"


def get_experiment_name(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_sequences = dict(zip(experiment_data['experiment'],experiment_data['experiment_name']))
    return experiment_sequences.get(exp,"")


def get_experiment_microscopy(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_data = experiment_data.replace(np.nan,"")
    experiment_microscopy = dict(zip(experiment_data['experiment'],experiment_data['microscopy_directory']))
    return experiment_microscopy.get(exp,"")


def get_experiment_channels(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_data = experiment_data.replace(np.nan,"")
    experiment_channels = dict(zip(experiment_data['experiment'],experiment_data['channel_names']))
    channels = experiment_channels.get(exp)
    return eval(channels) if channels != "" else None


def get_experiment_signals(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_data = experiment_data.replace(np.nan,"")
    experiment_signals = dict(zip(experiment_data['experiment'],experiment_data['signal_names']))
    signals = experiment_signals.get(exp)
    return eval(signals) if signals != "" else None


def get_experiment_reference(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_data = experiment_data.replace(np.nan,"")
    experiment_references = dict(zip(experiment_data['experiment'],experiment_data['reference_name']))
    reference = experiment_references.get(exp)
    return reference if reference != "" else None


def get_experiment_microscope_orientation(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_data = experiment_data.replace(np.nan,"")
    micoscope_orientation = ""
    if 'microscope_orientation' in experiment_data.columns:
        experiment_micoscope_orientations = dict(zip(experiment_data['experiment'],experiment_data['microscope_orientation']))
        micoscope_orientation = experiment_micoscope_orientations.get(exp)
    return micoscope_orientation if micoscope_orientation != "" else -1


def get_nomenclature_name(czi_file, dirname=sam_spaghetti_dirname):
    czi_filename = os.path.split(czi_file)[1]
    nomenclature_file = dirname+"/nomenclature.csv"
    nomenclature_data = pd.read_csv(nomenclature_file,sep=',')
    if not 'filename' in nomenclature_data.columns:
        nomenclature_data = pd.read_csv(nomenclature_file,sep=';')
    if 'filename' in nomenclature_data.columns:
        nomenclature_names = [get_experiment_name(exp,dirname)+"_sam"+str(sam_id).zfill(2)+"_t"+str(t).zfill(2) for exp,sam_id,t in nomenclature_data[['experiment','sam_id','hour_time']].values]
        nomenclature_names = dict(zip(nomenclature_data['filename'],nomenclature_names))
        # print czi_filename
        return nomenclature_names.get(czi_filename,None)
    else:
        return


def get_sequence_orientation(sequence_name, dirname=sam_spaghetti_dirname):
    orientation_file = dirname + "/nuclei_image_sam_orientation.csv"
    orientation_data = pd.read_csv(orientation_file,sep=",")
    if not 'experiment' in orientation_data.columns:
        orientation_data = pd.read_csv(orientation_file,sep=";")
    orientation_data['sequence_name'] = [get_experiment_name(exp,dirname)+"_sam"+str(sam_id).zfill(2) for exp,sam_id in orientation_data[['experiment','sam_id']].values]
    if sequence_name in orientation_data['sequence_name'].values:
        meristem_orientation = int(orientation_data[orientation_data['sequence_name']==sequence_name]['orientation'])
        return meristem_orientation
    else:
        raise(KeyError("No SAM orientation information could be found for sequence "+str(sequence_name)))


def update_lut_ranges(dirname=sam_spaghetti_dirname):
    lut_file = dirname+"/signal_ranges.csv"
    if os.path.exists(lut_file):
        lut_data = pd.read_csv(lut_file,sep=',')
        if not 'signal_name' in lut_data.columns:
            lut_data = pd.read_csv(lut_file,sep=";")
        print(lut_data)
        signal_colormaps.update(dict([(s,c) for s,c in lut_data[['signal_name','colormap']].values if not pd.isnull(c)]))
        signal_ranges.update(dict([(s,eval(r)) for s,r in lut_data[['signal_name','signal_range']].values if not pd.isnull(r)]))
        signal_lut_ranges.update(dict([(s,eval(r)) for s,r in lut_data[['signal_name','color_range']].values if not pd.isnull(r)]))
        channel_ranges.update(dict([(s,eval(r)) for s,r in lut_data[['signal_name','channel_range']].values if not pd.isnull(r)]))

