import numpy as np
import pandas as pd
import os

import sam_spaghetti

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
    if not 'Name' in nomenclature_data.columns:
        nomenclature_data = pd.read_csv(nomenclature_file,sep=';')
    nomenclature_names = dict(zip(nomenclature_data['Name'],nomenclature_data['Nomenclature Name']))
    # print czi_filename
    return nomenclature_names.get(czi_filename,None)

def get_sequence_orientation(sequence_name, dirname=sam_spaghetti_dirname):
    orientation_file = dirname + "/nuclei_image_sam_orientation.csv"
    orientation_data = pd.read_csv(orientation_file,sep=",")
    if not 'sequence_name' in orientation_data.columns:
        orientation_data = pd.read_csv(orientation_file,sep=";")
    if sequence_name in orientation_data['sequence_name'].values:
        meristem_orientation = int(orientation_data[orientation_data['sequence_name']==sequence_name]['orientation'])
        return meristem_orientation
    else:
        raise(KeyError("No SAM orientation information could be found for sequence "+str(sequence_name)))
