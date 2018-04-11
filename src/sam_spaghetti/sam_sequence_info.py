import pandas as pd
import os

import sam_spaghetti

sam_spaghetti_dirname = sam_spaghetti.__path__[0]+"/../../share/data"

def get_experiment_name(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_sequences = dict(zip(experiment_data['experiment'],experiment_data['experiment_name']))
    # experiment_sequences["E35"] = "qDII-PIN1-CLV3-PI-LD_E35_171110"
    return experiment_sequences.get(exp,"")

def get_experiment_microscopy(exp, dirname=sam_spaghetti_dirname):
    experiment_file = dirname+"/experiment_info.csv"
    experiment_data = pd.read_csv(experiment_file,sep=';')
    experiment_microscopy = dict(zip(experiment_data['experiment'],experiment_data['microscopy_directory']))
    # experiment_microscopy["E35"] = "20171110 MS-E35 LD qDII-CLV3-PIN1-PI"
    return experiment_microscopy.get(exp,"")

def get_nomenclature_name(czi_file, dirname=sam_spaghetti_dirname):
    czi_filename = os.path.split(czi_file)[1]
    nomenclature_file = dirname+"/nomenclature.csv"
    nomenclature_data = pd.read_csv(nomenclature_file,sep=',')
    if not 'Name' in nomenclature_data.columns:
        nomenclature_data = pd.read_csv(nomenclature_file,sep=';')
    nomenclature_names = dict(zip(nomenclature_data['Name'],nomenclature_data['Nomenclature Name']))
    # print czi_filename
    return nomenclature_names.get(czi_filename,os.path.splitext(czi_filename)[0])
