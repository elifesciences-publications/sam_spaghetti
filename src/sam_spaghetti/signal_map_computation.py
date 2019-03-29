import logging
from copy import deepcopy

import numpy as np

from vplants.tissue_nukem_3d.signal_map import SignalMap

from sam_spaghetti.utils.signal_luts import quantified_signals


def compute_signal_maps(signal_data, signal_names=None, filenames=None, normalized=True, registered=False, aligned=False, reference_name='TagBFP', cell_radius=7.5, density_k=0.55, r_max=110., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if filenames is None:
        filenames = np.sort(signal_data.keys())

    signal_maps = {}

    if len(filenames) > 0:

        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            if normalized:
                signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals]
            else:
                signal_names = [c for c in signal_data[filenames[0]].columns if c in quantified_signals and (not 'Normalized' in c)]
            # signal_names.remove(reference_name)

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer'] == 1]
            logging.info("".join(["  " for l in xrange(loglevel)]) + "--> Computing signal maps for " + filename)

            if aligned:
                position_name = 'aligned'
                center = np.array([0, 0])
            elif registered:
                position_name = 'registered'
                center = np.array([-r_max, -r_max])
            else:
                position_name = 'center'
                center = np.array([-r_max, -r_max])

            signal_map = SignalMap(file_data, position_name=position_name, extent=r_max, origin=center, polar=False, radius=cell_radius, density_k=density_k)
            for signal_name in signal_names:
                signal_map.compute_signal_map(signal_name)

            signal_maps[filename] = signal_map

    return signal_maps


def compute_primordia_signal_maps(primordia_signal_data, signal_names=None, filenames=None, normalized=True, reference_name='TagBFP', cell_radius=7.5, density_k=0.55, r_max=80., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    primordia_range = np.sort(primordia_signal_data.keys())

    if filenames is None:
        filenames = np.sort(np.unique(np.concatenate([primordia_signal_data[p].keys() for p in primordia_range])))

    primordia_signal_maps = {}

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        if signal_names is None:
            # signal_names = [c for c in signal_data[filenames[0]].columns if (not "center" in c) and (not "layer" in c) and (not 'Unnamed' in c) and (not "label" in c)]
            if normalized:
                signal_names = [c for c in primordia_signal_data.values()[0].values()[0].columns if c in quantified_signals]
            else:
                signal_names = [c for c in primordia_signal_data.values()[0].values()[0].columns if c in quantified_signals and (not 'Normalized' in c)]
            # signal_names.remove(reference_name)

        file_primordia = [[(p, f) for f in np.sort(primordia_signal_data[p].keys())] for p in np.sort(primordia_signal_data.keys())]
        file_primordia = np.concatenate([p for p in file_primordia if len(p) > 0])

        for i_p, (primordium, filename) in enumerate(file_primordia):
            if filename in primordia_signal_data[int(primordium)].keys():
                file_primordium_data = primordia_signal_data[int(primordium)][filename]
                time = file_times[filenames == filename][0]
                i_time = np.arange(len(filenames))[filenames == filename][0]
                logging.info("".join(["  " for l in xrange(loglevel)]) + "--> Computing P"+str(primordium)+" signal maps for " + filename)

                file_primordium_data['slice_x'] = file_primordium_data['radial_distance'].values
                file_primordium_data['slice_y'] = file_primordium_data['aligned_z'].values
                signal_map = SignalMap(file_primordium_data, position_name='slice', extent=r_max, origin=np.array([0,0]), polar=False, radius=cell_radius, density_k=density_k)
                for signal_name in signal_names:
                    signal_map.compute_signal_map(signal_name)
                primordia_signal_maps[(primordium, filename)] = signal_map

    return primordia_signal_maps


def compute_average_signal_maps(experiment_signal_maps, average_by='time', signal_names=None, filenames=None, time_range=None, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if signal_names is None:
        signal_names = experiment_signal_maps.values()[0].signal_names()

    if filenames is None:
        filenames = list(experiment_signal_maps.keys())

    if time_range is None:
        time_range = np.sort(np.unique([int(filename[-2:]) for filename in filenames]))

    average_signal_maps = {}
    if average_by == 'time':
        for time in time_range:
            average_signal_maps["t" + str(time).zfill(2)] = deepcopy(experiment_signal_maps.values()[0])

        for time in time_range:
            average_signal_maps["t" + str(time).zfill(2)].confidence_map = np.nanmean([experiment_signal_maps[filename].confidence_map for filename in filenames if int(filename[-2:]) == time], axis=0)
            for signal_name in signal_names:
                average_signal_maps["t"+str(time).zfill(2)].signal_maps[signal_name] = np.nanmean([experiment_signal_maps[filename].signal_maps[signal_name] for filename in filenames if (int(filename[-2:]) == time) and (experiment_signal_maps[filename].signal_maps.has_key(signal_name))], axis=0)

    return average_signal_maps


def compute_average_primordia_signal_maps(experiment_primordia_signal_maps, average_by='time', signal_names=None, filenames=None, primordia_range=None, time_range=None, verbose=False, debug=False, loglevel=0):

    primordia_range = np.sort(np.unique([int(p) for p,_ in experiment_primordia_signal_maps.keys()]))

    time_primordia = [[(str(p), t) for t in np.sort(np.unique([int(f[-2:]) for primordium,f in experiment_primordia_signal_maps.keys() if primordium==str(p)]))] for p in primordia_range]
    time_primordia = np.concatenate([p for p in time_primordia if len(p) > 0])

    if signal_names is None:
        signal_names = experiment_primordia_signal_maps.values()[0].signal_names()

    if time_range is None:
        time_range = np.sort(np.unique([t for _,t in time_primordia]))

    average_signal_maps = {}
    if average_by == 'time':
        for primordium, time in time_primordia:
            if time in time_range:
                average_signal_maps[(str(primordium),"t" + str(time).zfill(2))] = deepcopy(experiment_primordia_signal_maps.values()[0])

        for primordium, time in time_primordia:
            if time in time_range:
                primordium_filenames = [f for p,f in experiment_primordia_signal_maps.keys() if primordium==p]

                average_signal_maps[(str(primordium),"t" + str(time).zfill(2))].confidence_map = np.nanmean([experiment_primordia_signal_maps[(primordium,filename)].confidence_map for filename in primordium_filenames if int(filename[-2:]) == int(time)], axis=0)
                for signal_name in signal_names:
                    average_signal_maps[(str(primordium),"t"+str(time).zfill(2))].signal_maps[signal_name] = np.nanmean([experiment_primordia_signal_maps[(primordium,filename)].signal_maps[signal_name] for filename in primordium_filenames if int(filename[-2:]) == int(time)], axis=0)

    return average_signal_maps

