import numpy as np

from tissue_nukem_3d.utils.matplotlib_tools import draw_box, hex_to_rgb

from sam_spaghetti.utils.signal_luts import signal_colormaps, signal_ranges, signal_lut_ranges, primordia_colors, clv3_color

signal_lut_ranges['radial_distance'] = (0,100)
signal_lut_ranges['hour_time'] = (-5,15)
signal_lut_ranges['primordium'] = (-4,6)
signal_lut_ranges['aligned_z'] = (-50,0)

signal_legends = {}
signal_legends['radial_distance'] = "Radial distance ($\mu m$)"
signal_legends['distance_std'] = "Radial extent ($\mu m$)"
signal_legends['relative_distance_std'] = "Radial extent ($\mu m$)"
signal_legends['relative_distance'] = "Distance from primordium ($\mu m$)"
signal_legends['angle_std'] = "Angular extent ($^\circ$)"
signal_legends['relative_angle_std'] = "Angular extent ($^\circ$)"
signal_legends['relative_angle_opening'] = "Angular opening ($^\circ$)"
signal_legends['elliptic_area'] = "Estimated 2D Projected Area ($\mu m^2$)"
signal_legends['2D_area'] = "2D Projected Area ($10^{3} \mu m^2$)"
signal_legends['Auxin'] = "Auxin (1 - qDII)"
signal_legends['qDII'] = "qDII"
signal_legends['Normalized_Auxin'] = "Normalized Auxin"
signal_legends['Normalized_qDII'] = "Normalized qDII"
signal_legends['gaussian_curvature'] = "Gaussian curvature ($10^{-3} \mu m^{-2}$)"
signal_legends['circularity'] = "Circularity"
signal_legends['aligned_theta'] = "Aligned angle ($^\circ$)"
signal_legends['DR5'] = "Auxin output ($10^{4}$ DR5)"
signal_legends['CLV3'] = "CZ domain ($10^{4}$ CLV3)"
signal_legends['hour_time'] = "Time ($h$)"
signal_legends['next_relative_surfacic_growth'] = "Surfacic growth ($\%$)"

signal_legends['primordium_radial_speed'] = "Radial speed ($\mu m . h^{-1}$)"
signal_legends['radial_speed'] = "Nuclei radial speed ($\mu m . h^{-1}$)"
signal_legends['next_primordium_radial_speed'] = signal_legends['previous_primordium_radial_speed'] = signal_legends['primordium_radial_speed']
signal_legends['next_radial_speed'] = signal_legends['previous_radial_speed'] = signal_legends['radial_speed']

signal_legends['next_radial_speed_ratio'] = "Auxin maximum / Nuclei radial speed ratio"
signal_legends['next_radial_relative_speed'] = "Radial relative speed ($\mu m . h^{-1}$)"
signal_legends['radial_speed_ratio'] = "Auxin maximum / Nuclei radial speed ratio "
signal_legends['radial_relative_speed'] = "Radial relative speed ($\mu m . h^{-1}$)"


golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)


def primordium_data_plot(primordia_data, figure, signal_name, primordium_range=[-3 ,-2 ,-1 ,0 ,1 ,2 ,3 ,4 ,5], time_range=[0 ,5 ,10], data_to_plot="detected_auxin_extrema", figure_size=(10 ,10), boxes_alpha=1, individual_curve_alpha=0, primordium_subplots=True, profile_n_boxes=11, sector_opening_angle=17.5):
    """
    """

    figure.clf()
    figure.patch.set_facecolor('w')

    for i_p, primordium in enumerate(primordium_range):

        primordium_angle = (primordium *np.degrees(golden_angle ) +180 ) %360 - 180

        if data_to_plot in ['radial_profiles']:
            figure.add_subplot(len(time_range) ,len(primordium_range) ,i_p +1)
            reference_signal = 'radial_distance'
        elif data_to_plot in ['ring_profiles']:
            figure.add_subplot(len(time_range) ,1 ,1)
            reference_signal = 'aligned_theta'
        else:
            if primordium_subplots:
                figure.add_subplot(1 ,len(primordium_range) ,i_p +1)
                reference_signal = 'primordium'
            else:
                reference_signal = 'hour_time'


        if (primordium_subplots) or (i_p == 0):
            if 'theta' in signal_name:
                for p in primordium_range:
                    primordium_angle = ( p *np.degrees(golden_angle ) +180 ) %360 - 180
                    figure.gca().plot(signal_lut_ranges[reference_signal] ,[primordium_angle ,primordium_angle] ,color=primordia_colors[p] ,alpha=0.5)

            if ('curvature' in signal_name) or ('divergence' in signal_name):
                figure.gca().plot(signal_lut_ranges[reference_signal] ,[0 ,0] ,color='k' ,alpha=0.2)

            if 'radial_distance' in signal_name:
                clv3_radius_percentiles = {}
                clv3_radius_values = [primordia_data[primordia_data['filename' ]==f]['clv3_radius'].mean() for f in np.unique(primordia_data['filename'])]
                for percentile in [25 ,50 ,75]:
                    clv3_radius_percentiles[percentile] = np.nanpercentile(clv3_radius_values ,percentile)

                # figure.gca().plot(signal_lut_ranges[reference_signal],[clv3_radius_percentiles[50],clv3_radius_percentiles[50]],color=clv3_color,alpha=1,linewidth=2,zorder=0)
                # figure.gca().fill_between(signal_lut_ranges[reference_signal],[clv3_radius_percentiles[25],clv3_radius_percentiles[25]],[clv3_radius_percentiles[75],clv3_radius_percentiles[75]],color=clv3_color,alpha=0.2,linewidth=1,zorder=0)
                figure.gca().plot([-10 ,20] ,[clv3_radius_percentiles[50] ,clv3_radius_percentiles[50]] ,color=clv3_color ,alpha=1 ,linewidth=2 ,zorder=0)
                figure.gca().fill_between([-10 ,20] ,[clv3_radius_percentiles[25] ,clv3_radius_percentiles[25]] ,[clv3_radius_percentiles[75] ,clv3_radius_percentiles[75]] ,color=clv3_color
                                          ,alpha=0.2 ,linewidth=1 ,zorder=0)

                for cell in range(-20 ,20):
                    if cell!=0:
                        figure.gca().plot([-10 ,20] ,[clv3_radius_percentiles[50 ] +cell *5.7 ,clv3_radius_percentiles[50 ] +cell *5.7] ,color='lightgrey' ,alpha=1 ,linewidth=2 ,zorder=0)

            if 'azimuthal_distance' in signal_name or 'radial_regression_distance' in signal_name:
                figure.gca().plot([-10 ,20] ,[-5.7 ,-5.7] ,color='lightgrey' ,alpha=1 ,linewidth=2 ,zorder=0)
                figure.gca().plot([-10 ,20] ,[0 ,0] ,color='lightgrey' ,alpha=1 ,linewidth=2 ,zorder=0)
                figure.gca().plot([-10 ,20] ,[5.7 ,5.7] ,color='lightgrey' ,alpha=1 ,linewidth=2 ,zorder=0)


        if data_to_plot in ['detected_auxin_extrema']:
            data_types = ['auxin_minimum' ,'auxin_maximum']
        elif data_to_plot in ['detected_auxin_maxima']:
            data_types = ['auxin_maximum']
        elif data_to_plot in ['detected_auxin_minima']:
            data_types = ['auxin_minimum']
        elif data_to_plot in ['primordium_sector_statisitics']:
            data_types = ['ring_sector_stats']
        elif data_to_plot in ['primordium_sectors']:
            data_types = ['ring_sector']
        elif data_to_plot in ['motion_sectors']:
            data_types = ['motion_sector']
        elif data_to_plot in ['inhibition_fields']:
            data_types = ['inhibition']
        elif data_to_plot in ['radial_profiles']:
            data_types = ['radial_profile']
        elif data_to_plot in ['ring_profiles']:
            data_types = ['ring_profile']

        for data_type in data_types:

            if data_type == 'auxin_maximum':
                primordium_data = primordia_data[(primordia_data['primordium' ]==primordium ) &(primordia_data['extremum_type' ]=='minimum' ) &(primordia_data['hour_time' ]<=np.max(time_range))]
            elif data_type == 'auxin_minimum':
                primordium_data = primordia_data[(primordia_data['primordium' ]==primordium ) &(primordia_data['extremum_type' ]!='minimum' ) &(primordia_data['hour_time' ]<=np.max(time_range))]
            # elif data_type == 'ring_sector_stats':
            #     primordium_data = sector_data[(sector_data['primordium' ]==primordium)]
            # elif data_type == 'ring_sector':
            #     primordium_data = all_data
            #         [(np.abs(all_data['radial_distance' ] -ring_radius )<=ring_width ) &(np.abs(((all_data['aligned_theta'].values -primordium_angle +180 ) %360 ) -180 )<=sector_opening_angle)]
            # elif data_type == 'motion_sector':
            #     primordium_data = all_data[(np.abs(((all_data['aligned_theta'].values -primordium_angle +180 ) %360 ) -180 )<=sector_opening_angle)]
            # elif data_type == 'inhibition':
            #     primordium_data = inhibition_data[(inhibition_data['primordium' ]==primordium)]
            # elif data_type == 'radial_profile':
            #     primordium_data = all_data[(np.abs(((all_data['aligned_theta'].values -primordium_angle +180 ) %360 ) -180 )<=sector_opening_angle)]
            # elif data_type == 'ring_profile':
            #     primordium_data = all_data[(np.abs(all_data['radial_distance' ] -ring_radius )<=ring_width)]

            sequence_individual_curves = {}
            sequence_individual_times = {}
            for sequence_name in np.unique([f[:-4] for f in primordium_data['filename']]):
                sequence_individual_curves[sequence_name] = []
                sequence_individual_times[sequence_name] = []

            for i_time, time in enumerate(time_range):
                if data_type in ['ring_profile']:
                    figure.add_subplot(len(time_range) ,1 ,i_time +1)

                    profile_boxes = np.linspace(signal_lut_ranges[reference_signal][0] ,signal_lut_ranges[reference_signal][1] ,profile_n_boxes)

                    if i_p == 0:
                        # time_data = all_data[(all_data['hour_time']==time)]
                        primordium_time_data = primordium_data[(primordium_data['hour_time' ]==time)]

                        circle_opening_angle = (profile_boxes[1 ] -profile_boxes[0] ) /2.

                        for t in profile_boxes:
                            primordium_time_theta_data = primordium_time_data[np.abs(((primordium_time_data['aligned_theta'].values - t +180 ) %360 ) -180 )<=circle_opening_angle]

                            color = np.array([0.95 ,0.95 ,0.95])
                            box_color = np.array([0.75 ,0.75 ,0.75])

                            for p in primordium_range[::-1]:
                                primordium_angle = ( p *np.degrees(golden_angle ) +180 ) %360 - 180
                                # primordium_weight = np.exp(-np.power(((primordium_angle-t+180)%360)-180,2)/np.power(sector_opening_angle,2))
                                primordium_weight = np.abs(((primordium_angle - t +180 ) %360 ) -180 )<=sector_opening_angle
                                color = primordium_weight *hex_to_rgb(primordia_colors[p]) + ( 1 -primordium_weight ) *color
                                box_color = primordium_weight *np.zeros(3) + ( 1 -primordium_weight ) *box_color

                            signal_values = primordium_time_theta_data[signal_name].values
                            draw_box(figure, signal_values, box_x=t, box_width=circle_opening_angle -1, color=color, box_color=box_color, outlier_size=5, outlier_alpha=0.05)

                        figure.gca().set_xlim(*signal_lut_ranges[reference_signal])
                        figure.gca().set_xticks(np.linspace(signal_lut_ranges[reference_signal][0] ,signal_lut_ranges[reference_signal][1] ,5))

                        if i_time == len(time_range ) -1:
                            figure.gca().set_xlabel(signal_legends[reference_signal] ,size=32)
                            figure.gca().set_xticklabels([int(t) for t in figure.gca().get_xticks()] ,size=24)
                        else:
                            figure.gca().set_xlabel("" ,size=32)
                            figure.gca().set_xticklabels([] ,size=24)

                # elif data_type in ['radial_profile']:
                #     figure.add_subplot(len(time_range) ,len(primordium_range) ,len(primordium_range ) *i_time +i_p +1)
                #
                #     profile_boxes = np.linspace(signal_lut_ranges[reference_signal][0] ,signal_lut_ranges[reference_signal][1] ,profile_n_boxes)
                #
                #     time_primordium_data = primordium_data[(primordium_data['hour_time' ]==time)]
                #
                #     clv3_radius_percentiles = {}
                #     clv3_radius_values = [primordia_data[primordia_data['filename' ]==f]['clv3_radius'].mean() for f in np.unique(primordia_data['filename'])]
                #     for percentile in [25 ,50 ,75]:
                #         clv3_radius_percentiles[percentile] = np.nanpercentile(clv3_radius_values ,percentile)
                #     figure.gca().plot([clv3_radius_percentiles[50] ,clv3_radius_percentiles[50]] ,signal_lut_ranges[signal_name] ,color=clv3_color ,alpha=1 ,linewidth=2 ,zorder=0)
                #     # figure.gca().fill_betweenx(signal_lut_ranges[signal_name],[clv3_radius_percentiles[25],clv3_radius_percentiles[25]],[clv3_radius_percentiles[75],clv3_radius_percentiles[75]],color=clv3_color,alpha=0.2,linewidth=1,zorder=0)
                #
                #     time_primordium_distance_percentiles = {}
                #     time_primordium_distances = [primordia_distances[primordium][filename] for filename in np.unique(time_primordium_data['filename'].values) if filename in primordia_distances[primordium].keys()]
                #     for percentile in [10 ,25 ,50 ,75 ,90]:
                #         time_primordium_distance_percentiles[percentile] = np.nanpercentile(time_primordium_distances ,percentile)
                #     figure.gca().plot([time_primordium_distance_percentiles[50] ,time_primordium_distance_percentiles[50]] ,signal_lut_ranges[signal_name] ,color='k' ,alpha=0.5 ,linewidth=2 ,zorder=0)
                #     figure.gca().fill_betweenx(signal_lut_ranges[signal_name] ,[time_primordium_distance_percentiles[25] ,time_primordium_distance_percentiles[25]]
                #                                ,[time_primordium_distance_percentiles[75] ,time_primordium_distance_percentiles[75]] ,color=primordia_colors[primordium] ,edgecolor='k' ,alpha=0.5
                #                                ,linewidth=1 ,zorder=0)
                #
                #     time_primordium_inhibition_distance_percentiles = {}
                #     time_primordium_inhibition_distances = [primordia_inhibition_distances[primordium][filename] for filename in np.unique(time_primordium_data['filename'].values) if filename in primordia_inhibition_distances[primordium].keys()]
                #     for percentile in [10 ,25 ,50 ,75 ,90]:
                #         time_primordium_inhibition_distance_percentiles[percentile] = np.nanpercentile(time_primordium_inhibition_distances ,percentile)
                #     figure.gca().plot([time_primordium_inhibition_distance_percentiles[50] ,time_primordium_inhibition_distance_percentiles[50]] ,signal_lut_ranges[signal_name]
                #                       ,color=primordia_colors[primordium] ,alpha=0.5 ,linewidth=2 ,zorder=0)
                #     figure.gca().fill_betweenx(signal_lut_ranges[signal_name] ,[time_primordium_inhibition_distance_percentiles[25] ,time_primordium_inhibition_distance_percentiles[25]]
                #                                ,[time_primordium_inhibition_distance_percentiles[75] ,time_primordium_inhibition_distance_percentiles[75]] ,color='none'
                #                                ,edgecolor=primordia_colors[primordium] ,alpha=0.5 ,linewidth=1 ,zorder=0)
                #
                #     radial_width = (profile_boxes[1 ] -profile_boxes[0] ) /2.
                #
                #     for r in profile_boxes:
                #         time_distance_primordium_data = time_primordium_data[np.abs(time_primordium_data['radial_distance'].values -r )<=radial_width]
                #         signal_values = time_distance_primordium_data[signal_name].values
                #
                #         if ( r - 2 *radial_width )<=time_primordium_distance_percentiles[75]:
                #             if len(signal_values ) >3:
                #                 color = np.array([0.95 ,0.95 ,0.95])
                #                 box_color = np.array([0.75 ,0.75 ,0.75])
                #
                #                 if (( r -radial_width )<=time_primordium_distance_percentiles[75]) and (( r +radial_width )> =time_primordium_distance_percentiles[25]):
                #                     color = hex_to_rgb(primordia_colors[primordium])
                #                     box_color = 'k'
                #                 elif (( r -2 . *radial_width )<=time_primordium_inhibition_distance_percentiles[75]) and \
                #                         ((r + 2. * radial_width) >= time_primordium_inhibition_distance_percentiles[25]):
                #                     color = 'w'
                #                     box_color = hex_to_rgb(primordia_colors[primordium])
                #                 elif ((r - radial_width) <= time_primordium_distance_percentiles[75]) and ((r + radial_width) >= time_primordium_inhibition_distance_percentiles[25]):
                #                     color = 0.9 * color + 0.1 * hex_to_rgb(primordia_colors[primordium])
                #                     box_color = 0.9 * box_color + 0.1 * hex_to_rgb(primordia_colors[primordium])
                #                 draw_box(figure, signal_values, box_x=r, box_width=radial_width - 1, color=color, box_color=box_color, outlier_size=5, outlier_alpha=0.05)
                #
                #     figure.gca().set_xlim(0, profile_boxes.max())
                #     figure.gca().set_xticks(np.linspace(0, profile_boxes.max(), profile_boxes.max() / 20 + 1))
                #     if i_time == len(time_range) - 1:
                #         figure.gca().set_xticklabels([int(t) for t in figure.gca().get_xticks()], size=24)
                #     else:
                #         figure.gca().set_xticklabels([], size=24)
                else:
                    if data_type in ['auxin_maximum', 'ring_sector_stats', 'ring_sector', 'motion_sector']:
                        # color = cm.ScalarMappable(cmap=signal_colormaps[signal_name],norm=Normalize(vmin=0,vmax=1)).to_rgba(1)
                        color = hex_to_rgb(primordia_colors[primordium])
                        # box_color = np.array(color)/2.
                        box_color = np.zeros(3)
                    elif data_type in ['auxin_minimum', 'inhibition']:
                        color = np.ones(3)
                        box_color = hex_to_rgb(primordia_colors[primordium])

                    time_primordium_data = primordium_data[(primordium_data['hour_time'] == time)]
                    # time_primordium_data = primordium_data[(primordium_data['hour_time']>=time)]
                    # if data_type in ['motion_sector']:
                    #     time_primordium_data = time_primordium_data[(time_primordium_data['radial_distance'] >= primordium_sectors[(primordium, time)][0]) & (
                    #                 time_primordium_data['radial_distance'] <= primordium_sectors[(primordium, time)][-1])]

                    if individual_curve_alpha>0:
                        for filename in np.unique(time_primordium_data['filename']):
                            sequence_individual_curves[filename[:-4]] += [np.nanpercentile(time_primordium_data[time_primordium_data['filename'] == filename][signal_name].values, 50)]
                            sequence_individual_times[filename[:-4]] += [time]

                        if i_time == len(time_range) - 1:
                            for sequence_name in sequence_individual_curves.keys():
                                color_amplitude = 0.5
                                sequence_color = np.minimum(1, np.maximum(0, color + color_amplitude * (0.5 - np.random.rand(3))))
                                if primordium_subplots:
                                    curve_x = primordium + np.array([np.where(np.array(time_range) == t)[0][0] for t in sequence_individual_times[sequence_name]]) / float(len(time_range))
                                else:
                                    curve_x = np.array(sequence_individual_times[sequence_name])

                                figure.gca().plot(curve_x, sequence_individual_curves[sequence_name], color=sequence_color, linewidth=2, alpha=2. * individual_curve_alpha / 3.)
                                figure.gca().scatter(curve_x, sequence_individual_curves[sequence_name], s=50, color=[sequence_color for t in time_range], edgecolor=box_color, marker='s', linewidth=1,
                                                     alpha=individual_curve_alpha, zorder=10)

                    signal_values = time_primordium_data[signal_name].values

                    if (len(signal_values) > 0):
                        if primordium_subplots:
                            box_x = primordium + i_time / float(len(time_range))
                            box_width = 0.13333
                        else:
                            box_x = time
                            box_width = 1.6667

                        if data_type in ['auxin_minimum', 'ring_sector_stats', 'auxin_maximum', 'inhibition']:
                            outlier_alpha = 0.33 * (individual_curve_alpha == 0)
                            outlier_size = 20
                        else:
                            outlier_alpha = 0.05 * (individual_curve_alpha == 0)
                            outlier_size = 5

                        draw_box(figure, signal_values, box_x=box_x, box_width=box_width, color=color, box_color=box_color, outlier_size=outlier_size, alpha=boxes_alpha,
                                 outlier_alpha=outlier_alpha * boxes_alpha)

                        if primordium_subplots:
                            figure.gca().set_xlim(primordium - 1.5 * box_width, primordium + (len(time_range) - 1) / float(len(time_range)) + 1.5 * box_width)
                            figure.gca().set_xticks(primordium + np.arange(len(time_range)) / float(len(time_range)))
                        else:
                            figure.gca().set_xlim(time_range[0] - 1.5 * box_width, time_range[-1] + 1.5 * box_width)
                            figure.gca().set_xticks(time_range)

                if (i_time == 0) and primordium_subplots and (not data_to_plot in ['ring_profiles']):
                    figure.gca().set_title("P" + str(int(primordium)), size=28)

                figure.gca().set_ylim(*signal_lut_ranges[signal_name])
                if 'aligned_theta' in signal_name:
                    figure.gca().set_yticks([np.round((p * np.degrees(golden_angle) + 180) % 360 - 180, 1) for p in primordium_range])
                if i_p == 0:
                    if ('area' in signal_name):
                        figure.gca().set_yticklabels([np.round(t / 1000., 1) for t in figure.gca().get_yticks()], size=24)
                    elif ('DR5' in signal_name) or ('CLV3' in signal_name) or (signal_name == 'PIN1'):
                        figure.gca().set_yticklabels([np.round(t / 10000., 1) for t in figure.gca().get_yticks()], size=24)
                    elif 'curvature' in signal_name:
                        figure.gca().set_yticklabels([np.round(t * 1000., 1) for t in figure.gca().get_yticks()], size=24)
                    elif ('distance' in signal_name) or ('angle' in signal_name):
                        figure.gca().set_yticklabels([int(t) for t in figure.gca().get_yticks()], size=24)
                    elif ('theta' in signal_name) or ('Auxin' in signal_name) or ('Normalized' in signal_name):
                        figure.gca().set_yticklabels([np.round(t, 1) for t in figure.gca().get_yticks()], size=24)
                    else:
                        figure.gca().set_yticklabels(figure.gca().get_yticks(), size=24)
                    if data_to_plot in ['radial_profiles', 'ring_profiles']:
                        figure.gca().set_yticklabels([l if i_l > 0 else "" for i_l, l in enumerate(figure.gca().get_yticklabels())])
                    if (data_to_plot not in ['radial_profiles', 'ring_profiles']) or (i_time == len(time_range) / 2):
                        figure.gca().set_ylabel(signal_legends[signal_name] if signal_name in signal_legends.keys() else signal_name, size=32)
                        # figure.gca().get_yaxis().set_label_coords(x=figure.gca().get_yaxis().label.get_position()[0],y=0.5+0.5*(len(time_range)%2==0))
                elif (data_to_plot in ['radial_profiles', 'ring_profiles']) and (i_p == len(primordium_range) - 1) and (len(time_range) > 1):
                    figure.gca().yaxis.tick_right()
                    if data_to_plot not in ['ring_profiles']:
                        figure.gca().set_yticklabels([], size=24)
                    figure.gca().twinx().set_ylabel("t=" + str(int(time)) + "$h$", size=28, rotation=0)
                    if data_to_plot not in ['ring_profiles']:
                        figure.gca().get_yaxis().set_label_coords(1.25, 0.5)
                    else:
                        figure.gca().get_yaxis().set_label_coords(1.05, 0.5)
                    figure.gca().set_yticklabels([], size=24)
                elif primordium_subplots:
                    figure.gca().set_yticklabels([], size=24)

        if data_to_plot in ['radial_profiles']:
            figure.gca().set_xlabel("", size=32)
            figure.gca().set_xticklabels([str(int(t)) if t < profile_boxes.max() else "" for t in figure.gca().get_xticks()], size=24)
        elif data_to_plot in ['ring_profiles']:
            figure.gca().set_xlabel("", size=32)
            figure.gca().set_xticklabels([str(int(t)) for t in figure.gca().get_xticks()], size=24)
        else:
            if primordium_subplots:
                figure.gca().set_xlabel("", size=32)
                figure.gca().set_xticklabels([str(t) if t in [np.min(time_range), np.max(time_range)] else "" for t in time_range], size=24)
            else:
                figure.gca().set_xlabel(signal_legends['hour_time'], size=32)
                figure.gca().set_xticklabels([int(t) for t in figure.gca().get_xticks()], size=24)

    figure.set_size_inches(*figure_size)
    if data_to_plot in ['radial_profiles']:
        figure.suptitle(signal_legends['radial_distance'], size=32, y=0.05)
        figure.tight_layout(rect=[0, 0.05, 1, 1])
        figure.subplots_adjust(hspace=0)
    elif data_to_plot in ['ring_profiles']:
        figure.suptitle(signal_legends['aligned_theta'], size=32, y=0.05)
        figure.tight_layout(rect=[0, 0.05, 1, 1])
        figure.subplots_adjust(hspace=0)
    else:
        if primordium_subplots:
            figure.suptitle(signal_legends['hour_time'], size=32, y=0.05)
            figure.tight_layout(rect=[0, 0.05, 1, 1])
        else:
            figure.tight_layout()
    figure.subplots_adjust(wspace=0)