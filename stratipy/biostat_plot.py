import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
from stratipy import biostat
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
plt.switch_backend('agg')


def get_biostat_files(result_directory):
    # get all biostat_files in result_directory
    biostat_files = [os.path.join(path, name)
                     for path, dirs, files in os.walk(result_directory)
                     for name in files
                     if name.startswith(("biostat"))]
    return biostat_files


def p_stringent_loose(val, p_stringent, p_loose):
    if val < p_stringent:
        return 2
    elif val < p_loose:
        return 1
    else:
        return 0


def concatenate_2_pval(row, col1, col2, p_stringent, p_loose):
    profile1 = p_stringent_loose(row[col1], p_stringent, p_loose)
    profile2 = p_stringent_loose(row[col2], p_stringent, p_loose)
    return [profile1, profile2]


def formatting_biostat_data(data_folder, result_folder,
                            biostat_factorization_directory, ssc_mutation_data,
                            gene_data, ppi_data, patient_data, p_stringent,
                            p_loose):
    print(" ==== Plot data formatting", flush=True)
    # to avoid SettingWithCopyWarning\
    pd.options.mode.chained_assignment = None
    # create list of all biostatistics files in direcytory
    if patient_data == 'SSC':
        # load both SSC1 & SSC2
        result1 = re.sub(r'SSC.', 'SSC1', biostat_factorization_directory)
        result2 = re.sub(r'SSC.', 'SSC2', biostat_factorization_directory)
        biostat_files = get_biostat_files(result1) + get_biostat_files(result2)

    else:
        biostat_files = get_biostat_files(result_folder)

    # read and concatenate them
    df = pd.concat((pd.read_pickle(file) for file in biostat_files))

    # keep only binary results
    df = df[[
        'data_k', 'data_ssc',  'sex_pval', 'sp_pval', 'iq_pval', 'srs_pval',
        'vineland_pval', 'distCEU_pval', 'mutation_pval']]
    # slice into SSC 1 & 2
    df1 = df[df['data_ssc'] == 'SSC1']
    df2 = df[df['data_ssc'] == 'SSC2']
    df = df1.merge(df2, how='inner', left_on='data_k', right_on='data_k', suffixes=[1, 2])

    df.rename(columns={'data_k': 'k'}, inplace=True)
    df = df.sort_values(by=['k'])
    df = df.reset_index(drop=True)

    # new columns with p-value -> ternary profile
    df['Sex'] = df.apply(lambda row: concatenate_2_pval(
        row, 'sex_pval1', 'sex_pval2', p_stringent, p_loose), axis=1)
    df['Affected / Unaffected'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'sp_pval1', 'sp_pval2', p_stringent, p_loose), axis=1)
    df['IQ'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'iq_pval1', 'iq_pval2', p_stringent, p_loose), axis=1)
    df['SRS'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'srs_pval1', 'srs_pval2', p_stringent, p_loose), axis=1)
    df['Vineland'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'vineland_pval1', 'vineland_pval2', p_stringent, p_loose),
        axis=1)
    df['Ancestral distance'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'distCEU_pval1', 'distCEU_pval2', p_stringent, p_loose),
        axis=1)
    df['# mutated genes'] = df.apply(
        lambda row: concatenate_2_pval(
            row, 'mutation_pval1', 'mutation_pval2', p_stringent, p_loose),
        axis=1)

    return df


def marker_style(p_profile):
    marker = 'o'
    size = 16
    edge = 'gray'
    # p-value < 0.05
    col005 = 'orangered'
    # p-value < 0.1
    col01 = 'orange'

    if p_profile == [0, 0]:
        fill = 'none'
        return dict(marker=marker, markersize=size, markeredgecolor=edge,
                    fillstyle=fill)

    else:
        fill = 'left' if p_profile[0] > p_profile[1] else 'right'

        if (p_profile == [0, 1]) or (p_profile == [1, 0]):
            color = col01
            return dict(marker=marker, markersize=size, markeredgecolor=edge,
                        fillstyle=fill, color=color)

        elif (p_profile == [0, 2]) or (p_profile == [2, 0]):
            color = col005
            return dict(marker=marker, markersize=size, markeredgecolor=edge,
                        fillstyle=fill, color=color)

        else:
            edge = 'black'
            width = 2

            if p_profile == [1, 1]:
                color = col01
                fill = 'full'
                return dict(marker=marker, markersize=size,
                            markeredgecolor=edge, fillstyle=fill, color=color,
                            markeredgewidth=width)

            elif (p_profile == [1, 2]) or (p_profile == [2, 1]):
                color = col005
                face = col01
                return dict(marker=marker, markersize=size,
                            markeredgecolor=edge, fillstyle=fill, color=color,
                            markeredgewidth=width, markerfacecoloralt=face)

            elif p_profile == [2, 2]:
                color = col005
                fill = 'full'
                return dict(marker=marker, markersize=size,
                            markeredgecolor=edge, fillstyle=fill, color=color,
                            markeredgewidth=width)

            else:
                # NOTE error message
                print('Wrong profile format')


def biostat_individuals_plot(df, data_folder, ssc_mutation_data, gene_data,
                             patient_data, ppi_data, mut_type, lambd):
    print(" ==== Plotting", flush=True)
    p_col = ['k', 'Sex', 'Affected / Unaffected', 'IQ', 'SRS',
             'Vineland', 'Ancestral distance', '# mutated genes']
    df_fill = df[p_col]

    m_style = dict(marker='o', markersize=16, markeredgecolor='gray')
    legend_elements = [
        Line2D([0], [0], fillstyle='left', linestyle='', color='gray',
               **m_style, label='SSC1'),
        Line2D([0], [0], fillstyle='right', linestyle='', color='gray',
               **m_style, label='SSC2'),
        Line2D([0], [0], fillstyle='none', linestyle='',
               markeredgecolor='black', markeredgewidth=2, marker='o',
               markersize=16, label='reciprocal SSC1 & SSC2'),
        Line2D([0], [0], fillstyle='full', linestyle='', color='orange',
               **m_style, label='p-value < 0.1'),
        Line2D([0], [0], fillstyle='full', linestyle='', color='orangered',
               **m_style, label='p-value < 0.05')]

    # k=20 -> figsize=(5, 9)
    fig, ax = plt.subplots(nrows=df_fill.shape[0], ncols=df_fill.shape[1],
                           sharex=True, sharey=True, figsize=(5, 20))
    if lambd > 0:
        nmf = 'GNMF'
    else:
        nmf = 'NMF'

    fig.suptitle(
        "Statistical significance between individual clusters\n(mutation:{} // gene:{} // PPI:{} // {} // {})".
        format(ssc_mutation_data, gene_data, ppi_data, mut_type, nmf), x=0.5,
        y=1.15, fontsize=14, linespacing=2)

    for col in range(len(p_col)):
        for row in range(df_fill.shape[0]):
            if col == 0:
                ax[row, col].text(0.02, 0.5, df_fill.iloc[row, col],
                                  horizontalalignment='right',
                                  verticalalignment='center', fontsize=12)
                ax[row, col].axis('off')
            else:
                ax[row, col].plot(0.5, **marker_style(df_fill.iloc[row, col]))
                ax[row, col].axis('off')

        ax[0, col].text(0, 0.54, p_col[col], horizontalalignment='left',
                        verticalalignment='bottom', rotation=45, fontsize=12)

    fig.subplots_adjust(hspace=0, wspace=0)
    # plt.legend(handles=legend_elements, loc='center', labelspacing=1,
    #            fontsize=14, bbox_to_anchor=(6, 12), frameon=False)

    if patient_data == 'SSC':
        fig_directory = (
            data_folder + 'figures/biostat_individuals/' + ssc_mutation_data +
            '_' + gene_data + '_' + ppi_data + '/')
    else:
        fig_directory = (data_folder + 'figures/biostat_individuals/' +
                         patient_data + '_' + ppi_data + '/')
    os.makedirs(fig_directory, exist_ok=True)
    fig_name = ('{}_lambd={}'.format(mut_type, lambd))
    plt.savefig('{}{}.png'.format(fig_directory, fig_name),
                bbox_inches='tight')
    plt.savefig('{}{}.svg'.format(fig_directory, fig_name),
                bbox_inches='tight')
    plt.close()


def load_plot_biostat_individuals(result_folder, data_folder,
                                  ssc_mutation_data, gene_data, patient_data,
                                  ppi_data, mut_type, lambd, influence_weight,
                                  simplification, alpha, tol, keep_singletons,
                                  ngh_max, min_mutation, max_mutation,
                                  n_components, n_permutations, tol_nmf,
                                  linkage_method):
    p_stringent = 0.05
    p_loose = 0.1

    biostat_factorization_directory, biostat_file = biostat.biostatistics_file(
        result_folder, mut_type, influence_weight, simplification, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf, linkage_method)

    df = formatting_biostat_data(
        data_folder, result_folder, biostat_factorization_directory,
        ssc_mutation_data, gene_data, ppi_data, patient_data, p_stringent,
        p_loose)

    biostat_individuals_plot(
        df, data_folder, ssc_mutation_data, gene_data, patient_data,
        ppi_data, mut_type, lambd)

# def binary_cmap(val_seqence, i):
#     cmap = matplotlib.cm.get_cmap('binary')
#     normalize = matplotlib.colors.Normalize(vmin=0, vmax=max(val_seqence))
#     colors = [cmap(normalize(value)) for value in val_seqence]
#     return colors[i]
#

# def marker_filling(row, col1, col2):
#     if row[col1]:
#         if row[col2]:
#             return 'full'
#         else:
#             return 'left'
#     else:
#         if row[col2]:
#             return 'right'
#         else:
#             return 'none'
#
#
# def graduated_marker_style(left, right):
#     if (left[0] == 1) & (right[0] == 1):
#         fillstyle = 'none'
#     else:
#         fillstyle = 'left'
#     return dict(fillstyle=fillstyle, color=left, markerfacecoloralt=right,
#                 markeredgecolor='black', marker='o', markersize=15)


# def formatting_biostat_data(data_folder, result_folder,
#                             biostat_factorization_directory, ssc_mutation_data,
#                             gene_data, ppi_data, p_val_threshold,
#                             patient_data):
#     # to avoid SettingWithCopyWarning\
#     pd.options.mode.chained_assignment = None
#     # create list of all biostatistics files in direcytory
#     if patient_data == 'SSC':
#         # load both SSC1 & SSC2
#         result1 = re.sub(r'SSC.', 'SSC1', biostat_factorization_directory)
#         result2 = re.sub(r'SSC.', 'SSC2', biostat_factorization_directory)
#         biostat_files = get_biostat_files(result1) + get_biostat_files(result2)
#     else:
#         biostat_files = get_biostat_files(biostat_factorization_directory)
#
#     # read and concatenate them
#     df = pd.concat((pd.read_pickle(file) for file in biostat_files))
#     df = df.reset_index(drop=True)
#
#     df_plot = df[[
#         'data_k', 'data_ssc', 'sp_pval', 'iq_pval', 'sex_pval', 'srs_pval',
#         'vineland_pval', 'distCEU_pval', 'mutation_pval']]
#
#     # new columns: True if significant p-value
#     df_plot.loc[:, 'sp'] = (df_plot['sp_pval'] < p_val_threshold)
#     df_plot.loc[:, 'iq'] = (df_plot['iq_pval'] < p_val_threshold)
#     df_plot.loc[:, 'sex'] = (df_plot['sex_pval'] < p_val_threshold)
#     df_plot.loc[:, 'srs'] = (df_plot['srs_pval'] < p_val_threshold)
#     df_plot.loc[:, 'vineland'] = (df_plot['vineland_pval'] < p_val_threshold)
#     df_plot.loc[:, 'distCEU'] = (df_plot['distCEU_pval'] < p_val_threshold)
#     df_plot.loc[:, 'mutation'] = (df_plot['mutation_pval'] < p_val_threshold)
#
#     # keep only binary results
#     df_bin = df_plot.drop(['sp_pval', 'iq_pval', 'sex_pval', 'srs_pval',
#                            'vineland_pval', 'distCEU_pval', 'mutation_pval'],
#                           axis=1)
#     # slice into SSC 1 & 2
#     df_bin1 = df_bin[df_bin['data_ssc'] == 'SSC1']
#     df_bin2 = df_bin[df_bin['data_ssc'] == 'SSC2']
#     # count True by row (k)
#     df_bin1['Total'] = df_bin1.iloc[:, 1:].sum(axis=1)
#     df_bin2['Total'] = df_bin2.iloc[:, 1:].sum(axis=1)
#
#     # then merge on k
#     df_bin = df_bin1.merge(df_bin2, how='inner', left_on='data_k',
#                            right_on='data_k', suffixes=[1, 2])
#     df_bin = df_bin.sort_values(by=['data_k'])
#     df_bin = df_bin.drop(['data_ssc1', 'data_ssc2'], axis=1)
#     df_bin = df_bin.reset_index(drop=True)
#
#     # count all True by row (k) for SSC1&2
#     df_bin['Total'] = df_bin[['Total1', 'Total2']].sum(axis=1)
#
#     # new columns with marker fjilling style
#     df_bin['Affected / Unaffected'] = df_bin.apply(
#         lambda row: marker_filling(row, 'sp1', 'sp2'), axis=1)
#     df_bin['IQ'] = df_bin.apply(
#         lambda row: marker_filling(row, 'iq1', 'iq2'), axis=1)
#     df_bin['Sex'] = df_bin.apply(
#         lambda row: marker_filling(row, 'sex1', 'sex2'), axis=1)
#     df_bin['SRS'] = df_bin.apply(
#         lambda row: marker_filling(row, 'srs1', 'srs2'), axis=1)
#     df_bin['Vineland'] = df_bin.apply(
#         lambda row: marker_filling(row, 'vineland1', 'vineland2'), axis=1)
#     df_bin['Ancestral distance'] = df_bin.apply(
#         lambda row: marker_filling(row, 'distCEU1', 'distCEU2'), axis=1)
#     df_bin['# mutated genes'] = df_bin.apply(
#         lambda row: marker_filling(row, 'mutation1', 'mutation2'), axis=1)
#
#     df_bin.rename(columns={'data_k': 'k'}, inplace=True)
#
#     return df_bin
#
#
# def biostat_individuals_plot(df_bin, data_folder, ssc_mutation_data, gene_data,
#                              patient_data, ppi_data, mut_type, lambd):
#     p_col = ['k', 'Total', 'Affected / Unaffected', 'IQ', 'Sex', 'SRS',
#              'Vineland', 'Ancestral distance', '# mutated genes']
#     df_fill = df_bin[p_col]
#
#     marker_style = dict(color='black', marker='o', markersize=15)
#     legend_elements = [
#         Line2D([0], [0], fillstyle='left', linestyle='', **marker_style,
#                label='SSC1'),
#         Line2D([0], [0], fillstyle='right', linestyle='', **marker_style,
#                label='SSC2')]
#
#     fig, ax = plt.subplots(nrows=df_fill.shape[0], ncols=df_fill.shape[1],
#                            sharex=True, sharey=True, figsize=(6, 9))
#     if lambd > 0:
#         nmf = 'GNMF'
#     else:
#         nmf = 'NMF'
#
#     fig.suptitle(
#         "Statistical significance between individual clusters\n(mutation:{} // gene:{} // PPI:{} // {} // {})".
#         format(ssc_mutation_data, gene_data, ppi_data, mut_type, nmf), y=1.15,
#         fontsize=15, linespacing=2)
#
#     for col in range(len(p_col)):
#         for row in range(df_fill.shape[0]):
#             if col == 0:
#                 ax[row, col].text(0.02, 0.5, df_fill.iloc[row, col],
#                                   horizontalalignment='right',
#                                   verticalalignment='center', fontsize=12)
#                 ax[row, col].axis('off')
#             elif col == 1:
#                 color_left = binary_cmap(df_bin['Total1'], row)
#                 color_right = binary_cmap(df_bin['Total2'], row)
#                 ax[row, col].plot(0.5, **graduated_marker_style(color_left,
#                                                                 color_right))
#                 ax[row, col].spines['left'].set_visible(False)
#                 ax[row, col].spines['top'].set_visible(False)
#                 ax[row, col].spines['bottom'].set_visible(False)
#                 ax[row, col].set_xticklabels([])
#                 ax[row, col].tick_params(axis='both', which='both', length=0)
#             else:
#                 ax[row, col].plot(0.5, fillstyle=df_fill.iloc[row, col],
#                                   **marker_style)
#                 ax[row, col].axis('off')
#
#         ax[0, col].text(0, 0.54, p_col[col], horizontalalignment='left',
#                         verticalalignment='bottom', rotation=45, fontsize=12)
#
#     fig.subplots_adjust(hspace=0, wspace=0)
#     plt.legend(handles=legend_elements, loc='center', labelspacing=1,
#                fontsize=14, bbox_to_anchor=(3, 17.9), frameon=False)
#
#     if patient_data == 'SSC':
#         fig_directory = (
#             data_folder + 'figures/biostat_individuals/' + ssc_mutation_data +
#             '_' + gene_data + '_' + ppi_data + '/')
#     else:
#         fig_directory = (data_folder + 'figures/biostat_individuals/' +
#                          patient_data + '_' + ppi_data + '/')
#     os.makedirs(fig_directory, exist_ok=True)
#     fig_name = ('{}_lambd={}'.format(mut_type, lambd))
#     plt.savefig('{}{}.png'.format(fig_directory, fig_name),
#                 bbox_inches='tight')
#     plt.savefig('{}{}.svg'.format(fig_directory, fig_name),
#                 bbox_inches='tight')
#     plt.close()
