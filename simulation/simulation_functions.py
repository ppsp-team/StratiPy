import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import datetime
import glob
import pickle
from itertools import repeat
sys.path.append(os.path.dirname(os.path.abspath('.')))


def generate_network(pathwaysNum, genesNum, connNeighboors, connProbability,
                     marker_shapes):
    pathways = []
    for n in range(0, pathwaysNum):
        pathway = nx.connected_watts_strogatz_graph(genesNum, connNeighboors,
                                                    connProbability)
        [pathway.add_node(x, shape=marker_shapes[n]) for x in range(genesNum)]
        # Pathways are initialized as independant Watts-Strogatz networks
        mapping = dict(
            zip(pathway.nodes(), [x+genesNum*n for x in pathway.nodes()]))
        pathway = nx.relabel_nodes(pathway, mapping)
        pathways.append(pathway)
    PPI = nx.union_all(pathways)
    return PPI


def addBetweenPathwaysConnection(PPI, pathwaysNum, genesNum):
    # ... which are connected together afterward.
    # Select randomly two different pathways
    n1 = np.random.randint(pathwaysNum)
    n2 = n1
    while n2 == n1:
        n2 = np.random.randint(pathwaysNum)
    # Connect them
    PPI.add_edge(np.random.randint(genesNum)+n1*genesNum,
                 np.random.randint(genesNum)+n2*genesNum)
    return PPI


def generateMutationProfile(genesNum, pathwaysNum, mutationProb):
    mutationProfile = np.zeros(pathwaysNum*genesNum)
    mutatedPathway = np.random.randint(pathwaysNum)
    pathwayMutations = (np.random.rand(genesNum) <= mutationProb)*1
    mutationProfile[(mutatedPathway*genesNum):(
        mutatedPathway*genesNum+genesNum)] = pathwayMutations
    return mutationProfile, mutatedPathway


def assignMutation(PPI, mutationProfile):
    for n in range(0, len(PPI)):
        PPI.node[n]['val'] = mutationProfile[n]
    return PPI


def generate_all_mutation_profile(patientsNum, PPI, genesNum, pathwaysNum,
                                  mutationProb):
    patients = np.zeros((patientsNum, len(PPI)))
    phenotypes = []
    for patient in range(0, patientsNum):
        mutationProfile, mutatedPathway = generateMutationProfile(genesNum,
                                                                  pathwaysNum,
                                                                  mutationProb)
        patients[patient, :] = mutationProfile
        # patients.append(assignMutation(PPI,mutationProfile).copy())
        phenotypes.append(mutatedPathway)
    return patients, phenotypes


def save_dataset(PPI, position, patients, phenotypes, pathwaysNum, genesNum,
                 connProbability, connNeighboors, connBetweenPathways,
                 patientsNum, mutationProb, output_folder, new_data=False):
    data_folder = output_folder + 'data/'
    os.makedirs(data_folder, exist_ok=True)
    if new_data:
        file = open(data_folder+'dataset_{}.txt'.format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")), 'wb')
        data = {'PPI': PPI,
                'position': position,
                'patients': patients,
                'phenotypes': phenotypes,
                'pathwaysNum': pathwaysNum,
                'genesNum': genesNum,
                'connProbability': connProbability,
                'connNeighboors': connNeighboors,
                'connBetweenPathways': connBetweenPathways,
                'patientsNum': patientsNum,
                'mutationProb': mutationProb}
        # pathwaysNum,genesNum,connProbability,connNeighboors,connBetweenPathways,patientsNum,mutationProb
        pickle.dump(data, file)
        file.close()

    # newest_file = max(glob.iglob(data_folder + '*.txt'), key=os.path.getctime)
    # with open(newest_file, 'rb') as handle:
    #     b = pickle.load(handle)
    #     print('load data = ', b)


def nodes_position(PPI):
    colors=[]
    for color in ['r', 'b', 'g', 'k', 'm', 'y']:
        colors.extend(repeat(color, genesNum))

    pos = nx.spring_layout(PPI, k=0.07, iterations=70)
    nx.draw_networkx(PPI, pos=pos, node_size=100, label=False,
                     node_color=colors)
    # plt.figure(1, figsize=(5,5))

    file = open('input/ppi_node_position.txt', 'wb')
    pickle.dump(pos, file)
    file.close()


def plot_network_patient(mut_type, alpha, tol, PPI, position, patients,
                         patientsNum, phenotypes, marker_shapes, result_folder):
    plot_directory = result_folder + 'final_influence/plots/'
    os.makedirs(plot_directory, exist_ok=True)
    plot_file = (
        plot_directory +
        'network_plot_{}_alpha={}_tol={}.pdf'.format(
            mut_type, alpha, tol))

    fig = plt.figure(1, figsize=(10, 22))
    plt.suptitle("{} mutation profile over PPI network for each patient"
                 .format(mut_type), y=0.92, fontsize=14)

    # draw networks with several pathways
    for pn in range(0, patientsNum):
        fig.add_subplot(patientsNum/2, 2, pn+1)
        frame = fig.gca()
        # draw nodes (pathway by pathway)
        for aShape in marker_shapes:
            nodeList = set([sNode[0] for sNode in filter(
                lambda x: x[1]["shape"] == aShape, PPI.nodes(data=True))])
            pathway_nodes = nx.draw_networkx_nodes(
                PPI, pos=position, node_shape=aShape, nodelist=nodeList,
                node_size=100, node_color=patients[pn, list(nodeList)],
                cmap=plt.cm.viridis, alpha=0.85, linewidths=0.5,
                with_labels=False)
            pathway_nodes.set_edgecolor('black')

        # draw edges
        nx.draw_networkx_edges(PPI, position, edge_color='dimgray', width=0.5)

        frame.set_title('Patient number = {}\nMutated pathway = {}'
                        .format(pn, phenotypes[pn]), fontsize=10)
        frame.set_xticks([])
        frame.set_yticks([])

    # patyway markers legend
    marker_legend = []
    for i in range(len(marker_shapes)):
        leg = mlines.Line2D([], [], linewidth=0, marker=marker_shapes[i],
                            markersize=10, label=i)
        marker_legend.append(leg)
    ax_legends = fig.add_axes([0.97, 0.7, 0.02, 0.1], frameon=False)
    ax_legends.set_title('Pathways', fontsize=12)
    ax_legends.legend(handles=marker_legend, loc='upper center', fontsize=12,
                      frameon=False)
    ax_legends.set_xticks([])
    ax_legends.set_yticks([])

    # colorbar
    ax_colorbar = fig.add_axes([0.97, 0.3, 0.02, 0.3])
    ax_colorbar.set_title('Mutation score\n', fontsize=12)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colorbar = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=plt.cm.viridis,
                                         norm=norm, alpha=1)

    # save
    plt.savefig(plot_file, bbox_inches='tight')

    plt.clf()
