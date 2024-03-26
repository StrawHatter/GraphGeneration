import random

import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy.random
from pathlib import Path
import os

# Selects between individual graph or dataset
mode = input("Enter 1 to generate an individual graph. Enter 2 to generate a dataset of graphs. ")
# Input sanitization
while not (mode == "1" or mode == "2"):
    mode = input("Enter 1 to generate an individual graph. Enter 2 to generate a dataset of graphs. ")

# Individual graph
if mode == "1":
    # Number of nodes
    nodes = input("Enter the desired number of nodes: ")
    while not nodes.isnumeric():
        nodes = input("Enter the desired number of nodes: ")

    # Give nodes specific names
    node_names = []
    node_name = input("Enter a name for a desired node. Enter a blank name to finish: ")
    while not node_name == "" and len(node_names) < int(nodes)-1:
        node_names.append(node_name)
        node_name = input("Enter a name for a desired node. Enter a blank name to finish: ")

    # Number of edges
    edges = input("Enter the desired number of edges: ")
    while not edges.isnumeric():
        edges = input("Enter the desired number of edges: ")

    # Specific edges
    spec_edges = []
    spec_edge = input("Enter a desired edge. Enter a blank line to finish: ")
    # Ensure edge is valid
    while not spec_edge == "":
        try:
            spec_edge = spec_edge.split(",")
        except (ValueError, SyntaxError):
            spec_edge = input("Edge must be a valid tuple: ")
            continue
        if not len(spec_edge) == 2:
            spec_edge = input("Edge must be a valid tuple: ")
        elif not spec_edge[0] in node_names:
            spec_edge = input(str(spec_edge[0]) + " is not a valid node: ")
        elif not spec_edge[1] in node_names:
            spec_edge = input(str(spec_edge[1]) + " is not a valid node: ")
        else:
            spec_edges.append(spec_edge)
            spec_edge = input("Enter a desired edge. Enter a blank line to finish: ")

    # Random graph
    G = nx.gnm_random_graph(int(nodes), int(edges)-len(spec_edges), directed=True)
    # Graph of specified nodes and edges
    H = nx.DiGraph()
    # Add specified edges
    for edge in spec_edges:
        H.add_edge(edge[0], edge[1])

    # Rename nodes to specified names
    mapping = dict(zip(sorted(G, reverse=True), node_names))
    G = nx.relabel_nodes(G, mapping)

    # Combine random graph and specified graph
    G = nx.compose(G, H)
    # Add new random edges until length is correct
    for i in range(int(edges) - len(list(G.edges))):
        new_edge = random.choice(list(nx.non_edges(G)))
        G.add_edge(new_edge[0], new_edge[1])

    # Plot graph
    pos = nx.circular_layout(G)
    plt.subplot()
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    # Draw two-way connections as curved edges
    for edge in G.edges(data=True):
        rad = 0
        if G.has_edge(edge[1], edge[0]):
            rad = 0.1
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], connectionstyle=f'arc3, rad = ' + str(rad))

    # Set connectivity type
    graph_type = 0
    if nx.is_weakly_connected(G):
        graph_type = 1
    if nx.is_strongly_connected(G):
        graph_type = 2
    # Complete directed graph
    if G.number_of_edges() == int(nodes) * (int(nodes) - 1):
        graph_type = 3

    # JSON dictionary of graph data
    json_dict = {'nodes': list(G.nodes),
                 'edges': list(G.edges),
                 'type': graph_type,
                 'degree': list(G.degree()),
                 'planarity': 1 if nx.check_planarity(G)[0] else 0}

    # Save file to active directory
    with open('graph.json', 'w') as outfile:
        json.dump(json_dict, outfile)

    # Show plot
    plt.show()

# Dataset
if mode == "2":

    # Minimum number of nodes
    minnodes = input("Enter the minimum number of nodes: ")
    while not minnodes.isnumeric():
        minnodes = input("Enter the minimum number of nodes: ")

    # Maximum number of nodes
    nodes = input("Enter the maximum number of nodes: ")
    while not nodes.isnumeric():
        nodes = input("Enter the maximum number of nodes: ")

    # Number of graphs
    total = input("Enter desired number of graphs to generate. ")
    while not total.isnumeric():
        total = input("Enter desired number of graphs to generate. ")

    # Create graph folder
    cwd = os.getcwd()
    Path("graphs").mkdir(parents=True, exist_ok=True)

    # Generate random graph
    for i in range(int(total)):
        current_nodes = numpy.random.randint(minnodes, int(nodes)+1)
        current_edges = numpy.random.randint(1, current_nodes * (current_nodes-1)+1)
        G = nx.gnm_random_graph(current_nodes, current_edges, directed=True)

        graph_type = 0
        if nx.is_weakly_connected(G):
            graph_type = 1
        if nx.is_strongly_connected(G):
            graph_type = 2
        # Complete directed graph
        if G.number_of_edges() == current_nodes * (current_nodes-1):
            graph_type = 3

        json_dict = {'nodes': list(G.nodes),
                     'edges': list(G.edges),
                     'type': graph_type,
                     'degree': list(G.degree()),
                     'planarity': 1 if nx.check_planarity(G)[0] else 0
                     }

        filename = os.path.join(cwd, 'graphs/graph' + str(i) + '.json')
        with open(filename, 'w') as outfile:
            json.dump(json_dict, outfile)

        # Save config file
        node_dict = {'max_nodes': int(nodes)}
        filename = os.path.join(cwd, 'graphs/config.json')
        with open(filename, 'w') as outfile:
            json.dump(node_dict, outfile)

