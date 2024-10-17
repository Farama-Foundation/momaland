"""Contains utilities to converts route choice game networks from https://github.com/maslab-ufrgs/transportation_networks to NetworkX networks.

Each network in MASLAB consists of two files:
    * network.net: contains the definition of the network (nodes, edges, latency functions, and origin-destination pairs)
    * network.routes: contains the possible routes for the origin-destination pairs of the network

This file contains methods to parse both files and return one .json file which contains:
    * "graph": the network in NetworkX JSON format (can be read with 'node_link_graph' method of NetworkX)
    * "od": a list of possible origin-destination (OD) pairs in the network
    * "routes": a dictionary which contains routes for OD-pairs
"""

import json
import os

import networkx as nx
from py_expression_eval import Parser


def read_network_file(problem_name):
    """Reads and parses the .net file of the network which contains the definition of the nodes, edges and OD-pairs of the network.

    Args:
        problem_name: the name of the problem for which the network file (.net) will be parsed

    Returns:
        graph: the NetworkX graph in JSON format (can be read with 'node_link_graph' method of NetworkX)
        od_pairs: a list of possible origin-destination pairs in the network
    """
    # networkX graph to store the network
    graph = nx.DiGraph()
    # store the parsed latency/time-cost functions
    latency_functions = {}
    # store the encountered origin-destination (OD) pairs
    od_pairs = []

    # keep track of current line for error reporting
    lineid = 0
    for line in open(problem_name + ".net"):
        lineid += 1
        # ignore \n
        line = line.rstrip()
        # ignore comments
        hash_pos = line.find("#")
        if hash_pos > -1:
            line = line[:hash_pos]
        # split the line
        taglist = line.split()
        # ignore empty lines
        if len(taglist) == 0:
            continue

        # -- Latency Function -- #
        # format of function definitions: "type name formula variables"
        if taglist[0] == "function":
            # process the params
            params = taglist[2][1:-1].split(",")
            # only latency functions with one parameter are supported
            if len(params) > 1:
                raise Exception(
                    "Cost functions with more than one parameter are not yet acceptable! (parameters defined: %s)"
                    % str(params)[1:-1]
                )
            # process the function
            expr = taglist[3]
            # process the constants
            function = Parser().parse(expr)
            constants = function.variables()
            # the parameter must be ignored (parameter is not a constant)
            if params[0] in constants:
                constants.remove(params[0])
            # store the function
            latency_functions[taglist[1]] = [params[0], constants, expr]

        # -- Node -- #
        # format of node definitions: type name
        elif taglist[0] == "node":
            # extract the name of the node
            node_name = taglist[1]
            # add the node to the network
            graph.add_node(node_name)

        # -- Edge -- #
        # format of edge definitions: type name origin destination function constants
        elif taglist[0] == "dedge" or taglist[0] == "edge":  # dedge is a directed edge
            # extract values from taglist
            edge_type = taglist[0]
            edge_name = taglist[1]
            edge_origin = taglist[2]
            edge_destination = taglist[3]
            # retrieve the latency function from previously parsed latency functions
            func_tuple = latency_functions[taglist[4]]
            # associate constants (from function def) and values specified in the line (in order of occurrence)
            param_values = dict(zip(func_tuple[1], map(float, taglist[5:])))
            # add a directed edge from the origin to the destination
            graph.add_edge(
                edge_origin,
                edge_destination,
                name=edge_name,
                latency_function={"expr": func_tuple[2], "param": func_tuple[0], "constants": param_values},
            )
            # if edge is not a directed edge also add the inverse directed edge
            if edge_type == "edge":
                graph.add_edge(
                    edge_destination,
                    edge_origin,
                    name=edge_name,
                    latency_function={"expr": func_tuple[2], "param": func_tuple[0], "constants": param_values},
                )

        # -- Origin/Destination pairs -- #
        # format of OD pairs definition: type name origin destination flow
        elif taglist[0] == "od":
            # ODs with no flow are ignored
            if taglist[4] != 0:
                od_pairs.append(taglist[1])
        else:
            raise Exception('Network file does not comply with the specification! (line %d: "%s")' % (lineid, line))

    # convert Networkx Graph to JSON format
    graph_json = nx.node_link_data(graph)
    return graph_json, od_pairs


def read_routes_file(problem_name):
    """Parses .routes files of networks which contains possible routes for each origin-destination (OD) pair.

    Args:
        problem_name: the name of the problem for which the .routes file will be parsed

    Returns:
        routes: a dictionary which contains a list of possible roads for each OD pair
    """
    # keep track of OD pairs for which we have encountered routes
    routes = dict()
    with open(problem_name + ".routes") as routes_file:
        for line in routes_file:
            # ignore \n
            line = line.rstrip()
            # ignore comments
            hash_pos = line.find("#")
            if hash_pos > -1:
                line = line[:hash_pos]
            # split the line
            taglist = line.split()
            # skip empty lines
            if len(taglist) == 0:
                continue

            # ignore origin-destination pairs with no links (no flow)
            if len(taglist) != 1:
                # get the OD pair
                od = taglist[0]
                # if OD pair is not yet in routes dictionary add it to the dictionary
                if od not in routes:
                    routes[od] = [taglist[1]]
                # if OD pair is already in the routes dictionary append the new route to the list of routes
                else:
                    routes[od].append(taglist[1])
    return routes


def save_json(problem_name, graph, od, routes):
    """Creates a .JSON file which contains a network of the MORouteChoice game environment.

    Args:
        problem_name: the name of the problem that is parsed
        graph: the JSON representation of the NetworkX network
        od: the list of origin-destination (OD) pairs in the network
        routes: a dictionary with the OD pairs as keys and all possible routes per OD pair as values

    Returns:
        /
    """
    # save JSON NetworkX graph
    with open(problem_name + ".json", "w") as json_out_file:
        out_json = json.dumps({"graph": graph, "od": od, "routes": routes}, indent=4)
        json_out_file.write(out_json)


if __name__ == "__main__":
    # look for all .net files in the "./networks/" directory and generate NetworkX graphs saved as JSON files
    all_net_files = [filename for filename in os.listdir("./") if filename.endswith(".net")]
    for problem_filename in all_net_files:
        found_problem_name = problem_filename.split(".net")[0]
        print(f"Creating NetworkX JSON file for {found_problem_name}")
        graph, od = read_network_file(found_problem_name)
        routes = read_routes_file(found_problem_name)
        save_json(found_problem_name, graph, od, routes)
