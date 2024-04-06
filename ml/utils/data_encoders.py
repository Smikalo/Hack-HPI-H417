import pandas as pd


def encode_categorical_variables(G, cat_var, edge_or_node):
    '''
    Encodes categorical variables in the graph object G.
    The function creates a new entry in the edge or node object for each level of the categorical variable.
    The new entry is a binary variable that is 1 if the level is present in the edge or node object and 0 otherwise.
    The function also collects the levels of the categorical variable and prints the share of irregularities.
    Irregularities are cases where the categorical variable is not present in the edge or node object.
    
    :param G: graph object
    :param cat_var: categorical variable to be encoded
    :param edge_or_node: "edge" or "node"
    :param return: the list of levels of the categorical variable.
    '''
    print("encoding of categorical variable: ", cat_var, " in ", edge_or_node )
    cat_var_collector = []
    irregularity_counter = 0

    # Collect levels of the categorical variable to be encoded
    if edge_or_node == "edge":
        for s, r, e in G.edges(data=True):
            e, cat_var_collector, irregularity_counter = level_collector(e, cat_var, cat_var_collector, irregularity_counter)
        print("irregularity share: ", irregularity_counter, "/", len(G.edges(data=True)))
        cat_levels = list(set(cat_var_collector))
        #For each level of that variable create a new entry in the edge object
        for s, r, e in G.edges(data=True):
            level_encoder(e, cat_var, cat_levels)
    elif edge_or_node == "node":
        for i, n in G.nodes(data=True):
            n, cat_var_collector, irregularity_counter = level_collector(n, cat_var, cat_var_collector, irregularity_counter)
        print("irregularity share: ", irregularity_counter, "/", len(G.nodes(data=True)))
        cat_levels = list(set(cat_var_collector))
        #For each level of that variable create a new entry in the edge object
        for i, n in G.nodes(data=True):
            level_encoder(n, cat_var, cat_levels)
    else:
        raise(Exception("Argument 'edge_or_node' must be specified either 'edge' or 'node'."))

    df = pd.DataFrame (cat_var_collector, columns = [cat_var])
    print(df[cat_var].value_counts())
    print("\n")

    new_attributes = [cat_var+'_'+str(l) for l in cat_levels]
    return G, new_attributes



def level_collector(object, cat_var, cat_var_collector, irregularity_counter):
    '''
    Collects the levels of the categorical variable (cat_var) and returns an appended list
    it updates the irregularity counter as well
    
    :param object: edge or node object
    :param cat_var: categorical variable to be encoded
    :param cat_var_collector: list of levels of the categorical variable
    :param irregularity_counter: counter for irregularities
    :param return: the edge or node object with the new entries, the updated list of levels of 
    the categorical variable and the updated irregularity counter
    '''
    if cat_var in object.keys():
        if not isinstance(object[cat_var], list):
            cat_var_collector.append(object[cat_var])
        else:
            #TODO: find better handling for list entries
            print(object[cat_var])
            object[cat_var] = "irregular_data"
            cat_var_collector.append(object[cat_var])
            irregularity_counter += 1
    else:
        object[cat_var] = "irregular_data"
        cat_var_collector.append(object[cat_var])
        irregularity_counter += 1

    return object, cat_var_collector, irregularity_counter

def level_encoder(object, cat_var, cat_levels):
    '''
    :param object: edge or node object
    :param cat_var: categorical variable to be encoded
    :param cat_levels: list of levels of the categorical variable
    :param return: the edge or node object with the new entries
    '''
    # Assign 1 or 0 to dummy variables in the object
    if cat_var in object.keys():
        for l in cat_levels:
            if object[cat_var] == l:
                object[cat_var+'_'+str(l)] = 1
            else:
                object[cat_var+'_'+str(l)] = 0