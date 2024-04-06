# packages
import os
import pickle
from torch import manual_seed
from torch import tensor as Ttensor
from torch import float32 as Tfloat32
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


def get_dataset(datasetname, NBHdatapath, datasetfnames=None, report=False):
    if datasetname == 'MUTAG':
        dataset = TUDataset(root='data/TUDataset',
                            name='MUTAG',
                            transform=add_global_attr)
        check_onewaynodes(dataset)
        train_loader, val_loader, test_loader = get_data_loaders_mutag(
            dataset, report=True)
        if report:
            summarize_dataset_mutag(dataset)
        DSpars = None #just a dummy variable, it won't be used for the mutag case
    elif datasetname == 'AccInd':
        datadict = get_data_accind(NBHdatapath, datasetfnames)
        #check_onewaynodes(datadict['train'])
        print('creating data loaders')
        train_loader, val_loader, test_loader = get_data_loaders_accind(
            datadict, report)
        if report:
            summarize_dataset_accind(train_loader.dataset)
        #this is just a dummy variable which won't be used at the end
        dataset = (train_loader.dataset)
        DSpars = parameters(train_loader.dataset[0])

    return dataset, DSpars, train_loader, val_loader, test_loader


class parameters():
    #takes an example data (graph object) and returns parameters needed for building the model
    def __init__(self, exdata):
        self.num_features = exdata.num_features
        self.num_edge_features = exdata.num_edge_features
        #self.u_size_0 = exdata.u.size(0) #this is not used
        self.u_size_1 = exdata.u.size(1)
        ##we may have to change something if we have >1 features 
        self.num_y = exdata.y.size(1)


def get_data_accind(NBHdatapath,num_nbh_dict):
    print('loading datasets:')
    datadicts={}
    for datasetname, num_nbh in num_nbh_dict.items():
        print(datasetname, end=' ')
        datasetfname = datasetname + str(int(num_nbh)) + '.pkl'
        with open(os.path.join(NBHdatapath,datasetfname), 'rb') as pickleFile:
            datalist = pickle.load(pickleFile)
            # This is a list of n 'torch_geometric.data.Data' objects, i.e., [D1, D2, ..., Dn])
        datadicts[datasetname] = datalist
        # this is a dictionary of 3 lists with keys train, val, test,
    print('- done.')
    return datadicts


def get_data_loaders_accind(datadict, report):
    # manual_seed(12345)
    # dataset = dataset.shuffle()

    train_dataset = datadict['train']
    val_dataset = datadict['val']
    test_dataset = datadict['test']

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of val graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #this is why batches/epoches look different from each other?
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if report:
        data_loader_dict = {
            'training': train_loader,
            'validation': val_loader,
            'testing': test_loader
        }
        for splitname, data_loader in data_loader_dict.items():
            print(f'Graphs in the {splitname} set:')
            for step, data in enumerate(data_loader):
                print(f'Step {step + 1}:')
                print('=======')
                print(
                    # Check: num_graph attribute might be created by the torch DataLoader
                    f'Number of graphs in the current batch: {data.num_graphs}'
                )
                print(data.batch)
                print()

    return train_loader, val_loader, test_loader


def add_global_attr(data):
    data.u = Ttensor([[0]]).to(Tfloat32)
    return data


def get_data_loaders_mutag(dataset, report):
    manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:128]
    val_dataset = dataset[128:158]
    test_dataset = dataset[158:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of val graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if report:
        data_loader_dict = {
            'training': train_loader,
            'validation': val_loader,
            'testing': test_loader
        }
        for splitname, data_loader in data_loader_dict.items():
            print(f'Graphs in the {splitname} set:')
            for step, data in enumerate(data_loader):
                print(f'Step {step + 1}:')
                print('=======')
                print(
                    f'Number of graphs in the current batch: {data.num_graphs}'
                )
                print(data.batch)
                print()

    return train_loader, val_loader, test_loader


def summarize_dataset_mutag(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Have a look at the first graph.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def summarize_dataset_accind(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of neighbourhoods: {len(dataset)}')
    # print(type(dataset))
    print(f'Number of node features: {dataset[0].x.shape[1]}')
    print(f'Number of edge features: {dataset[0].edge_attr.shape[1]}')
    print(f'Number of labels: {dataset[0].y.shape[0]}')
    # print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Have a look at the first graph.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of edges: {data.edge_index.shape[1]}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    a = 1

def check_onewaynodes(dataset):
    print('Checking no-way or one-way nodes in the neighborhoods...')
    Gi = 0
    onewayGcount = 0
    for data in dataset:
        Gi += 1
        G = to_networkx(data)
        G1 = G.copy()
        both = 0
        only_in = 0
        only_out = 0
        neither = 0
        for i, n in G1.nodes(data=True):
            if (len(G.out_edges(i)) > 0) & (len(G.in_edges(i)) > 0):
                both += 1
            elif (len(G.out_edges(i)) > 0) & (len(G.in_edges(i)) == 0):
                G.remove_node(i)
                only_out += 1
            elif (len(G.out_edges(i)) == 0) & (len(G.in_edges(i)) > 0):
                G.remove_node(i)
                only_in += 1
            elif (len(G.out_edges(i)) == 0) & (len(G.in_edges(i)) == 0):
                neither += 1
            else:
                print("da fuq?")
        if  only_in or only_out or neither:
            onewayGcount += 1
            print(f' Graph: {Gi}')
            print("   both: ", both)
            print("   only_in: ", only_in)
            print("   only_out: ", only_out)
            print("   neither: ", neither)
    print(f'# of graphs with no-way or one-way nodes: {onewayGcount}')
    #if onewayGcount>0:
        #raise(Exception('Graphs with no-way or one-way nodes found. Eliminate them and come back.'))
    
def checkpaths(data_path,dataset_name,NBHdatapath, num_nbh_dict):
    dataset_path = os.path.join(data_path, dataset_name)
    if os.path.exists(dataset_path):
        if os.path.exists(NBHdatapath):
            found = True
            for datasetname, num_nbh in num_nbh_dict.items():
                print(datasetname, end=' ')
                datasetfname = datasetname + str(int(num_nbh)) + '.pkl'
                if not os.path.exists(os.path.join(NBHdatapath, datasetfname)):
                    found = False
            if found:
                print('All necessary data sets found.')
            else:
                raise(Exception(f'Neighborhood folder ({NBHdatapath}) found, but at least one of train/validate/test.pkl are not there'))
        else:
            raise(Exception(f'Dataset folder ({dataset_path}) found, but the expected subfolder "{NBHdatapath.split(os.sep)[-1]}" is not there'))
    else:
        raise(Exception(f'Dataset folder ({dataset_path}) was not found'))