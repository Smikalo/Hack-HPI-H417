# libraries:
import os
import pickle
import torch
from utils.utility_funcs import get_user_confirmation,get_datetime_string,get_splitsuf
from visualizations import visualize_training_hist
# local functions:
from model_funcs_accind import build_model_accind
from data_funcs import get_dataset, checkpaths

def main_train_test_validate():
    #Global definitions and parameters
    use_case = 'AccInd'
    verbosetest = False
    numepochs = 20
    datasetrootname = "v221217_NodeR0"
    split_strategy = 'yearly' #options: randomsamp, yearly, NBH, historical
    n = 2 #number of hops used to generate the neighborhoods
    num_nbh = 60000 #number of neighbourhoods to generate
    num_nbh_VT_factor = 0.25 #0.2
    modelrootname = 'DO025_lr0005'

    # set paths and datasetname
    dataset_name = "Berlin_" + datasetrootname
    dataset_version = ' '.join(datasetrootname.split('_')[1:])
    rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
    data_path = os.path.join(rootpath,'data')
    splitsuf = get_splitsuf(split_strategy)
    graphdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)
    num_nbh_train = num_nbh*(1-2*num_nbh_VT_factor) if split_strategy=='NBH' else num_nbh
    num_nbh_dict = {"train" : int(num_nbh_train),
                    "val" : int(num_nbh*num_nbh_VT_factor),
                    "test" : int(num_nbh*num_nbh_VT_factor)}
    splitsizestr = ''.join([key+str(int(val)) for key,val in num_nbh_dict.items()])
    NBHdatapath=os.path.join(graphdatapath, 'nbhs_graphs_nhop%d'%(n),splitsizestr)
    model_folder_path = os.path.join(NBHdatapath, 'models')
    os.mkdir(model_folder_path) if not os.path.exists(model_folder_path) else print(f'Model folder exists:{model_folder_path}')
    date_string, time_string = get_datetime_string()
    model_name = modelrootname #+ '_' + date_string # + '_' + time_string
    model_rootpath = os.path.join(model_folder_path, model_name + '_' + str(numepochs)+'epochs')
    model_path = model_rootpath + ".pt"
    modelskill_path = model_rootpath + "_skill.pkl"

    #Check if doing the training is needed/desired
    do_training = True
    if (os.path.exists(model_path)) and (os.path.exists(modelskill_path)):
        msg = f'Model and model-skill files exist:\n{model_path}\n{modelskill_path}\nShall I continue and overwrite them?[y/(n)]'
        overwrite = get_user_confirmation(msg)
        if not overwrite:
            print(f'ok, will not overwrite the files. Unpickling the model-skill file.')
            ModelSkillDict = pickle.load(file=open(modelskill_path, 'rb'))
            do_training = False

    if do_training:
        print('Will do the training from scratch.')
        #Load datasets, do the training and model evaluation
        checkpaths(data_path, dataset_name, NBHdatapath, num_nbh_dict)

        dataset, DSpars, train_loader, val_loader, test_loader = get_dataset(
            datasetname = use_case,
            NBHdatapath = NBHdatapath,
            datasetfnames = num_nbh_dict,
            report = False)

        #visualize the dataset
        #nbh.plot_neighbourhood(to_networkx(dataset[3])) #uncomment to visualize the dataset

        # build the model:
        gn_model, optimizer, train_loss_function, test_loss_function = build_model_accind(
            DSpars, report=True)

        #train, validate, test
        ModelSkillDict = train_val_test_accind(
                            gn_model, numepochs,
                            train_loader, val_loader, test_loader,
                            train_loss_function, test_loss_function,
                            optimizer, model_path, modelskill_path, verbosetest)

    expname = f'Data: {dataset_version} Split: {splitsuf} model: {model_name}'
    visualize_training_hist(ModelSkillDict, modelskill_path, expname)

    print('All tasks complete')


def train_val_test_accind(gn_model, numepochs, train_loader, val_loader, test_loader, train_loss_function, test_loss_function, optimizer, model_path, modelskill_path, verbosetest=False):
    print('Doing the training:')
    train_loss_hist = [None] * numepochs
    val_loss_hist = [None] * numepochs
    for epoch in range(numepochs):
        print(f'Epoch: {epoch:03d}', end=':')
        train_accind(gn_model, train_loader, train_loss_function, optimizer)
        train_loss = test_accind(gn_model, train_loader, test_loss_function, verbosetest)
        val_loss = test_accind(gn_model, val_loader, test_loss_function, verbosetest)
        train_loss_hist[epoch] = float(train_loss)
        val_loss_hist[epoch] = float(val_loss)
        print(
            f'\n  Train Loss: {train_loss_hist[epoch]:.4f}, Val Loss: {val_loss_hist[epoch]:.4f}'
        )
    # do testing
    print('Doing the testing against the test data')
    test_loss = test_accind(gn_model, test_loader, test_loss_function, verbosetest)
    print(f'Test Loss: {test_loss:.4f}')

    ModelSkillDict = {'train': train_loss_hist,
                    'val': val_loss_hist,
                    'test': float(test_loss)}

    #save the model and modelskill
    torch.save(gn_model.state_dict(), model_path)
    pickle.dump(ModelSkillDict, file=open(modelskill_path,'wb'))
    print("Saved the model to drive:" + model_path)
    print("Saved the model-skill to drive:" + modelskill_path)

    return ModelSkillDict


def train_accind(gn_model, train_loader, train_loss_function, optimizer):
    gn_model.train()

    batch=0
    for data in train_loader:  # Iterate over the batches of graphs
        batch +=1
        #print(f'\n -B{batch:01d}_data: {data}')
        #print(f'\n -B{batch:01d}_edgei: {data.edge_index}')
        out = gn_model(data.x, data.edge_attr, data.u, data.edge_index,
                       data.n_pred_idx, data.e_pred_idx, data.batch)  # Forward pass(es)
        # Compute the loss #convert y data typeto float32
        loss = train_loss_function(out, data.y.to(torch.float32))
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the weights based on the computed gradients
        optimizer.zero_grad()  # Clear the computed gradients


def test_accind(gn_model, loader, test_loss_function, verbose=False):
    gn_model.eval()

    # correct = 0
    loss_cum = 0.0 #we want to calculate the cumulative loss
    for batchno,data in enumerate(loader):
        # Iterate over the batches
        out = gn_model(data.x, data.edge_attr, data.u, data.edge_index,
                       data.n_pred_idx, data.e_pred_idx, data.batch)
        # Prediction is the output itself
        # pred = out  # todo: maybe max(0,out) if we get negative values
        # Check against the ground truth (todo: maybe use MAE or something else?)
        # Compute the loss
        loss_batchmean = test_loss_function(out, data.y.to(torch.float32)) #one number for one batch
        #(((out[:,1] - data.y[:,1])**2).mean() + ((out[:,0] - data.y[:,0])**2).mean())/2
        loss_batchsum = loss_batchmean * len(out[:,0]) #multiply with the number of neighborhoods
        loss_cum += loss_batchsum #this is the total loss for all neighborhoods
        if verbose:
            print(f'  batch #: {batchno} (containing {len(out[:,0])} neighborhoods), mean loss:{loss_batchmean}')
    
    #we divide by the total number of nbhs in all batches
    meanloss = loss_cum/len(loader.dataset)
    #so that this is the average loss we make per neighborhood, i.e., average loss for a node and edge
    if verbose:
        print(f'  mean loss: %6.4f after %d neighbourhoods across %d batches'%(meanloss, len(loader.dataset), batchno+1))

    return meanloss


if __name__ == '__main__':
    main_train_test_validate()