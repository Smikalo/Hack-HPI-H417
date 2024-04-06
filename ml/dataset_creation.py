#libraries
import pickle
import os
from pathlib import Path

import shapely

from source import neighbourhood as nbh
import pandas as pd
from shapely import wkt
from sklearn.model_selection import train_test_split
# local functions
from utils.utility_funcs import get_splitsuf

def main_dataset_creation():
	#parameters
	test_mode = False
	datasetrootname = 'v221217_NodeR0'
	split_strategy = 'yearly' #options: randomsamp, yearly, historical, NBH
	n = 2 #number of hops
	num_nbh = 60000 #total number of nbhs to generate (if nbh) or nbhs to generate for train (if not nbh)
	num_nbh_VT_factor = 0.25
	batchsize = 60000 #make sure modulus of num_nbh*num_nbh_VT_factor/batchsize is 0

	num_nbh_train = num_nbh*(1-2*num_nbh_VT_factor) if split_strategy=='NBH' else num_nbh
	num_nbh_dict = {"train" : int(num_nbh_train),
	                "val" : int(num_nbh*num_nbh_VT_factor),
	                "test" : int(num_nbh*num_nbh_VT_factor)}
	# set path and datasetname
	rootpath = os.path.dirname(os.path.realpath(__file__)) #Abs path of AccIndex.git
	data_path = os.path.join(rootpath,'data')
	# datasetname prefix and suffixes
	regioncode = 'DSR' if test_mode else 'Berlin'
	dataset_name = regioncode + '_'+ datasetrootname
	splitsuf = get_splitsuf(split_strategy)
	graphdatapath = os.path.join(data_path, dataset_name, "Graphs_split"+split_strategy+'_'+splitsuf)
	splitsizestr = ''.join([key+str(val) for key,val in num_nbh_dict.items()])
	NBHdataparentpath = os.path.join(graphdatapath, 'nbhs_graphs_nhop%d'%(n))
	NBHdatapath = os.path.join(NBHdataparentpath, splitsizestr)
	os.mkdir(NBHdataparentpath) if not os.path.exists(NBHdataparentpath) else print(f'Continuing may overwrite any potential content in existing neighborhood-parent dir:{NBHdataparentpath}')
	os.mkdir(NBHdatapath) if not os.path.exists(NBHdatapath) else print(f'Continuing may overwrite any potential content in existing neighborhood dir:{NBHdatapath}')

	# Density and urban utilisation data downloaded via QGIS from https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s06_06ewdichte2021
	population_density_data = 'Berlin_pop_density.csv' # File can be found on Dropbox
	density_df = pd.read_csv(os.path.join(data_path, population_density_data))

	#density_df['geometry'] = density_df['geometry'].apply(lambda x : print(x))
	if split_strategy == 'NBH':
		D ={'nosplit': num_nbh}
	else:
		D = num_nbh_dict
	for split, num_nbh in D.items():
		fulldata_path = os.path.join(NBHdatapath, split)
		scaledfname = fulldata_path + str(num_nbh) + '.pkl'
		fnameminmaxscaler = os.path.join(NBHdatapath, 'density_minmax_scaler.pkl')
		fnamestdscaler = os.path.join(NBHdatapath, 'coords_standard_scaler.pkl')
		
		#check if the scaled files are already there, if not, create them
		if (split == 'train') and (os.path.exists(scaledfname)) and (os.path.exists(fnameminmaxscaler)) and (os.path.exists(fnamestdscaler)):
			print(f"For {split}, found previously created scaled neighborhood dataset: {scaledfname}")
			print(f"Found previously created scalers:\n {fnameminmaxscaler}\n {fnamestdscaler}")
		elif (split in ['validate', 'test']) and os.path.exists(scaledfname):
			print(f"Found previously created scaled neighborhood dataset: {scaledfname}.")
		else: #if the scaledfname does not exist
			rawfname =  fulldata_path + str(num_nbh) + '_raw.pkl'
			list_nbh = get_list_nbh(graphdatapath, rawfname, density_df, n, num_nbh, batchsize, fulldata_path, split_strategy, split)
		
			if split_strategy != 'NBH':
				apply_scalings(split,list_nbh,fnameminmaxscaler,fnamestdscaler,scaledfname)
			else:
				save_raw_apply_scalings_nbh(NBHdatapath, num_nbh_VT_factor, num_nbh_dict, list_nbh,fnameminmaxscaler,fnamestdscaler,scaledfname)


def get_list_nbh(graphdatapath, rawfname, density_df, n, num_nbh, batchsize, fulldata_path, split_strategy, split):
	if os.path.exists(rawfname):
		print(f"Found a previously created raw neighborhood dataset: {rawfname}. Loading..")
		list_nbh = pickle.load(file = open(rawfname,'rb'))
	else: #if rawfname does not exist
		print(f"Creating {num_nbh} neighborhoods for {split}:")
		if split_strategy == 'NBH':
			G = pickle.load(open(os.path.join(graphdatapath, 'nosplit.pkl'), 'rb'))
		else:
			G = pickle.load(open(os.path.join(graphdatapath, split+'.pkl'), 'rb'))
		list_nbh = create_list_nbh(G, density_df, n, num_nbh, batchsize, fulldata_path, split_strategy, split)
		print(' - done. Saving...')
		pickle.dump(list_nbh, open(rawfname, 'wb'))
		print(f" saved raw neighborhood file: {rawfname}")
	return list_nbh

def create_list_nbh(G, density_df, n, num_nbh, batchsize, fulldata_path, split_strategy, split):
	if num_nbh > batchsize:
		numbatches = int(num_nbh/batchsize)
		#create the neighborhoods
		for batchno in range(numbatches):
			batchfname = fulldata_path + '_raw_batch-' + str(batchno) + '.pkl'
			#if the file is not already there, create it
			if os.path.exists(batchfname):
				print(f" found previously created batch: {batchno+1}/{numbatches}")
			else:
				print(f" creating batch: {batchno+1}/{numbatches}")
				if split_strategy == 'historical':
					list_nbh_batch = nbh.datapoints_neighbourhood_with_labels(G, density_df, n, batchsize, split)
				else:
					list_nbh_batch = nbh.datapoints_neighbourhood_with_labels(G, density_df, n, batchsize)
				pickle.dump(list_nbh_batch, open(batchfname, 'wb'))
		#stitch all batch files
		print(f" loading {numbatches} batch files to stitch: ", end='')
		list_nbh = []
		for batchno in range(numbatches):
			print(str(batchno+1), end=' ')
			batchfname = fulldata_path + '_raw_batch-' + str(batchno) + '.pkl'
			list_nbh_batch = pickle.load(file = open(batchfname,'rb'))
			list_nbh += list_nbh_batch
	else: #if single batch
		if split_strategy == 'historical':
			list_nbh = nbh.datapoints_neighbourhood_with_labels(G, density_df, n,  int(num_nbh), split)
		else:
			list_nbh = nbh.datapoints_neighbourhood_with_labels(G, density_df, n, int(num_nbh))
	
	return list_nbh 


def apply_scalings(split,list_nbh,fnameminmaxscaler,fnamestdscaler,scaledfname):
	# Apply scaling with fit scalers based on train data, so far only global attributes
	print('Applying scalings')
	if split == "train":
		density, coords = nbh.scaling_collect(list_nbh)
		print(f'Creating scalers based on {split} dataset')
		density_minmax_scaler, coords_standard_scaler = nbh.scaling_fit(density, coords)
		#save the scalers
		pickle.dump(density_minmax_scaler, open(fnameminmaxscaler, 'wb'))
		pickle.dump(coords_standard_scaler, open(fnamestdscaler, 'wb'))
		print(f' - saved the scalers:\n {fnameminmaxscaler}\n {fnamestdscaler}')
		print(f'Applying scalers to {split} dataset')
		list_nbh_scaled = nbh.scaling_transform(list_nbh, density, coords, density_minmax_scaler, coords_standard_scaler)
	elif split == "validate" or "test":
		#load the scalers:
		print(f'Loading the scalers:\n {fnameminmaxscaler}\n {fnamestdscaler}')
		density_minmax_scaler = pickle.load(file= open(fnameminmaxscaler, 'rb'))
		coords_standard_scaler = pickle.load(file= open(fnamestdscaler, 'rb'))
		print(f'Applying the scalers to {split} dataset')
		density, coords = nbh.scaling_collect(list_nbh)
		list_nbh_scaled = nbh.scaling_transform(list_nbh, density, coords, density_minmax_scaler, coords_standard_scaler)
	pickle.dump(list_nbh_scaled, open(scaledfname, 'wb'))
	print(f" - saved scaled neighborhood file: {scaledfname}")


def save_raw_apply_scalings_nbh(NBHdatapath, num_nbh_VT_factor, num_nbh_dict, list_nbh,fnameminmaxscaler,fnamestdscaler,scaledfname):
	#split list_nbh to train-val-test
	list_nbh_dict = get_nbh_splits(list_nbh, num_nbh_VT_factor)
	#sav raw TVT splits, and scale them
	for nbhsplit, list_nbh in list_nbh_dict.items():
		fulldata_path = os.path.join(NBHdatapath, nbhsplit)
		num_nbh = num_nbh_dict[nbhsplit]
		rawsplitfname = fulldata_path + str(num_nbh) + '_raw.pkl'
		pickle.dump(list_nbh, open(rawsplitfname, 'wb'))
		print(f" saved raw neighborhood file: {rawsplitfname}")
		scaledsplitfname = fulldata_path + str(num_nbh) + '.pkl'
		apply_scalings(nbhsplit,list_nbh,fnameminmaxscaler,fnamestdscaler,scaledsplitfname)


def get_nbh_splits(list_nbh, num_nbh_VT_factor):
	list_nbh_train, list_nbh_valtest = train_test_split(list_nbh, test_size=2*num_nbh_VT_factor, shuffle=True)
	list_nbh_val, list_nbh_test =  train_test_split(list_nbh_valtest, test_size=0.5, shuffle=True)
	list_nbh_dict = {'train':list_nbh_train, 'val':list_nbh_val, 'test':list_nbh_test}
	return list_nbh_dict
	
	
if __name__ == '__main__':
	main_dataset_creation()