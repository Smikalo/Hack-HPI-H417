## Installation
```
conda create -n acc_predict python=3.8
conda activate acc_predict
conda install pip
pip install -r requirements.txt
```  
Apart from libraries in the requirement files, you will need to install torch and several of its dependencies.
These libraries are notoriously incompatible, so no definite installation guide can be provided here, except for:
```
pip install torch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```

## Data Creation

Donload the raw accident data from https://unfallatlas.statistikportal.de/_opendata2022.html.
Unzip the files in /data/Berlin_accident_raw/. Since the raw data provider does not follow a 
consistent naming pattern in the yaerly accient data files, it might be necessary that you will
have to manually adjust the names or even the exact file structure of the resulting unzipped raw csv files. 
The naming convention is:
"data/Berlin_accident_raw/Unfallorte2016_EPSG25832_CSV/csv/Unfallorte2016_LinRef.csv" , with the respective year replaced.
(Or see create_accident_data() in graph_object_creation.py)

### graph_object_creation.py

Run graph_object_creation.py. This file created the accident data from the raw accident data (see above), if it does not exist yet.
Then it downloads the Open Street Map (OSM) data as a graph. The accident are assigned to the closest nodes and edges.
Additionally, attributes (e.g. max speed, bridge etc) are assigned to nodes and edges, that will later serve as predictors. Where necessary, these
attributes are encoded. The outcome are 3 Graph objects. Each object has the same structure, but each has different data assigned to.
They serve as train, validation and test data sets.

### dataset_creation.py

Having the Graph objects is not sufficient. Since the GNN learns on small neighbourhoods, these neighbourhoods need to be created first.
Each neighbourhood consists of a small number of nodes with the connecting edges in between. dataset_creation.py creates these neighbourhoods in great amounts.
Additionally, it adds global features to the graph. These features are population density and area use. Both is downloaded using QGIS the address:
https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s06_06ewdichte2021. Since this a nontrivial endeavour, the data is provided on our github repo. 
Please download the file "Berlin_pop_density_and_areause.csv" and save it in /data/

## Modelling

### main_train_val_test.py

Now, that the necessary graph objects are in place, we can start modeling.
In main_train_test_validate() adjust all parameters. These include the correct naming of the datasets,
as well model specific parameters. Depending on the data set and number of epochs training can take long (hours).
The generated model is saved in the corresponding folders that contain the data.

## Prediction

### main_predict.py
There are two modes to employ the models. in main_predict.py you can predict the expected accidents on a specific route.
Enter the geo coordinates of the departure and desitination, and specify the name and path of the model to use. The outcome
is a graphical representation of the shortest route and the predicted accidents at each edge.

### global_predict.py
In order to predict for an entire graph (i.e. a map), use global_predict.py. Again, specify the correct paths and names
of the model and the graph being used. The output is a csv file with predictions of every edge in the graph. To make
the the global predictions more palpable, run the following script.

### analyze_predictions.py
In total, 3 functions are provided in the analysis tool.
plotmap will produce an interactive plotly map that shows the predicted accidents at every single edge throught the map.
This comes with all the amenities that plotly provides: zooming, hovering, scrolling. A mapbox access token, however is required.
plotscatter statistically examines the model performance against the true values. Several charts are output.
create_pred_graph_toggle produces a Graph that contains the predictions on all adges. That forms the basis for the following script.

### dash_app.py
Based on the graph object generated in analyze_predictions.py (create_pred_graph_toggle), this script creates an 
interactive application, that lets you enter start and destination address in the browser. That will return plotly map
with the shortest route displayed plus the number of predicted accidents on this route.