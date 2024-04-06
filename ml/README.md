# AccIndex

## Quick Links
- [Installation](#Installation)
<!-- - [Train and Test](#Train-and-Test) -->
<!-- - [Related Work](#Related-Work) -->
- [Datasets](#Datasets)
<!-- - [Baselines](#Baselines) -->
<!-- - [Predictive Performances](#Predictive-performances) -->

## Installation
For GNN run the following commands to create a conda environment:
```bash
conda create -n gnn_cpu python=3.8
conda activate gnn_cpu
pip install torch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```
[OSMnx](https://github.com/gboeing/osmnx) is a Python package that lets you download geospatial data from OpenStreetMap and model, project, visualize, and analyze real-world street networks and any other geospatial geometries.
```bash
conda install -c conda-forge osmnx
```
## Datasets
Road traffic accidents data for Germany from the year 2016 to 2021 can be found [here!](https://unfallatlas.statistikportal.de/_opendata2022.html)

## Example Results

[Here](https://onurkerimoglu-accind-routepr-accind-route-prediction-app-objw0o.streamlit.app/) a bare-bones web app to estimate the cumulative number of accidents on a route between two addresses to be provided by the user.

[Here](https://htmlpreview.github.io/?https://github.com/OnurKerimoglu/SharedPlots/blob/main/pred_acc_on_path.html) is an interactive version of such an example route. 

[Here](https://htmlpreview.github.io/?https://github.com/OnurKerimoglu/SharedPlots/blob/main/predicted_edge_acc_sqrt_density_and_raw_dots.html) is an interactive map with predicted accidents for the entire Berlin, along with the accidents that did happen between 2018-2021. 