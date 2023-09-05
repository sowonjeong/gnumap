#!/bin/bash

# Run the Python script with the specified dataset
python experiment_main.py --name_dataset Circles
python experiment_main.py --name_dataset Blobs
python experiment_main.py --name_dataset Moons
python experiment_main.py --name_dataset Sphere
python experiment_main.py --name_dataset Swissroll

# save_img yes
# , 'UMAP', 'DenseMAP'
# In terminal
# chmod +x exp.sh
# bash exp.sh