#!/bin/bash

# Step 1: Create Conda environment
echo -e "\n\033[1;36m>>> Creating Conda environment: translation_env <<<\033[0m"
conda create -n translation_env python=3.9 -y

# Step 2: Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate translation_env

# Step 3: Install required packages
echo -e "\n\033[1;36m>>> Installing dependencies... <<<\033[0m"
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt

# Step 4: Preprocess the data
echo -e "\n\033[1;36m>>> Preprocessing data files... <<<\033[0m"
python -c "from data.preprocessing import preprocess; preprocess('en-id.txt/tico-19.en-id.id', 'en-id.txt/tico-19.en-id.en')"

# Step 5: Start training the model
echo -e "\n\033[1;36m>>> Training started! Please wait... <<<\033[0m"
python3 train.py

# Step 6: Run inference on the trained model
echo -e "\n\033[1;36m>>> Running inference... <<<\033[0m"
python3 inference.py

echo -e "\n\033[1;32m✔ All steps completed successfully! 🎉\033[0m"
