#!/usr/bin/env bash

set -e
cd ..

echo "Installing common requirements..."
pip3 install -q -r requirements.txt

if [[ $1 == "vanilla" ]]; then
    echo "[Vanilla model choose]"

    echo "Installing requirements..."
    pip install -q -r ./pistoBot/01_RNN/requirements.txt

    echo "Training model..."
    python ./pistoBot/01_RNN/rnn_vanilla.py

elif [[ $1 == "gpt2-simple" ]]; then
    echo "[gpt2-simple model choose]"

    echo "Installing requirements..."
    pip install -q -r ./pistoBot/02_gpt2_simple/requirements.txt

    echo "Training model..."
    python ./pistoBot/02_gpt2_simple/gpt2_simple.py -v

elif [[ $1 == "gpt2-scratch" ]]; then
    echo "[gpt2-scratch model chosen]"

    echo "Installing requirements..."
    pip install -q -r ./pistoBot/03_gpt2_scratch/requirements.txt

    echo "Training model..."
    python ./pistoBot/03_gpt2_scratch/gpt2_scratch.py -v
elif [[ $1 == "gpt_neo_scratch" ]]; then
    echo "[gpt_neo-scratch model chosen]"

    echo "Installing requirements..."
    pip3 install -q -r ./pistoBot/04_gpt_neo_scratch/requirements.txt

    echo "Training model..."
    python3 ./pistoBot/04_gpt_neo_scratch/gpt_neo_scratch.py -v
else
    echo "$1 model not recognized"
fi




