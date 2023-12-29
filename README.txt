INSTALATION
1-create new conda env (python 3.9.18)
2-pip install -r requirements.lock

for gpu:
3-conda install cudatoolkit=11.8 cudnn=8.9 -c=conda-forge
4-pip3 install torch torchvision torchaudio --index url https://download.pytorch.org/whl/cu118


RUN
1-python muzero.py
2-tensorboard --logdir ./results (on a new terminal)