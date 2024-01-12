INSTALATION
1-create new conda env (python 3.9.18)
2-pip install -r requirements.lock

for gpu:
3-conda install cudatoolkit=11.8 cudnn=8.9 -c=conda-forge
4-pip3 install torch torchvision torchaudio --index url https://download.pytorch.org/whl/cu118
5-change game settings: self.max_num_gpus = 1


RUN
1-python muzero.py
2-tensorboard --logdir ./results (run on a new terminal to see training results)


PLAY VS ADV AGENT
1-run server.py
2-run muzero.py (on a new terminal)
3-run other agent (new terminal)
step 3 can also be done with muzero.py (self vs self but in the server)