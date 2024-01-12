# Muzero for Attaxx and Go

Muzero agent able to learn / play autonomously


## Installation

1. Create new conda env (python 3.9.18)
2. Install the requirements
```bash
pip install -r requirements.lock
```

- For GPU:

3. Install Cuda and Cudnn (Nvidia)
```bash
conda install cudatoolkit=11.8 cudnn=8.9 -c=conda-forge
```
4. Install Torch compatible with Cuda
```bash
pip3 install torch torchvision torchaudio --index url https://download.pytorch.org/whl/cu118
```
5. Change game settings @ self.max_num_gpus = 1




## Usage

- RUN:

1. Run muzero.py (follow UI instructions)
2. Run Tensorboard to see training results (on a new terminal)
```bash
tensorboard --logdir ./results
```

- PLAY VS ADV AGENT:

1. Run server.py
2. Run muzero.py (on a new terminal)
3. Select "Play against other Agent"
4. Run other agent (on a new terminal)


## LIACD @ FCUP
Group 6 - JAN 2024
