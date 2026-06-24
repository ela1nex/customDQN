# customDQN
a personal project to create a customizable DQN (deep Q-learning) algorithm with dynamic environment prediction for training and testing. visualizes results through video output.

built with pytorch and tested on the cartpole environment from gymnasium.

## features(?)
- pretrained models for easy loading and testing
- SO MANY ANNOTATIONS bc im silly and forget easily so this is a way for me to learn and remember!
- configs file for easy editing (most of) the important parameters
- dynamic implementation for model-based planning

# preview
![preview](preview.gif)
trained w default configs

## how 2 use
### install dependencies
``` bash
pip install -r requirements.txt
```

### train (optional)
see notes before training!!<br>
change hyperparams in configs.py (also optional)
```bash
python train.py
```

### test
```bash
python test.py
```

outputs a `.mp4` file of the agent in the cartpole environment

## notes
- time taken to train increases significantly with more episodes
- training was tested with 4090 and may not run well on cpu
- untested for environments other than cartpole so it may perchance maybe break bc i may have hardcoded some stuff that should be in configs but hopefully will be fixed soon 
- does not work with nondiscrete environments cos dqn :(
- default tests with dynamic model but using select_action instead of select_planned_action tests with DQN

## future
- n-step truncated return 

---

README last updated 06/23/2026
