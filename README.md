# Multistep-DDPG
The implementation of Multistep-DDPG and Mixed-Multistep-DDPG proposed in [The Effect of Multi-step Methods on Overestimation in Deep Reinforcement Learning](https://arxiv.org/pdf/2006.12692.pdf).

Installation
============
Install Multistep-DDPG and dependencies.

```
    git clone https://github.com/LinghengMeng/Multistep-DDPG.git
    cd Multistep-DDPG
    pip install -e .
```
To use tasks from [PyBulletGym](https://github.com/benelot/pybullet-gym), please follow the instructions
in https://github.com/benelot/pybullet-gym to install Pybullet-Gym.

Run
============
python .\mddpg\mddpg_main.py

Note
============
The code for other baselines:
* DDPG, SAC, TD3 can be found in [Spinningup](https://spinningup.openai.com/en/latest/)
* MVE, STEVE can be found in [Stochastic-Ensemble-Value-Expansion](https://github.com/LinghengMeng/Stochastic-Ensemble-Value-Expansion.git).
