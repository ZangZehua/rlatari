# rl-atari

| Algo   | Done         |
|--------|--------------|
| DQN    | 1            |
| PPO    | $\checkmark$ |
| CURL   | $\checkmark$ |
| RAD    | $\checkmark$ |
| DrQ-v2 | $\checkmark$ |
| PUPG   | $\checkmark$ |

## Run

```bash
# simple training
python train.py algo=dqn env=Alien-v5

# not save models while training
python train.py algo=dqn env=Alien-v5 model=False

# change other parameters
python train.py algo=dqn env=CrazyClimber-v5 algo.max_steps=2005000

```
