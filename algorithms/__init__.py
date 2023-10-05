from .dqn import Agent as DQN
from .c51 import Agent as C51
from .rainbow import Agent as Rainbow
from .curl import Agent as CURL
from .drq_v2 import Agent as DrQ_v2
from .pupg import Agent as PUPG
from .ppo import Agent as PPO

atari_agents = {
    "dqn": DQN,
    "c51": C51,
    "rainbow": Rainbow,
    "curl": CURL,
    "drq_v2": DrQ_v2,
    "pupg": PUPG,
    "ppo": PPO,

}
