from ._ppo import PPO, Config as PPOConfig
from ._ddpg import DDPG, Config as DDPGConfig

algos_and_configs = {
	"PPO": (PPO, PPOConfig),
	"DDPG": (DDPG, DDPGConfig)
}