import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import realax as rx

env = "Pendulum-v1"
obs, act, disc = rx.ENV_SPACES[env]

mdl = rx.training.rl._ppo.ActorCritic(obs, act, disc, 2., key=jr.key(1))
prms, sttcs = eqx.partition(mdl, eqx.is_array)
fctry = lambda prms: eqx.combine(prms, sttcs)

ppo_cfg = rx.PPOConfig(env_name=env)
ppo = rx.PPO(fctry, ppo_cfg, params_init=lambda k: prms)