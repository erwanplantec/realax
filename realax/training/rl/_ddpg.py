from ..base import BaseTrainer

from typing import NamedTuple, Callable, TypeAlias, Union, Optional, Tuple
from gymnax.environments.environment import Environment
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Float, Int, PyTree
import optax
import gymnax as gx
import matplotlib.pyplot as plt
import realax as rx
from functools import partial
from equinox import filter_jit as jit, filter_vmap as vmap
import abc


Transition: TypeAlias = Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

class Actor(eqx.Module):
	#-------------------------------------------------------------------
	mlp: nn.MLP
	#-------------------------------------------------------------------
	def __init__(self, obs_dims, action_dims, width=64, depth=2, *, key):
		self.mlp = nn.MLP(obs_dims, action_dims, width, depth, key=key, final_activation=jnn.tanh)
	#-------------------------------------------------------------------
	def __call__(self, obs):
		y = self.mlp(obs)
		return y
	#-------------------------------------------------------------------

class Critic(eqx.Module):
	#-------------------------------------------------------------------
	mlp: nn.MLP
	#-------------------------------------------------------------------
	def __init__(self, obs_dims, action_dims, width=64, depth=2, *, key):
		self.mlp = nn.MLP(obs_dims+action_dims, "scalar", width, depth, key=key)
	#-------------------------------------------------------------------
	def __call__(self, obs, action):
		return self.mlp(jnp.concatenate([obs, action]))

class Config(NamedTuple):
	env: str="Pendulum-v1"
	n_envs: int=8
	n_steps: int=4
	train_steps: int=1_000_000
	warmup_steps: int=1_000
	lr_actor: float=1e-4
	lr_critic: float=1e-3
	gamma: float=0.99
	tau: float=0.001
	replay_buffer_size: int=100_000
	batch_size: int=64
	max_grad_norm: float=0.5

class ReplayBuffer(NamedTuple):
	# ---
	max_size: int
	counter: Int
	# ---
	obs: jax.Array
	actions: jax.Array
	next_obs: jax.Array
	rewards: jax.Array
	dones: jax.Array
	# ---
	def add(self, transitions):
		obs, actions, next_obs, rewards, dones = jax.tree.map(lambda x: x.reshape((-1,*x.shape[2:])), transitions)
		n = obs.shape[0]
		ids = (jnp.arange(n)+self.counter) % self.max_size
		return self._replace(
			obs = self.obs.at[ids].set(obs),
			actions = self.actions.at[ids].set(actions),
			next_obs = self.next_obs.at[ids].set(next_obs),
			rewards = self.rewards.at[ids].set(rewards),
			dones = self.dones.at[ids].set(dones),
			counter = self.counter+n)
	


class TrainState(NamedTuple):
	# ---
	actor_prms: PyTree
	critic_prms: PyTree
	target_actor_prms: PyTree
	target_critic_prms: PyTree
	opt_state_actor: PyTree
	opt_state_critic: PyTree
	replay_buffer: ReplayBuffer
	obs: jax.Array
	env_state: PyTree
	done: jax.Array
	# ---


class DDPG:
	
	#-------------------------------------------------------------------
	
	def __init__(self, cfg: Config, actor_factory: Callable, critic_factory: Callable):
		self.cfg = cfg

		self.tx_actor = optax.chain(
			optax.clip_by_global_norm(cfg.max_grad_norm),
			optax.adam(cfg.lr_actor, eps=1e-5),
		)
		self.tx_critic = optax.chain(
			optax.clip_by_global_norm(cfg.max_grad_norm),
			optax.adam(cfg.lr_actor, eps=1e-5),
		)

		self.env, self.env_prms = gx.make(cfg.env)

		self.eval_fn = rx.GymnaxTask(cfg.env, actor_factory)

		self.actor_factory = actor_factory
		self.critic_factory = critic_factory
	
	#-------------------------------------------------------------------
	
	def initialize(self, actor_prms: PyTree, critic_prms: PyTree, key: jax.Array):

		key_rst, key_wrmp = jr.split(key)
		obs_dims, action_dims, action_type = rx.tasks.ENV_SPACES[self.cfg.env]
		sz = self.cfg.replay_buffer_size
		replay_buffer = ReplayBuffer(
			max_size = sz, 
			counter = jnp.zeros((), int), 
			obs = jnp.zeros((sz, obs_dims)),
			actions = jnp.zeros((sz,), dtype=int) if action_type=="discrete" else jnp.zeros((sz, action_dims)),
			next_obs = jnp.zeros((sz, obs_dims)),
			rewards = jnp.zeros((sz,)),
			dones = jnp.zeros((sz,), dtype=bool))

		opt_state_actor = self.tx_actor.init(actor_prms)
		opt_state_critic = self.tx_critic.init(critic_prms)

		obs, env_state = jax.vmap(self.env.reset, in_axes=(0,None))(jr.split(key_rst, self.cfg.n_envs), self.env_prms)

		train_state =  TrainState(
			actor_prms, critic_prms, actor_prms, critic_prms, 
			opt_state_actor, opt_state_critic, 
			replay_buffer,
			obs, env_state, jnp.zeros((self.cfg.n_envs,), bool)
		)

		# Warmup: fill replay buffer
		train_state, transitions = jax.lax.scan(partial(self.env_step, warmup=True), train_state, jr.split(key_wrmp, self.cfg.warmup_steps))
		replay_buffer = replay_buffer.add(transitions)
		train_state = train_state._replace(replay_buffer=replay_buffer)

		return train_state
	
	#-------------------------------------------------------------------
	
	def train(self, train_state, key):
		keys = jr.split(key, self.cfg.train_steps)
		train_state, data = jax.lax.scan(self.train_step, train_state, keys)
		return train_state, data
	
	#-------------------------------------------------------------------
	
	@jit
	def init_and_train(self, actor_prms: PyTree, critic_prms: PyTree, key: jax.Array):
		key_init, key_train = jr.split(key)
		train_state = self.initialize(actor_prms, critic_prms, key_init)
		return self.train(train_state, key_train)
	
	#-------------------------------------------------------------------
	
	def train_step(self, train_state, key):
		# 1. Do steps in env
		key, _key = jr.split(key)
		train_state, transitions = jax.lax.scan(
			partial(self.env_step, warmup=False), train_state, jr.split(_key, self.cfg.n_steps)	
		)
		replay_buffer = train_state.replay_buffer.add(transitions)
		# 2. Sample_batch
		key, _key = jr.split(key)
		batch = self.sample_batch(replay_buffer, _key)
		train_state = train_state._replace(replay_buffer=replay_buffer)
		# 3. Update critic
		train_state, critic_loss = self.update_critic(train_state, batch)
		# 4. Update actor
		train_state, actor_loss = self.update_actor(train_state, batch)
		# 5. Update target networks
		train_state = self.update_targets(train_state)
		# 6. Evaluate
		key, _key = jr.split(key)
		eval_score = self.evaluate(train_state, _key)


		return train_state, dict(critic_loss=critic_loss, actor_loss=actor_loss, eval_score=eval_score)
	
	#-------------------------------------------------------------------
	
	def env_step(self, train_state: TrainState, key, warmup:bool=False)->Tuple[TrainState, Transition]:

		actor = self.actor_factory(train_state.actor_prms)
		obs, env_state, done = train_state.obs, train_state.env_state, train_state.done

		key_action, key_step = jr.split(key)
		if warmup:
			action = jax.vmap(self.env.action_space(self.env_prms).sample)(jr.split(key_action,self.cfg.n_envs)) #type:ignore
		else:
			action = jax.vmap(actor)(obs)

		next_obs, env_state, reward, done, _ = jax.vmap(self.env.step, in_axes=(0,0,0,None))(jr.split(key_step, self.cfg.n_envs), env_state, action, self.env_prms)

		transition = (obs, action, next_obs, reward, done)

		return train_state._replace(obs=next_obs, env_state=env_state, done=done), transition #type:ignore
	
	#-------------------------------------------------------------------
	
	def update_actor(self, train_state: TrainState, batch: PyTree)->Tuple[TrainState, Float]:

		@eqx.filter_value_and_grad
		def actor_loss(actor_prms, obs, critic):
			actor = self.actor_factory(actor_prms)
			actions = vmap(actor)(obs)
			q_values = -vmap(critic)(obs, actions)
			return q_values.mean()

		critic = self.critic_factory(train_state.critic_prms)
		loss, grads = actor_loss(train_state.actor_prms, batch[0], critic)
		updates, opt_state = self.tx_actor.update(
			grads, train_state.opt_state_actor, train_state.actor_prms
		)
		prms = optax.apply_updates(train_state.actor_prms, updates)
		return train_state._replace(opt_state_actor=opt_state, actor_prms=prms), loss

	#-------------------------------------------------------------------
	
	def update_critic(self, train_state: TrainState, batch)->Tuple[TrainState, Float]:
		
		@eqx.filter_value_and_grad
		def critic_loss(critic_prms, batch, target_critic, target_actor):

			critic = self.critic_factory(critic_prms)
			obs, actions, next_obs, reward, done = batch	
			target_actions = vmap(target_actor)(next_obs)		
			target = reward + self.cfg.gamma * vmap(target_critic)(next_obs, target_actions) * (1-done)

			pred = vmap(critic)(obs, actions)

			return jnp.mean(jnp.square(pred-target))

		target_critic = self.critic_factory(train_state.target_critic_prms)
		target_actor = self.actor_factory(train_state.target_actor_prms)

		loss, grads = critic_loss(train_state.critic_prms, batch, target_critic, target_actor)
		updates, opt_state = self.tx_critic.update(
			grads, train_state.opt_state_critic, train_state.critic_prms
		)
		prms = optax.apply_updates(train_state.critic_prms, updates)
		return train_state._replace(opt_state_critic=opt_state, critic_prms=prms), loss
	
	#-------------------------------------------------------------------
	
	def update_targets(self, train_state: TrainState)->TrainState:
		tau = self.cfg.tau
		update_fn = lambda a, b: tau * a + (1-tau)*b
		target_critic = jax.tree.map(update_fn, train_state.critic_prms, train_state.target_critic_prms)
		target_actor = jax.tree.map(update_fn, train_state.actor_prms, train_state.target_actor_prms)
		return train_state._replace(target_actor_prms=target_actor, target_critic_prms=target_critic)
	
	#-------------------------------------------------------------------
	
	def sample_batch(self, replay_buffer, key):
		all_ids = jnp.arange(self.cfg.replay_buffer_size)
		mask_ids = jnp.where(all_ids<replay_buffer.counter, 1., 0.)
		prob_ids = mask_ids / mask_ids.sum()
		ids = jr.choice(key, all_ids, (self.cfg.batch_size,), p=prob_ids)
		return (replay_buffer.obs[ids], replay_buffer.actions[ids], replay_buffer.next_obs[ids], replay_buffer.rewards[ids], replay_buffer.dones[ids])
	
	#-------------------------------------------------------------------
	
	def evaluate(self, train_state, key):
		actor = self.actor_factory(train_state.actor_prms)
		def env_step(state, _):
			obs, env_state, done, valid, key = state
			key, _key = jr.split(key)
			action = actor(obs)
			obs, env_state, reward, done, _ = self.env.step(_key, env_state, action, self.env_prms)
			valid = valid * (1-done)
			return [obs, env_state, done, valid, key], reward*valid

		key, key_rst = jr.split(key)
		obs, env_state = self.env.reset(key_rst, self.env_prms)
		_, rewards = jax.lax.scan(env_step, [obs, env_state, jnp.array(False), jnp.array(1.), key], None, 200)
		return rewards.sum()



if __name__ == '__main__':

	cfg = Config(train_steps=100_000, n_steps=8)

	do, da, _ = rx.tasks.ENV_SPACES[cfg.env]

	actor = Actor(do, da, key=jr.key(1))
	actor_prms, actor_sttcs = eqx.partition(actor, eqx.is_array)
	actor_factory = lambda prms: eqx.combine(prms, actor_sttcs)
	critic = Critic(do, da, key=jr.key(2))
	critic_prms, critic_sttcs = eqx.partition(critic, eqx.is_array)
	critic_factory = lambda prms: eqx.combine(prms, critic_sttcs)


	trainer = DDPG(cfg, actor_factory, critic_factory)

	ts = trainer.initialize(actor_prms, critic_prms, jr.key(1))

	ts, losses = trainer.train(ts, jr.key(4))

	fig, ax = plt.subplots(1,3)
	ax[0].plot(losses["actor_loss"])
	ax[1].plot(losses["critic_loss"])
	ax[2].plot(losses["eval_score"])
	plt.show()

