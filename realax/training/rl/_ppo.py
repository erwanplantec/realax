#-------------------------------------------------------------------
from ...tasks._gymnax import GymnaxTask
from ...tasks._brax import BraxTask

from typing import NamedTuple, Callable, Union, Optional, Tuple
from gymnax.environments.environment import Environment
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree
import optax
import gymnax as gx
import matplotlib.pyplot as plt
from functools import partial
import chex
import distrax
from equinox import filter_jit as jit, filter_vmap as vmap
#-------------------------------------------------------------------


class Config(NamedTuple):
    lr: float=5e-4
    gamma: float=0.99
    num_envs: int=4
    num_steps: int=128
    total_timesteps: int=int(5e5)
    update_epochs: int=4
    num_minibatches: int=4
    gae_lambda: float=0.95
    clip_eps: float=0.2
    ent_coef: float=0.0000
    vf_coef: float=0.5
    max_grad_norm: float=0.5
    env_name: str="CartPole-v1"
    anneal_lr: bool=False
    debug: bool=True

    @property
    def num_updates(self):
        return self.total_timesteps // self.num_steps // self.num_envs
    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches


class LogEnvState(NamedTuple):
    env_state: gx.environments.environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: gx.environments.environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[gx.environments.environment.EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[gx.environments.environment.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class Transition(NamedTuple):
    done: jax.Array
    obs: jax.Array
    value: jax.Array
    action: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    info: PyTree

class TrainState(NamedTuple):
    params: PyTree
    opt_state: PyTree
    obs: jax.Array
    env_states: jax.Array
    dones: jax.Array
    step: int


class PPO:
    #-------------------------------------------------------------------
    def __init__(self, mdl_factory: Callable, env: Environment, config: Config) -> None:
        # ---
        #assert cfg.num_envs%cfg.num_minibatches==0
        # ---
        self.cfg = config
        self.mdl_fctry = mdl_factory
        self.env = LogWrapper(env)
        if config.anneal_lr:
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config.num_minibatches * config.update_epochs))
                    / config.num_updates
                )
                return config.lr * frac
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-4),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-4),
            )
        self.tx = tx
        self.eval_fn = GymnaxTask(env, lambda prms: EvalWrapper(mdl_factory(prms)))
    #-------------------------------------------------------------------
    @eqx.filter_jit
    def collect_trajectory(self, model, key, start_state):

        def _env_step(carry, _):
            obs, env_state, done, key = carry
            key, key_step, key_mdl = jr.split(key, 3)
            pi, val = model(obs)
            action, log_prob = pi.sample_and_log_prob(key_mdl)                                                                          
            new_obs, env_state, reward, done, info = self.env.step(key_step, env_state, action)
            transition = Transition(done=done, obs=obs, value=val, action=action, reward=reward, log_prob=log_prob, info=info)
            return (
                (new_obs, env_state, done, key),
                transition
            )

        k1, k2 = jr.split(key)
        env_state, obs, done = start_state
        state, transitions = jax.lax.scan(
            _env_step,
            (obs, env_state, False, k2),
            None, self.cfg.num_steps
        )
        return state, transitions
    #-------------------------------------------------------------------
    @eqx.filter_jit
    def compute_gae(self, trajs, last_val, last_done):
        def _get_advantages(carry, transition):
            gae, next_value = carry
            done, value, reward = transition.done, transition.value, transition.reward 
            delta = reward + self.cfg.gamma * next_value * (1 - done) - value
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - done) * gae
            return (gae, value), gae
        
        _, advantages = jax.lax.scan(
            _get_advantages, 
            (jnp.zeros_like(last_val), last_val), 
            trajs, reverse=True, unroll=16)
        return advantages, advantages + trajs.value
    #-------------------------------------------------------------------
    def update_step(self, train_state: TrainState, key):
        mdl = self.mdl_fctry(train_state.params)
        # 1. Collect trajectories
        key, _key = jr.split(key)
        keys = jr.split(_key, self.cfg.num_envs)
        states, transitions = vmap(self.collect_trajectory, in_axes=(None,0, 0))(
            mdl, keys, [train_state.env_states, train_state.obs, train_state.dones]
        ) #(n_envs, n_steps, ...)
        last_obs, last_env_states, last_dones,  _ = states 
        # 2. Compute advantages
        key, _key = jr.split(key)
        last_vals = vmap(mdl.critic)(last_obs)
        advantages, targets = vmap(self.compute_gae)(transitions, last_vals, last_dones)
        # 3. Update network
        key, _key = jr.split(key)
        update_state = (train_state, transitions, advantages, targets)
        train_state, losses = self._update_epoch(update_state, _key)
        # 4. Evaluate
        key, _key = jr.split(key)
        eval_score, _ = vmap(self.eval_fn, in_axes=(None,0))(train_state.params, jr.split(_key, 8))
        eval_score = jnp.mean(eval_score)

        return (train_state._replace(obs=last_obs, env_states=last_env_states, dones=last_dones),
            [losses,eval_score]
        )
    #-------------------------------------------------------------------
    def _update_epoch(self, state, key):
        train_state, transitions, advantages, targets = state
        # 1. Get minibatches
        key, key_perm, key_mb = jr.split(key, 3)
        batch = (transitions, advantages, targets) # (num_envs,num_steps,...)
        batch_size = self.cfg.num_envs*self.cfg.num_steps
        permutation = jr.permutation(key_perm, batch_size)
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(
                jnp.reshape(x, (-1, *x.shape[2:])),
                permutation, 
                axis=0), 
            batch)
        assert self.cfg.num_minibatches * self.cfg.minibatch_size == self.cfg.num_envs * self.cfg.num_steps
        minibatches = jax.tree_util.tree_map(lambda x: x.reshape((self.cfg.num_minibatches, self.cfg.minibatch_size, *x.shape[1:])), shuffled_batch)
        train_state, losses = jax.lax.scan(
            self._update_minibatch, 
            train_state,
            minibatches
        )
        return train_state, losses
    #-------------------------------------------------------------------
    def _update_minibatch(self, train_state, batch):
        
        transitions, advantages, targets = batch
        
        def _loss(params, transitions, gae, targets):
            mdl = self.mdl_fctry(params)
            pi, value = vmap(mdl)(transitions.obs)
            log_prob = pi.log_prob(transitions.action)
            entropy = pi.entropy()
            entropy = entropy.mean()
            # 1. Critic loss
            value_pred_clipped = transitions.value + jnp.clip(value-transitions.value, -self.cfg.clip_eps, self.cfg.clip_eps)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
            # 2. Actor loss
            ratio = jnp.exp(log_prob - transitions.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-5)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(
                    ratio, 
                    1-self.cfg.clip_eps, 
                    1+self.cfg.clip_eps)
                * gae
            )
            loss_actor = jnp.mean(-jnp.minimum(loss_actor1, loss_actor2))
            # 3. Compute total loss
            total_loss = loss_actor + self.cfg.vf_coef*value_loss -self.cfg.ent_coef*entropy
            return total_loss, (value_loss, loss_actor, entropy)

        grad_fn = eqx.filter_value_and_grad(_loss, has_aux=True)
        all_loss, grads = grad_fn(train_state.params, transitions, advantages, targets)
        train_state = self.apply_gradients(train_state, grads)

        return train_state, all_loss
    #-------------------------------------------------------------------s
    @eqx.filter_jit
    def train(self, params, key):
        key_init, key_train = jr.split(key)
        rs = self.init_train_state(params, key_init)
        train_keys = jr.split(key_train, self.cfg.num_updates)
        rs, [losses, eval_scores] = jax.lax.scan(self.update_step, rs, train_keys)
        return rs, [losses, eval_scores]
    #-------------------------------------------------------------------
    def init_train_state(self, params, key)->TrainState:
        opt_state = self.tx.init(params)
        obs, env_state = vmap(self.env.reset)(jr.split(key, self.cfg.num_envs))
        return TrainState(
            params=params, 
            opt_state=opt_state, 
            step=0,
            env_states=env_state, 
            obs=obs, 
            dones=jnp.zeros((self.cfg.num_envs,), dtype=bool))
    #-------------------------------------------------------------------
    def apply_gradients(self, train_state: TrainState, grads)->TrainState:
        updates, new_opt_state = self.tx.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        return train_state._replace(params=new_params, opt_state=new_opt_state, step=train_state.step+1)
    #-------------------------------------------------------------------


class NormalDiag(eqx.Module):
    # ---
    loc: jax.Array
    sigma: jax.Array
    # ---
    def __init__(self, loc, sigma):
        self.loc = loc
        self.sigma = sigma
    # ---
    def sample(self, key, shape=()):
        return jr.normal(key, (*shape, *self.loc.shape)) * self.sigma + self.loc 
    # ---
    def log_prob(self, x):
        return jax.scipy.stats.norm.logpdf(x, loc=self.loc, scale=self.sigma).sum(-1)
    # ---
    def sample_and_log_prob(self, key, shape=()):
        x = self.sample(key, shape)
        lp = self.log_prob(x)
        return x, lp
    # ---
    def entropy_(self):
        det = jnp.prod(self.sigma)
        D = self.loc.shape[-1]
        return 0.5 * jnp.log(det) + D/2 * (1+jnp.log(2*jnp.pi))

    def entropy(self):
        # Use log-sum instead of product to avoid numerical instability
        log_sigma = jnp.log(self.sigma)
        D = self.loc.shape[-1]
        return 0.5 * jnp.sum(log_sigma) + D / 2 * (1 + jnp.log(2 * jnp.pi))

    # ---
    def mean(self):
        return self.loc

class ActorCritic(eqx.Module):
    # ---
    actor: nn.MLP
    critic: nn.MLP
    sigma: jax.Array
    discrete: bool
    action_scale: float
    # ---
    def __init__(self, obs_dims, action_dims, discrete=True, action_scale=1., *, key):
        k1, k2 = jr.split(key)
        self.actor = nn.MLP(obs_dims, action_dims, 64, 2, key=k1, activation=jnn.relu)
        self.critic = nn.MLP(obs_dims, "scalar", 64, 2, key=k2, activation=jnn.relu)
        self.discrete = discrete
        self.sigma = jnp.ones((1,))*0.1
        self.action_scale = action_scale
    # ---
    def __call__(self, obs):
        y = self.actor(obs)
        value = self.critic(obs)
        if self.discrete:
            pi = distrax.Categorical(logits=y)
        else:
            sigma = jnp.exp(self.sigma)
            mu = jnn.tanh(y)*self.action_scale
            pi = NormalDiag(jnn.tanh(mu)*2, sigma)
        return pi, value
    # ---
    def initialize(self, key):
        return None

class EvalWrapper(eqx.Module):
    mdl: PyTree
    def __init__(self, model):
        self.mdl = model
    def __call__(self, obs, h, key):
        pi, _ = self.mdl(obs)
        action = pi.sample(key)
        #action = pi.mean()
        return action, None
    def initialize(self, key):
        return self.mdl.initialize(key)

if __name__ == '__main__':

    from ...tasks import ENV_SPACES

    key = jr.key(1)
    k1, k2 = jr.split(key)

    cfg = Config(
    num_envs=8, 
    update_epochs=10, 
    num_minibatches=32, 
    num_steps=2048, 
    lr=3e-4, 
    total_timesteps=int(6e6), 
    env_name="Pendulum-v1", 
    anneal_lr=False)
    
    env, _ = gx.make(cfg.env_name)
    obs_dims, action_dims, typ = ENV_SPACES[cfg.env_name]
    mdl = ActorCritic(obs_dims, action_dims, discrete=typ=="discrete", key=k1)
    
    prms, sttcs = eqx.partition(mdl, eqx.is_array)
    trainer = PPO(lambda p: eqx.combine(p, sttcs), env, cfg)#type:ignore

    rs, [losses, scores] = trainer.train(prms, k2)

    loss, [value_loss, actor_loss, entropy] = losses