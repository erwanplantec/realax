import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, TypeAlias
from jaxtyping import Float, PyTree
from ..training import EvosaxTrainer

import equinox as eqx
import evosax as ex

#-------------------------------------------------------------------
InnerParams: TypeAlias = PyTree
PRNGKey: TypeAlias = jax.Array
Data: TypeAlias = PyTree
InnerTask: TypeAlias = Callable[[InnerParams,PRNGKey,Data|None], Tuple[Float, Data]]
#-------------------------------------------------------------------

class MetaEvoTask:
	#-------------------------------------------------------------------
	def __init__(
		self, 
		inner_task, 
		inner_prms_like, 
		inner_steps=32, 
		inner_pop=16, 
		inner_es="SimpleGA", 
		gen_weights="uniform", 
		pop_aggr="max", 
		inner_eval_reps=1
	):
		self.inner_task = inner_task
		self.inner_prms_like = inner_prms_like
		self.inner_pop = inner_pop
		self.inner_steps = inner_steps
		self.inner_es = inner_es
		self.gen_weights = gen_weights
		self.pop_aggr=pop_aggr
		self.inner_eval_reps = inner_eval_reps
	#-------------------------------------------------------------------
	def __call__(self, outer_prms, key, data=None):
		es_state, data = self.inner_train(outer_prms, key)
		fitnesses = data["metrics"]["fitness"] #(ig,ip)
		
		if self.pop_aggr=="max":
			fits = fitnesses.max(-1) # ig
		elif self.pop_aggr=="avg":
			fits = fitnesses.mean(-1)   
		else:
			raise ValueError    
		
		if self.gen_weights=="uniform":
			fitness = fits.mean()
		elif self.gen_weights=="linear":
			w = jnp.arange(fits.shape[0])
			w = w / w.sum()
			fitness = (fits * w).sum() / w.sum()
		elif self.gen_weights=="exp":
			w = jnp.exp(jnp.arange(fits.shape[0])*.1)
			w = w / w.sum()
			fitness = (fits * w).sum()
		else:
			raise ValueError("not a valid gen_weights")
		return fitness, {"inner_fit":data["metrics"]["fitness"]}
	#-------------------------------------------------------------------
	def inner_train(self, outer_prms, key):
		def task(inner_prms, key, data=None):
			prms = eqx.combine(outer_prms, inner_prms)
			return self.inner_task(prms, key, data)
		trainer = EvosaxTrainer(self.inner_steps, self.inner_es, task, self.inner_prms_like, 
								popsize=self.inner_pop, fitness_shaper=ex.FitnessShaper(maximize=True), 
								eval_reps=self.inner_eval_reps)
		state, data = trainer.init_and_train(key)
		return state, data