import jax
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
        inner_task: InnerTask, 
        inner_prms_like: InnerParams, 
        inner_steps: int=32, 
        inner_pop: int=16, 
        inner_es: str="SimpleGA"):
        """MetaEvoTask
        
        Args:
            inner_task (Task): Task to be optimized on in the inner loop by the inner optimizer
            inner_prms_like (PyTree): PyTree definition of innner params
            inner_steps (int, optional): number of optimization steps in inner loop
            inner_pop (int, optional): number 
            inner_es (str, optional): Description
        """
        self.inner_task = inner_task
        self.inner_prms_like = inner_prms_like
        self.inner_pop = inner_pop
        self.inner_steps = inner_steps
        self.inner_es = inner_es

    #-------------------------------------------------------------------

    def __call__(self, outer_prms, key, data=None):
        _, data = self.inner_train(outer_prms, key)
        fitnesses = data["metrics"]["fitness"]
        return fitnesses.max(-1).mean(), data

    #-------------------------------------------------------------------

    def inner_train(self, outer_prms, key):
        def task(inner_prms, key, data=None):
            prms = eqx.combine(outer_prms, inner_prms)
            return self.inner_task(prms, key, data)
        trainer = EvosaxTrainer(self.inner_steps, self.inner_es, task, self.inner_prms_like, 
                                popsize=self.inner_pop, fitness_shaper=ex.FitnessShaper(maximize=True))
        state, data = trainer.init_and_train(key)
        return state, data

    #-------------------------------------------------------------------