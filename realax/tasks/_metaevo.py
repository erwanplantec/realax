from ..training import EvosaxTrainer

import equinox as eqx
import evosax as ex

class MetaEvoTask:
    def __init__(self, inner_task, inner_prms_like, inner_steps=32, inner_pop=16, inner_es="SimpleGA"):
        self.inner_task = inner_task
        self.inner_prms_like = inner_prms_like
        self.inner_pop = inner_pop
        self.inner_steps = inner_steps
        self.inner_es = inner_es
    def __call__(self, outer_prms, key, data=None):
        es_state, data = self.inner_train(outer_prms, key)
        fitnesses = data["metrics"]["fitness"]
        return fitnesses.max(-1).mean(), data
    def inner_train(self, outer_prms, key):
        def task(inner_prms, key, data=None):
            prms = eqx.combine(outer_prms, inner_prms)
            return self.inner_task(prms, key, data)
        trainer = EvosaxTrainer(self.inner_steps, self.inner_es, task, self.inner_prms_like, 
                                popsize=self.inner_pop, fitness_shaper=ex.FitnessShaper(maximize=True))
        state, data = trainer.init_and_train(key)
        return state, data