class MultiTaskAggregator:
    #-------------------------------------------------------------------
    def __init__(self, tasks):
        self.tasks = tasks
    #-------------------------------------------------------------------
    def __call__(self, params, key, data=None):
        total_f = 0.
        eval_data = {}
        for i, task in enumerate(self.tasks):
            f, d = task(params, key, data)
            total_f += f
            eval_data[f"tsk_{i}_fitness"] = f
            eval_data[f"tsk_{i}_data"] = d
        return total_f, eval_data