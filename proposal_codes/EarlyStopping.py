import copy

class EarlyStopping:
    """
    Early stop when a monitored metric stops improving.
    mode: "max" for metrics like mAP, "min" for losses
    """
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = -1

    def _is_improvement(self, current, best):
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def step(self, current_metric, model, epoch):
        # First value
        if self.best is None:
            self.best = current_metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False  # don't stop

        # Improvement?
        if self._is_improvement(current_metric, self.best):
            self.best = current_metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1

        # Stop?
        return self.num_bad >= self.patience
