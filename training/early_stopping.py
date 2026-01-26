class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta

        self.best_metric = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        """Update early stopping state.

        Args:
            value (float): Current value of the monitored validation metric.
        Returns:
            should_stop (bool), improved (bool)
        """
        if self.best_metric is None:
            self.best_metric = value
            return False, True

        if value > self.best_metric + self.min_delta:
            self.best_metric = value
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop, False
