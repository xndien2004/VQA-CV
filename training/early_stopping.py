class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta

        self.best_f1 = None
        self.counter = 0
        self.should_stop = False

    def step(self, f1):
        if self.best_f1 is None:
            self.best_f1 = f1
            return False, True

        if f1 > self.best_f1 + self.min_delta:
            self.best_f1 = f1
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop, False
