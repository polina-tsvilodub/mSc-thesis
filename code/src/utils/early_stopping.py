# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        Initialize the early stopping tracker.
        Args: 
        -----
            patience: int 
                how many validation steps to wait before stopping when loss is not improving
            min_delta: float
                minimum difference between new loss and old loss for new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update tracker with new validation loss.

        Args:
        ----
            val_loss: float
                New validation loss.
        """
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True