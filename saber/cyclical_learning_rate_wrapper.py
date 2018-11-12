from keras_contrib.callbacks import CyclicLR
import csv

class CyclicLRWRapper(CyclicLR):
    '''
    Wrapper for CyclicLR to overwrite the on_epoch_end function
    Saves cyclic learning info in the same format as the rest of the callbacks in saber
    '''
    def __init__(self, min_lr, max_lr, step_size, mode, output_dir):
        super(CyclicLRWRapper, self).__init__(base_lr=min_lr, max_lr=max_lr, step_size = step_size, mode=mode)
        self.output_dir = output_dir

#TODO add functionality to save results at the end of training