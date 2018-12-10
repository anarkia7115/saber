import json
import os

from keras_contrib.callbacks import CyclicLR

import matplotlib.pyplot as plt


class CyclicLRWRapper(CyclicLR):
    '''
    Wrapper for CyclicLR to overwrite the on_epoch_end function
    Saves cyclic learning info in the same format as the rest of the callbacks in saber
    '''
    def __init__(self, min_lr, max_lr, step_size, mode, output_dir, lr_find=False):
        super(CyclicLRWRapper, self).__init__(base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
        self.output_dir = output_dir
        self.lr_find = lr_find

    def on_train_end(self, logs):
        '''
        when training is finished, save cyclic learning data and plots in the output directory
        '''
        #save data
        #need to change data to floats before saving
        for key in self.history.keys():
            self.history[key] = [float(v) for v in self.history[key]]
        with open(self.output_dir + 'cyclic_lr_data.txt', 'a') as eval_file:
            eval_file.write(json.dumps(self.history, indent=4))

        #iterations vs lr
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        plt.title('iterations vs lr')
        plt.savefig(self.output_dir+'iterations_vs_lr.png')
        plt.close()

        #save plots of data
        if self.lr_find:
            #lr vs loss
            plt.plot(self.history['lr'], self.history['loss'])
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.title('learning rate vs loss')
            plt.savefig('lr_vs_loss.png')
            plt.close()
        else:
            # iterations vs loss
            plt.plot(self.history['iterations'], self.history['loss'])
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title('iterations vs loss')
            plt.savefig(os.path.join(self.output_dir, 'iterations_vs_loss.png'))
            plt.close()
        return
