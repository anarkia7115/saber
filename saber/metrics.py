import os
import codecs
from statistics import mean
from operator import itemgetter

import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

from utils_models import precision_recall_f1_support

# TODO (johngiorgi): there is some hard coded ugliness going on in print_table, fix this.
# TODO (johngiorgi): this is likely copying big lists, find a way to get around this

class Metrics(Callback):
    """ A class for handling performance metrics, inherits from Callback. """
    def __init__(self,
                 X_train,
                 X_valid,
                 y_train,
                 y_valid,
                 tag_type_to_idx,
                 output_dir):
        # training data
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

        self.tag_type_to_idx = tag_type_to_idx

        # get inversed mapping from idx: tag, speeds up computations downstream
        self.idx_to_tag_type = {v: k for k, v in tag_type_to_idx.items()}

        self.output_dir = output_dir

        # epoch counter for model tied to this object
        self.current_epoch = 0

        # Model performance metrics accumulators
        self.train_performance_metrics_per_epoch = []
        self.valid_performance_metrics_per_epoch = []

    def on_train_begin(self, logs={}):
        """Series of steps to perform when training begins."""
        pass

    def on_epoch_end(self, epoch, logs={}):
        """Series of steps to perform when epoch ends."""
        # get train/valid performance metrics
        train_scores = self._eval(self.X_train, self.y_train)
        valid_scores = self._eval(self.X_valid, self.y_valid)

        self._print_performance_scores(train_scores, title='train')
        self._print_performance_scores(valid_scores, title='valid')

        # accumulate peformance metrics
        self.train_performance_metrics_per_epoch.append(train_scores)
        self.valid_performance_metrics_per_epoch.append(valid_scores)

        # write the performance metrics for the current epoch to disk
        self._write_train_metrics()

        self.current_epoch += 1 # update the current epoch counter

    def _eval(self, X, y):
        """ Performs all evaluation steps for given X (input) and y (labels).

        For a given input (X) and labels (y) performs all the steps in the
        evaluation pipeline, namely: performs prediction on X, chunks the
        annotations by type, and computes performance scores by type.

        Args:
            X: input matrix, of shape (num examples X sequence length)
            y: lables, of shape (num examples X sequence length X num classes)
        """
        # get predictions and gold labels
        y_true, y_pred = self._get_y_true_and_pred(X, y)
        # convert idx sequence to tag sequence
        y_true_tag = [self.idx_to_tag_type[idx] for idx in y_true]
        y_pred_tag = [self.idx_to_tag_type[idx] for idx in y_pred]
        # chunk the entities
        y_true_chunks = self._chunk_entities(y_true_tag)
        y_pred_chunks = self._chunk_entities(y_pred_tag)

        # get performance scores per label
        performance_scores = self._get_precision_recall_f1_support(y_true_chunks,
                                                                   y_pred_chunks)

        return performance_scores

    def _get_y_true_and_pred(self, X, y):
        """ Get y_true and y_pred for given input data (X) and labels (y)

        Performs prediction for the current model (self.model), and returns
        a 2-tuple contain 1D array-like objects containing the true (gold)
        labels and the predicted labels, where labels are integers corresponding
        to the sequence tags as per self.tag_type_to_idx.

        Args:
            X: input matrix, of shape (num examples X sequence length)
            y: lables, of shape (num examples X sequence length X num classes)

        Returns:
            y_true: 1D array like object containing the gold label sequence
            y_pred: 1D array like object containing the predicted sequences
        """
        # gold labels
        y_true = y.argmax(axis=-1) # get class label
        y_true = np.asarray(y_true).ravel() # flatten to 1D array
        # predicted labels
        y_pred = self.model.predict(X).argmax(axis=-1)
        y_pred = np.asarray(y_pred).ravel()

        # sanity check
        assert y_true.shape == y_pred.shape, """y_true and y_pred have different
        shapes"""

        return y_true, y_pred

    def _chunk_entities(self, seq):
        """Chunks enities in the BIO or BIOES format.

        For a given sequence of entities in the BIO or BIOES format, returns
        the chunked entities.

        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            >>> seq = ['B-PRGE', 'I-PRGE', 'O', 'B-PRGE']
            >>> print(get_entities(seq))
            [('PRGE', 0, 2), ('PRGE', 3, 4)]
        """
        i = 0
        chunks = []
        seq = seq + ['O']  # add sentinel
        types = [tag.split('-')[-1] for tag in seq]
        while i < len(seq):
            if seq[i].startswith('B'):
                for j in range(i+1, len(seq)):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
                chunks.append((types[i], i, j))
                i = j
            else:
                i += 1
        return chunks

    def _get_precision_recall_f1_support(self, y_true, y_pred):
        """Returns precision, recall, f1 and support.

        For given gold (y_true) and predicited (y_pred) labels, returns the
        precision, recall, f1 and support per label and the average across
        labels. Expected y_true and y_pred to be a sequence of entity chunks.

        Args:
            y_true: list of (chunk_type, chunk_start, chunk_end)
            y_pred: list of (chunk_type, chunk_start, chunk_end)
        Returns:
            dict: dictionary containing (precision, recall, f1, support) for
            each chunk type and the average across chunk types.
        """
        performance_scores = {} # dict accumulator of per label of scores
        # micro performance accumulators
        FN_total = 0
        FP_total = 0
        TP_total = 0

        labels = list(set([chunk[0] for chunk in y_true])) # unique labels

        # get performance scores per label
        for lab in labels:
            # get chunks for current lab
            y_true_lab = [chunk for chunk in y_true if chunk[0] == lab]
            y_pred_lab = [chunk for chunk in y_pred if chunk[0] == lab]

            # per label performance accumulators
            FN = 0
            FP = 0
            TP = 0

            # FN
            for gold in y_true_lab:
                if gold not in y_pred_lab:
                    FN += 1

            for pred in y_pred_lab:
                # FP / TP
                if pred not in y_true_lab:
                    FP += 1
                # TP
                elif pred in y_true_lab:
                    TP += 1

            # get performance metrics
            performance_scores[lab] = precision_recall_f1_support(TP, FP, FN)

            # accumulate FNs, FPs, TPs
            FN_total += FN
            FP_total += FP
            TP_total += TP

        # get macro and micro peformance metrics averages
        macro_p = mean([v[0] for v in performance_scores.values()])
        macro_r = mean([v[1] for v in performance_scores.values()])
        macro_f1 = mean([v[2] for v in performance_scores.values()])
        total_support = TP_total + FN_total

        performance_scores['MACRO_AVG'] = (macro_p, macro_r, macro_f1, total_support)
        performance_scores['MICRO_AVG'] = precision_recall_f1_support(TP_total, FP_total, FN_total)

        return performance_scores

    def _print_performance_scores(self, performance_scores, title='train'):
        """Prints a table of performance scores.

        Args:
            performance_scores: a dictionary of label, score pairs where label
                                is a sequence tag and scores is a 4-tuple
                                containing precision, recall, f1 and support
        """
        # collect table dimensions
        col_width = 4
        col_space = ' ' * col_width
        len_longest_label = max(len(s) for s in performance_scores)
        dist_to_col_1 = len_longest_label + col_width
        dist_to_col_2 = dist_to_col_1 + len('Precision') + col_width
        dist_to_col_3 = dist_to_col_2 + len('Recall') + col_width
        dist_to_col_4 = dist_to_col_3 + len('F1') + col_width

        tab_width = dist_to_col_4 + len('Support') + col_width
        light_line = '-' * tab_width
        heavy_line = '=' * tab_width

        ## HEADER
        print()
        print()
        title = '{col}{t}{col}'.format(t=title.upper(),
                                       col=' '*((tab_width-len(title))//2))
        header = 'Label{col1}Precision{cols}Recall{cols}F1{cols}Support'.format(col1=' ' * (dist_to_col_1 - len('Label')),
                                                                                cols=col_space)
        print(heavy_line)
        print(title)
        print(light_line)
        print(header)

        print(heavy_line)
        ## BODY
        for label, score in performance_scores.items():
            # specify an entire row
            row = '{lab}{col1}{p:.2%}{col2}{r:.2%}{col3}{f1:.2%}{col4}{s}'.format(
                p=score[0],
                r=score[1],
                f1=score[2],
                s=score[3],
                lab=label,
                col1=' ' * (dist_to_col_1 - len(label) + len('Precision')//3 - 2),
                col2=' ' * (dist_to_col_2 - dist_to_col_1 - len('Precision')//3 - 2),
                col3=' ' * (dist_to_col_3 - dist_to_col_2 - len('Precision')//3 - 2),
                col4=' ' * (dist_to_col_4 - dist_to_col_3 - len('Precision')//3))
            print(row)
        print(heavy_line)

    def _write_train_metrics(self):
        """
        """
        # create output filepath
        perf_metrics_filename = 'epoch_{0:03d}.txt'.format(self.current_epoch + 1)
        eval_file_dirname = os.path.join(self.output_dir, perf_metrics_filename)
        # write performance metrics for current epoch to file
        micro_avg_per_epoch = [x['MICRO_AVG'] for x in self.valid_performance_metrics_per_epoch]
        macro_avg_per_epoch = [x['MACRO_AVG'] for x in self.valid_performance_metrics_per_epoch]

        best_micro_avg_val_score = max(micro_avg_per_epoch, key=itemgetter(2))
        best_macro_avg_val_score = max(macro_avg_per_epoch, key=itemgetter(2))
        best_micro_epoch = micro_avg_per_epoch.index(best_micro_avg_val_score)
        best_macro_epoch = macro_avg_per_epoch.index(best_macro_avg_val_score)
        best_micro_val_score = self.valid_performance_metrics_per_epoch[best_micro_epoch]
        best_macro_val_score = self.valid_performance_metrics_per_epoch[best_macro_epoch]

        with open(eval_file_dirname, 'a') as f:
            f.write(str(self.valid_performance_metrics_per_epoch[self.current_epoch]))
            f.write('\n')
            f.write('Best performing epoch based on macro average: {}\n'.format(best_macro_epoch + 1))
            f.write(str(best_macro_val_score))
            f.write('\n')
            f.write('Best performing epoch based on micro average: {}\n'.format(best_micro_epoch + 1))
            f.write(str(best_micro_val_score))
            f.write('\n')