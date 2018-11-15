import keras as keras
from keras import backend as K
import numpy as np
import warnings
import glob
import os
from keras.models import load_model
import pickle


class JZTrainCategory(keras.callbacks.Callback):
    def __init__(self, filepath, nb_epochs=20, nb_snapshots=1, monitor='val_loss', factor=0.1, verbose=1, patience=1,
                    save_weights_only=False,
                    decay_factor_value=1.0,
                    mode='auto', period=1):
        super(JZTrainCategory, self).__init__()
        self.nb_epochs = nb_epochs
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.init_factor = factor
        self.decay_factor_value = decay_factor_value
        self.factor = factor
        self.save_weights_only = save_weights_only
        self.patience = patience
        self.r_patience = 0
        self.check = nb_epochs // nb_snapshots
        self.monitor_val_list = []
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.init_best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.init_best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.init_best = -np.Inf
            else:
                self.monitor_op = np.less
                self.init_best = np.Inf

    @staticmethod
    def compile_official_f1_score(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, 10))
        y_true = K.cast(y_true, 'float32')
        y_pred = K.round(y_pred)

        tp = K.sum(y_pred * y_true)
        fp = K.sum(K.cast(K.greater(y_pred - y_true, 0.), 'float32'))
        fn = K.sum(K.cast(K.greater(y_true - y_pred, 0.), 'float32'))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2*p*r/(p+r)
        return f

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_train_begin(self, logs={}):
        self.init_lr = K.get_value(self.model.optimizer.lr)
        self.best = self.init_best
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        n_recurrent = epoch // self.check
        self.save_path = '{}/{}.h5'.format(self.filepath, n_recurrent)
        os.makedirs(self.filepath, exist_ok=True)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)

        else:
            if self.monitor_op(current, self.best):
                # if better result: save model
                self.r_patience = 0
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch + 1, self.monitor, self.best,
                             current, self.save_path))
                self.best = current
                if self.save_weights_only:
                    self.model.save_weights(self.save_path)
                    # pickle.dump(self.model.get_weights(), open('./debug_weight.pkl', 'wb'))
                    symbolic_weights = getattr(self.model.optimizer, 'weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    with open('{}/optimizer.pkl'.format(self.filepath), 'wb') as f:
                        pickle.dump(weight_values, f)
                else:
                    self.model.save(self.save_path)

            else:
                # if worse resule: reload last best model saved
                self.r_patience += 1
                if self.verbose > 0:
                    if self.r_patience == self.patience:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                            (epoch + 1, self.monitor, self.best))
                        if self.save_weights_only:
                            self.model.load_weights(self.save_path)
                            self.model._make_train_function()
                            with open('{}/optimizer.pkl'.format(self.filepath), 'rb') as f:
                                weight_values = pickle.load(f)
                            self.model.optimizer.set_weights(weight_values)
                        else:
                            self.model = load_model(self.save_path, custom_objects={'compile_official_f1_score': JZTrainCategory.compile_official_f1_score})
                        # set new learning rate
                        old_lr = K.get_value(self.model.optimizer.lr)
                        new_lr = old_lr * self.factor
                        self.factor *= self.decay_factor_value  # 衰减系数衰减
                        K.set_value(self.model.optimizer.lr, new_lr)
                        print('\nReload model and decay learningrate from {} to {}\n'.format(old_lr, new_lr))
                        self.r_patience = 0

        if (epoch+1) % self.check == 0:
            self.monitor_val_list.append(self.best)
            self.best = self.init_best
            self.factor = self.init_factor

            if (epoch+1) != self.nb_epochs:
                K.set_value(self.model.optimizer.lr, self.init_lr)
                print('At epoch-{} reset learning rate to mountain-top init lr {}'.format(epoch+1, self.init_lr))

