from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard, EarlyStopping
import itertools
import os


class KerasClassifierTB(KerasClassifier):

    def __init__(self, *args, **kwargs):
        super(KerasClassifierTB, self).__init__(*args, **kwargs)

    def fit(self, x, y, log_dir=None, **kwargs):
        cbs = None
        if log_dir is not None:
            # Make sure the base log directory exists
            try:
                os.makedirs(log_dir)
            except OSError:
                pass
            params = self.get_params()
            params['build_fn'] = params['build_fn'].__name__
            conf = ",".join("{}={}".format(k, params[k]) for k in sorted(params))
 
            conf_dir_base = os.path.join(log_dir, conf)
            # Find a new directory to place the logs
            for i in itertools.count():
                try:
                    conf_dir = "{}_split-{}".format(conf_dir_base, i)
                    os.makedirs(conf_dir)
                    break
                except OSError:
                    pass
            
            cbs = [
                TensorBoard(log_dir=conf_dir, histogram_freq=0, write_graph=True, write_images=True),
                EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')
            ]
        super(KerasClassifierTB, self).fit(x, y, callbacks=cbs, **kwargs)