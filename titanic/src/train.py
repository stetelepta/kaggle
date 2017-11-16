from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import callbacks
from sklearn.model_selection import GridSearchCV
from utils.monitoring_utils import KerasClassifierTB
import multiprocessing
import pickle
import time

preprocessed_data = pickle.load(open("processed_data.pkl", "rb"))
meta_data = pickle.load(open("meta.pkl", "rb"))

x_train = preprocessed_data['x_train']
y_train = preprocessed_data['y_train']
x_val = preprocessed_data['x_val']
y_val = preprocessed_data['y_val']
x_test = preprocessed_data['x_test']


def create_model(nodes=1, learning_rate=0.001):
    # setup the model
    model = Sequential()

    # add layers
    model.add(Dense(1, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(nodes, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# callbacks
# remote_cb = callbacks.RemoteMonitor(root='http://localhost:9000', headers=None)
# tensorboard_cb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
early_stopping_cv = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

# setup wrapper so we can use grid search
model = KerasClassifierTB(build_fn=create_model, verbose=0)

# define the grid search parameters
epochs = [1000]
learning_rate = [0.1]
nodes = [1, 2, 3]

nr_cpu = multiprocessing.cpu_count()

# setup grid
param_grid = dict(nodes=nodes, epochs=epochs, learning_rate=learning_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=nr_cpu, cv=2)

start = time.time()
# fit the model
grid_result = grid.fit(x_train, y_train, log_dir='./logs')


elapsed = time.time() - start

print("elapsed: %.2f" % elapsed)

# print the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
