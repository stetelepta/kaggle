from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import time


preprocessed_data = pickle.load(open("processed_data.pkl", "rb"))
meta_data = pickle.load(open("meta.pkl", "rb"))

x_train = preprocessed_data['x_train']
y_train = preprocessed_data['y_train']
x_val = preprocessed_data['x_val']
y_val = preprocessed_data['y_val']
x_test = preprocessed_data['x_test']


def create_model(nodes=1):
    # setup the model
    model = Sequential()

    # add layers
    model.add(Dense(1, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(nodes, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# setup wrapper so we can use grid search
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
epochs = [1000, 5000]
nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# setup grid
param_grid = dict(nodes=nodes, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

start = time.time()
# fit the model
grid_result = grid.fit(x_train, y_train)
elapsed = time.time() - start

print("elapsed: %.2f" % elapsed)

# print the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
