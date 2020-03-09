import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import MinMaxScaler


def min_max_scale(data):
  scaler = MinMaxScaler()
  scaler.fit(data)
  return scaler.transform(data)


class NN:
  def __init__(self, X, y, hidden_layers, loss_function='mae', bias=0,
               validation_percent=.2, validation_set=True, whole_batch=False):
    self.X = np.array(X)
    self.Y = np.array(y)
    self.Y = self.Y.reshape(self.Y.shape[0], 1)
    self.loss_function = loss_function
    self.bias = bias

    self.train_scores = []
    self.validation_scores = []
    self.validation_set = validation_set
    self.whole_batch = whole_batch

    self.generate_layers(hidden_layers)
    if validation_set:
      self.test_train_split(validation_percent)


  def test_train_split(self, validation_percent):
    if validation_percent >= 1:
      self.validation_percent //= 100
    self.validation_count = int(len(self.X) * validation_percent)
    indexes = np.array([i for i in range(len(self.X))])
    indexes = indexes[:self.validation_count]

    self.X_validation = self.X[indexes]
    self.Y_validation = self.Y[indexes]

    self.X = np.delete(self.X, indexes, axis=0)
    self.Y = np.delete(self.Y, indexes, axis=0)

  def generate_layers(self, hidden_layers):
    self.hidden_layers = hidden_layers
    np.random.seed(1)
    self.schema = [self.X.shape[1]] + hidden_layers + [self.Y.shape[1]]
    self.schema_len = range(len(self.schema[:-1]))
    for i, layer in enumerate(self.schema[:-1]):
      setattr(self, f'W{i}', np.random.uniform(-1, 1, (layer, self.schema[i+1])))
      if i != len(self.schema_len) - 1:
        setattr(self, f'b{i}', np.full((1, self.schema[i+1]), self.bias))


  # sigmoid
  def nonlin(self, x, deriv=False):
    if(deriv==True):
      return x * (1 - x)
    return 1/ (1 + np.exp(-x))


  def forward(self, X=None):
    if str(X) == 'None':
      if self.whole_batch:
        self.l0 = self.X
      else:
        self.l0 = self.x
    else:
      self.l0 = np.array(X)
    for i in self.schema_len:
      l, W = (getattr(self, j) for j in f'l{i} W{i}'.split())
      if i != len(self.schema_len)-1:
        b = getattr(self, f'b{i}')
        setattr(self, f'l{i+1}', self.nonlin(l.dot(W) + b))
      else:
        setattr(self, f'l{i+1}', self.nonlin(l.dot(W)))
    return getattr(self, f'l{len(self.schema_len)}')


  def loss(self, error):
    loss_dict = {
      'mae': lambda x: np.mean(abs(x)),
      'mse': lambda x: np.mean(x**2),
      'hmse': lambda x: np.mean((x**2)*.5),
    }
    loss_function = loss_dict[self.loss_function]
    return loss_function(error)


  def backward(self, back_prop=True, validation=False):
    for i in reversed(self.schema_len):
      i += 1
      l = getattr(self, f'l{i}')
      if i == len(self.schema_len):
        if validation:
          error = self.Y_validation - l
          self.validation_error = self.loss(error)
        else:
          error = self.y - l
          self.error = self.loss(error)
      else:
        delta, W = (getattr(self, j) for j in f'l{i}delta W{i}'.split())
        error = delta.dot(W.T)
      setattr(self, f'l{i-1}delta', error * self.nonlin(l, deriv=True))

    if back_prop: # adjust weights
      for i in reversed(self.schema_len):
        l, W, delta = (getattr(self, j) for j in f'l{i} W{i} l{i}delta'.split())
        delta = l.T.dot(delta)
        setattr(self, f'W{i}', W + delta * self.lr)


  def validate(self):
    self.forward(self.X_validation)
    self.backward(back_prop=False, validation=True)


  def train(self, epochs=1000, print_nth_epoch=100):
    for epoch in range(epochs):
      if self.whole_batch:
        self.x = self.X
        self.y = self.Y
        self.forward()
        self.backward()
      else:
        for i in range(len(self.X)):
          self.x = np.array([self.X[i]])
          self.y = np.array([self.Y[i]])
          self.forward()
          self.backward()
      if print_nth_epoch and not epoch % print_nth_epoch:
        if self.validation_set:
          self.validate()
          self.train_scores.append(self.error)
          self.validation_scores.append(self.validation_error)
          print(f'Test error: {round(self.error, 6)}\t Validation Error: {round(self.validation_error, 6)}')
        else:
          self.train_scores.append(self.error)
          print(f'Test error: {round(self.error, 6)}')


  def predict(self, x):
    return self.forward(x)


  def plot_test_validation(self):
    x1 = self.train_scores
    x2 = self.validation_scores

    fig, ax = plt.subplots(figsize=(20,10))
    ax = plt.plot(x1, label='Training')
    ax = plt.plot(x2, label='Validation')
    fig.legend(prop={'size': 20})
    plt.show()
