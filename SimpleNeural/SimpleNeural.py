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
  def __init__(self, X, y, hidden_layers=None, loss_function='mae', bias=0, validation_percent=.2, validation_set=True, lr=1):
    self.X = np.array(X)
    self.y = np.array(y)
    self.y = self.y.reshape(self.y.shape[0], 1)
    self.loss_function = loss_function
    self.bias = bias
    self.lr = lr
    self.train_scores = []
    self.validation_scores = []
    self.validation_set = validation_set
    if hidden_layers:
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
    self.y_validation = self.y[indexes]

    self.X = np.delete(self.X, indexes, axis=0)
    self.y = np.delete(self.y, indexes, axis=0)

  def generate_layers(self, hidden_layers):
    self.hidden_layers = hidden_layers
    np.random.seed(1)
    self.schema = [self.X.shape[1]] + hidden_layers + [self.y.shape[1]]
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
      self.l0 = self.X
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
          error = self.y_validation - l
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
    for j in range(epochs):
      self.forward()
      self.backward()
      if print_nth_epoch and not j % print_nth_epoch:
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


  # *** WARNING ***
  # The below code is experimental atypical of a neural network.
  # Disregard the below code for now.

  def genetic_alogrithm(self, population_size=30, mutation_rate=0.05, algorithm_epochs=30, nn_epochs=400):

    def make_new_generation(fitness):
      new_generation = []
      new_gen_labels = []
      for i in range(len(fitness)//2):
        parent = [int(j) for j in fitness[i][0].split()]
        child1 = [] # increment layers
        child2 = [] # decrement layers
        for layer_number in parent:
          if random.random() > mutation_rate: # change the number of neurons in each layer
            child1.append(layer_number + 1)
            if layer_number > 1:
              child2.append(layer_number - 1)
        if random.random() > mutation_rate: # add or remove a layer
          child1.append(random.choice(range(1,12)))
          if len(child2) > 2:
            child2.pop()
        for child in [child1, child2]:
          new_generation.append(child)
          new_gen_labels.append(str(child)[1:-1].replace(',', ''))
      return new_generation, new_gen_labels

    def create_initial_population():
      organisms = []
      organism_labels = []
      for i in range(population_size):
        organisms.append(np.random.choice(range(2,50), (random.choice(range(2,10)), )).tolist())
        organism_labels.append(str(organisms[i]).replace(',', '')[1:-1])
      return organisms, organism_labels

    def find_fitness(organisms, organism_labels):
      fitness = []
      for organism, organism_label in zip(organisms, organism_labels):
        self.generate_layers(organism)
        self.train(nn_epochs, None)
        fitness.append([organism_label, self.error])
      fitness = sorted(fitness, key=lambda x: x[1])
      fittest_list.append(fitness[0])
      return fitness

    fittest_list = [] # list of fittest organisms to be plotted.

    organisms, organism_labels = create_initial_population()
    fitness = find_fitness(organisms, organism_labels)

    for _ in range(algorithm_epochs):
      organisms, organism_labels = make_new_generation(fitness)
      fitness = find_fitness(organisms, organism_labels)

    self.x_labels = [label for label, x in fittest_list]
    self.x_fitness = [x for label, x in fittest_list]

    fig, ax = plt.subplots(figsize=(15,5))
    ax = plt.plot(self.x_fitness)
    plt.xticks(range(len(self.x_labels)), self.x_labels, rotation=90)
    plt.show()

  def find_best_bias(self, bias_range=np.arange(-5,5,.1), nn_epochs=400):
    score = []
    for i in bias_range:
      self.generate_layers(self.hidden_layers)
      self.bias = i
      self.train(nn_epochs, None)
      score.append(self.error)

    self.bias_score = list(sorted(zip(score, bias_range), key=lambda x: x[0]))
    self.bias_score_df = pd.DataFrame(self.bias_score, columns=['Error', 'Bias'])

    fig, ax = plt.subplots(figsize=(15,5))
    plt.xticks([x for x in range(len(bias_range))[::2]], [str(round(x,2)) for x in bias_range[::2]], rotation=90)
    ax = plt.plot(score)
    plt.show()
