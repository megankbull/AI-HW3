import numpy as np 
import pandas as pd 
from scipy.special import expit, softmax
import sys

BATCH_SIZE = 32
NUM_EPOCHS = 30
L_RATE = 0.01
# would need to adjust the following for a network where NUM_LAYERS!=3
# num of neurons per layer
L1 = 512
L2 = 256
L3 = 10

class Layer:
   def __init__(self, p_layer_neurons, curr_layer_neurons, output_layer=False):
      self.weights = 0.1 * np.random.randn(curr_layer_neurons, p_layer_neurons)
      self.biases = np.zeros((curr_layer_neurons, 1))
      self.output_layer = output_layer

   def forward(self, train_data):
      self.inputs = train_data 
      self.pre_activ = self.weights @ train_data + self.biases
      
      if self.output_layer: self.output = softmax(self.pre_activ, axis=0)
      
      else: self.output = expit(self.pre_activ)
      
      return self.output

   def back(self, d_vals, batch_size): 

      self.dweights = (d_vals @ self.inputs.T) / batch_size
      self.dbiases = np.sum(d_vals, axis=1, keepdims=True) / batch_size
      self.dinputs = self.weights.T @ d_vals
      self.sig_derv = self.dinputs * self.inputs * (1-self.inputs)
      self.soft_derv = self.output * (1 - self.output) if self.output_layer else None
      
      return self.sig_derv

   def get_params(self): 
      return self.weights, self.biases

   def set_params(self, new_weights, new_biases): 
      self.weights = new_weights
      self.biases = new_biases

class Model:
   def __init__(self):
      self.batch_size = BATCH_SIZE
      self.learning_rate = L_RATE
      self.num_epochs = NUM_EPOCHS

      self.layers = []
      
   def add_layer(self, layer):
      self.layers.append(layer)

   # split data into shuffled batches 
   def batch_data(self, train_data, train_labels): 
      train_labels = self.convert_onehot(train_labels.copy())
      # train_labels = train_labels.copy()
      train_data = train_data.copy()

      n_samps = train_data.shape[1]

      rng = np.random.default_rng()
      shuffled = np.arange(n_samps)
      rng.shuffle(shuffled)
      train_data = train_data[:,shuffled]
      train_labels = train_labels[:,shuffled]

      n_batches = n_samps // self.batch_size if n_samps > self.batch_size else 1
      batch_data = np.array_split(train_data, n_batches, axis=1)
      batch_labels = np.array_split(train_labels, n_batches, axis=1)
      self.batches = tuple(zip(batch_data, batch_labels))
      
   # converts one hot encodes target labels
   def convert_onehot(self, targ_labels): 
      n_classes = L3
      n_samps = targ_labels.size
      hot = np.zeros((n_samps, n_classes))
      # set correct classif to 1
      for i in range(n_samps):
         hot[i, targ_labels[:, i]] = 1
      return hot.T

   def forward(self, train_data): 
      next_inputs = train_data

      for layer in self.layers: 
         next_inputs = layer.forward(next_inputs)

      self.pred_labels = next_inputs
      return self.pred_labels

   def calc_loss(self, targ_labels, epoch_num, batch_num):
      if targ_labels.shape != self.pred_labels.shape:
         print("epoch num: ", epoch_num)
         print("batch num: ", batch_num)
         # exit()
      return self.pred_labels - targ_labels

   def back(self, targ_labels, train_data, epoch_num, batch_num): 

      batch_size = train_data[1].shape
      prev_loss = self.calc_loss(targ_labels,epoch_num, batch_num)

      for layer in reversed(self.layers):
         prev_loss = layer.back(prev_loss, batch_size)

   def update_params(self): 
      for layer in self.layers: 
         weights, biases = layer.get_params()

         new_weights = weights - (self.learning_rate * layer.dweights)
         new_biases = biases - (self.learning_rate * layer.dbiases)

         layer.set_params(new_weights, new_biases)
   
   def train_model(self, train_data, train_labels):
      n_features = len(train_data)

      self.add_layer(Layer(n_features, L1))
      self.add_layer(Layer(L1, L2))
      self.add_layer(Layer(L2, L3, True))

      for i in range(self.num_epochs):
         self.batch_data(train_data, train_labels)
         pred_acc = []
         for j in range(len(self.batches)):
         # for batch in self.batches:
            data, labels = self.batches[j]

            self.forward(data)
            self.back(labels, data, i, j)
            self.update_params()
            pred_acc = np.append(pred_acc, np.argmax(self.pred_labels, axis=0)== np.argmax(labels, axis=0))
         print(i, ":", np.mean(pred_acc))
            
# train_data = pd.read_csv('train_image1.csv', header=None).to_numpy().T
# train_labels = pd.read_csv('train_label1.csv', header=None).to_numpy().T
# test_data = pd.read_csv('test_image.csv', header=None).to_numpy().T

train_data = pd.read_csv(sys.argv[1], header=None).to_numpy().T
train_labels = pd.read_csv(sys.argv[2], header=None).to_numpy().T
test_data = pd.read_csv(sys.argv[3], header=None).to_numpy().T

model = Model()
model.train_model(train_data, train_labels)

test_preds = np.argmax(model.forward(test_data), axis=0)
# test_labels = pd.read_csv('test_label.csv', header=None).to_numpy().T
# test_acc = []
# test_acc = np.append(test_acc, test_preds==test_labels)
# print(np.mean(test_acc))

np.savetxt('test_predictions.csv', test_preds, fmt='%1d')