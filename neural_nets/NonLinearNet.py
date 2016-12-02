import numpy as np
import matplotlib.pyplot as plt
import math

class NonLinearNet(object):
    
  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    self.params['l1'] = np.zeros((input_size, hidden_size))
    
  def forward(self, X):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    z1 = np.dot(X, W1) + b1
    l1 = 1 / (1 + np.exp(-z1))
    self.params['l1'] = l1
    scores = np.dot(l1, W2) + b2
    return scores

  def loss(self, X, y=None, reg=0.0):
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = self.forward(X)

    # Compute the loss
    loss = 0
    
    # Use L1 distance
    dout = np.abs(y - scores)
    bp_mask = np.vectorize(lambda x: 1 if x > 0 else -1)(y - scores)
    loss += np.sum(dout) / N
    loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    dl2 = bp_mask 
    
    # Use L2 distance
    #dout = (y - scores) ** 2
    #loss += 0.5 * np.sum(dout) / N
    #loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    #dl2 = y - scores
    
    # Backward pass: compute gradients
    grads = {}
    l1 = self.params['l1']
    
    grads['W2'] = -np.dot(l1.T, dl2)
    grads['W2'] /= N
    grads['W2'] += reg * W2
    grads['b2'] = -(float)(np.sum(dl2)) / N

    dl1 = -(np.dot(dl2, W2.T)) * l1 * (1 - l1) / N
    grads['W1'] = np.dot(X.T, dl1)
    grads['W1'] += reg * W1
    grads['b1'] = np.sum(dl1, axis = 0)

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=500, verbose=False):
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      batch_ix = np.random.choice(num_train, batch_size)
      X_batch = X[batch_ix]
      y_batch = y[batch_ix]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']

      if verbose and it % 50 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Decay learning rate
        learning_rate *= learning_rate_decay
    
    # Use the validation set to validate
    print 'X_val :'
    print X_val[:10]
   
      #y_test[i] = 2 * X[i, 0] - 1 * X[i, 1]
    print 'y expected:'
    print y_val[:10]
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    
    scores = self.forward(X_val)
    print 'y get:'
    print scores[:10]
    
    return loss_history


