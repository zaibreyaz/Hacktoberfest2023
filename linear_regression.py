  import numpy as np

  # X is the input features and y is the target output
  X = np.array([...])  
  y = np.array([...])

  # Initialize parameters (slope and intercept)
  m = 0 
  c = 0

  # Learning rate 
  alpha = 0.01

  # Number of iterations
  num_iterations = 1000

  for iteration in range(num_iterations):

      # Predicted output
      y_predicted = m * X + c

      # Cost (MSE)
      cost = (1/len(X)) * np.sum(np.power(y - y_predicted,2))

      # Partial derivatives
      dm = (2/len(X)) * np.sum(X * (y - y_predicted)) 
      dc = (2/len(X)) * np.sum(y - y_predicted)

      # Parameter update
      m = m - alpha * dm
      c = c - alpha * dc
