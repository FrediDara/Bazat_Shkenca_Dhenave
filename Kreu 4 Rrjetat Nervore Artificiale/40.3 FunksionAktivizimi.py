# -*- coding: utf-8 -*-
"""
Funksionet e aktivizimi
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
# Funksioni ReLU
x1 = tf.constant([-10, -5, -1, 0.0, 2, 5, 10], dtype = tf.float32)
y1 = tf.keras.activations.relu(x1).numpy()
  
plt.plot(x1, y1)
plt.show()

# Funksioni SiS/sigmoid
x2 = np.linspace(-20,20,41)
y2 = tf.keras.activations.sigmoid(x2).numpy()
  
plt.plot(x2, y2)
plt.show()

# Funksioni 



# Funksioni 