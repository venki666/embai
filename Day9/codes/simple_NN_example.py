import numpy as np
from tensorflow import keras

# 1. Define the model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# 2. Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# 3. Define the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 4. Train the model
model.fit(xs, ys, epochs=50)

# 5. Make a prediction
# Best practice: pass input as a NumPy array rather than a plain list
print(model.predict(np.array([10.0])))