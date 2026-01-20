from simulation import generate_beam_data
from model import build_model
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate data
print("Generating Data...")
X, y = generate_beam_data(num_samples=1000)

# Normalize the data
X = X / np.max(X)

# 2. Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Build and train the model
model = build_model((64, 64))
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# 4. Test and display the result
prediction = model.predict(X_test[:1])
print(f"Actual Distance: {y_test[0]}, Predicted: {prediction[0][0]}")

# 5. Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Convergence')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# 6. Visualize a test sample and its prediction
sample_idx = 0
test_input = X_test[sample_idx:sample_idx+1]
actual_z = y_test[sample_idx]
predicted_z = model.predict(test_input)[0][0]

plt.imshow(X_test[sample_idx], cmap='jet')
plt.title(f"Actual Z: {actual_z:.2f}cm\nPredicted Z: {predicted_z:.2f}cm")
plt.colorbar(label='Intensity')
plt.show()