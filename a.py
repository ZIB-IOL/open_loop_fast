import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.logspace(1, 4, num=100, base=10)
y = np.log10(x)

# Create the log-log plot
plt.loglog(x, y, '-o')

# Set the markers at evenly spaced locations in logspace
num_markers = 5
marker_indices = np.logspace(0, np.log10(len(x)-1), num=num_markers, base=10, dtype=int)
plt.scatter(x[marker_indices], y[marker_indices], marker='o', color='red', label='Markers')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Display the plot
plt.show()