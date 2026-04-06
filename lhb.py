import matplotlib.pyplot as plt
from scipy.datasets import ascent

img = ascent()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
