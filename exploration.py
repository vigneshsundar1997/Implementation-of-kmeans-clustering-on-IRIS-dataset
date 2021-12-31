from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_raw = pd.read_csv("digits-raw.csv",header=None)
data_embedded = pd.read_csv("digits-embedding.csv",header=None)
digits_raw_array = data_raw.to_numpy()
digits_embedded_array = data_embedded.to_numpy()

np.random.seed(0)

#visualize each digit as a 28*28 grayscale matrix
for digit in np.unique(digits_raw_array[:,1]):
    digit_class = digits_raw_array[np.where(digits_raw_array[:,1]==digit)]
    random_digit = np.random.choice(digit_class.shape[0], size=1)
    image_array = digit_class[random_digit,: ][: ,2:786]
    image_array = image_array.reshape(28,28)
    plt.imshow(image_array,cmap="gray")
    plt.show()

random_embedded_digits = digits_embedded_array[np.random.randint(0, data_embedded.shape[0], size=1000),:]

plt.close()

#visualize 1000 random samples with their clusters
for digit in np.unique(digits_raw_array[:,1]):
    digit_class = random_embedded_digits[np.where(random_embedded_digits[:,1]==digit)]
    digit_class_x = digit_class[:,2]
    digit_class_y = digit_class[:,3]
    plt.scatter(digit_class_x,digit_class_y,label=digit)
plt.legend()
plt.show()