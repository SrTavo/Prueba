import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv_kernel(matrix, kernel):

    matrix_row, matrix_col = matrix.shape
    kernel_row, kernel_col = kernel.shape

    suma = 0

    for row in range(matrix_row):
        for col in range(matrix_col):
            suma += matrix[row, col] * kernel[row, col]

    return suma


def convolution(image, kernel):

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    convol = np.zeros(image.shape)

    for row in range(image_row):
        for col in range(image_col):
            convol[row, col] = conv_kernel(image[row:row + kernel_row, col:col + kernel_col], kernel)

    plt.imshow(convol, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
 
    return convol