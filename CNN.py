import numpy as np
import matplotlib.pyplot as plt
import gzip
import sys
import pickle
from tensorflow.keras.datasets import mnist
np.set_printoptions(threshold=sys.maxsize)

def load_mnist_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()

    return data

class relu:
    @staticmethod
    def activation(x):
        y = np.maximum(0, x)
        return y

    @staticmethod
    def prime(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class CNN:
    def __init__(self):
        self.self = self

    def GaussianBlur_1(self):
        a = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]])
        k = a / np.sum(a)
        return k

    def edge_detection(self):
        a = np.array([[1, 0, -1],
                      [0, 0, 0],
                      [-1, 0, 1]])
        return a

    def sharpen(self):
        a = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
        return a

    def GaussianBlur_2(self):
        a = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])
        return a

    def sobel_edge_1(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
        return a

    def sobel_edge_2(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]).T
        return a

    def laplacian(self):
        a = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
        return a

    def Laplacian_of_Gaussian(self):
        a = np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]])
        return a

    @staticmethod
    def flatten(x):
        return x.reshape(x.shape[0]*x.shape[1]*x.shape[2], -1)

    def convolve2D(self, img, num_filters, Kernel_size=None, padding=None, stride=None, activation=None):

        self.filters = []

        kernel_dim = Kernel_size[0]

        for i in range(num_filters):
            kernel = np.random.randn(kernel_dim, kernel_dim) / 9
            self.filters.append(kernel)

        k_l = Kernel_size[0]
        k_h = Kernel_size[1]

        if padding == 'SAME':

            s = 1

            if stride == None:
                s = 1

            if s != 1:
                raise Exception("Please change the stride to 1 with SAME padding")

            # Zero Padding
            pad = (k_l - 1) // 2
            total_pad = 2 * pad

            dim = int(((img.shape[0] - k_l + 2 * pad) // s) + 1)
            padded_image = np.zeros((dim + 2*pad, dim + 2*pad))

            padded_image[pad:-pad, pad:-pad] = img

            output_image = np.zeros((img.shape[0], img.shape[1], num_filters))

            a = 0
            b = 0
            for k in range(num_filters):
                filter = self.filters[k]
                filter = np.flipud(np.fliplr(filter))
                for i in range(padded_image.shape[0] - total_pad):
                    for j in range(padded_image.shape[1] - total_pad):
                        output_image[i, j, k] = np.multiply(filter, padded_image[i+a: i+a + k_l, j+b: j+b + k_h]).sum()
                        b = b + s - 1
                    b = 0
                    a = a + s - 1
                a = 0
                b = 0

        else:
            p = 0

            if stride == None:
                s = 1
            else:
                s = stride

            dim = int(((img.shape[0] - k_l + 2 * p) // s) + 1)
            output_image = np.zeros((dim, dim, num_filters))

            a = 0
            b = 0
            for k in range(num_filters):
                filter = self.filters[k]
                filter = np.flipud(np.fliplr(filter))
                for i in range(output_image.shape[0]):
                    for j in range(output_image.shape[1]):
                        output_image[i, j, k] = np.multiply(filter, img[i+a: i+a + k_l, j+b: j+b + k_h]).sum()
                        b = b + s - 1
                    b = 0
                    a = a + s - 1
                a = 0
                b = 0

        if activation == 'relu':
            output_image = relu.activation(output_image)

        return output_image

    def Maxpooling2D(self, img, pool_size=None, stride=None):

        if stride == None:
            s = 1
        else:
            s = stride

        pad = 0
        pool_dim = pool_size[0]

        output_dim = int(((img.shape[0] - pool_dim + 2 * pad) // s) + 1)
        output_image = np.zeros((output_dim, output_dim, img.shape[2]))

        a = 0
        b = 0
        for k in range(img.shape[2]):
            image = img[:, :, k]
            for i in range(output_image.shape[0]):
                for j in range(output_image.shape[1]):
                    output_image[i, j, k] = image[i + a: i + a + pool_dim, j + b: j + b + pool_dim].max()
                    b = b + s - 1
                b = 0
                a = a + s - 1
            a = 0
            b = 0
        return output_image

    def convolve3D(self, img, num_filters, Kernel_size=None, padding=None, stride=None, activation=None):

        self.filters = []

        kernel_dim = Kernel_size[0]

        for i in range(num_filters):
            kernel = np.random.randn(3, kernel_dim, kernel_dim) / 9
            self.filters.append(kernel)

        if Kernel_size == None:
            kernel_dim = 3
        else:
            kernel_dim = Kernel_size[0]


        k_l = Kernel_size[1]
        k_h = Kernel_size[2]

        if padding == 'SAME':

            s = 1

            if stride == None:
                s = 1

            if s != 1:
                raise Exception("Please change the stride to 1 with SAME padding")

            pad = (k_l - 1) // 2
            total_pad = 2 * pad

            dim = int(((img.shape[0] - k_l + 2 * pad) // s) + 1)

            padded_image = np.zeros((dim + 2*pad, dim + 2*pad, 3))
            padded_image[pad:-pad, pad:-pad, :] = img

            padded_image = padded_image.reshape((dim + 2*pad, dim + 2*pad, 3))

            l, h, d = img.shape
            output_image = np.zeros((l, h, num_filters))

            a = 0
            b = 0
            for k in range(num_filters):
                filter = self.filters[k]
                filter = np.flipud(np.fliplr(filter))
                for i in range(padded_image.shape[0] - total_pad):
                    for j in range(padded_image.shape[1] - total_pad):
                        output_image[i, j, k] = np.multiply(filter, padded_image[i+a: i+a + k_l, j+b: j+b + k_h, :]).sum()
                        b = b + s - 1
                    b = 0
                    a = a + s - 1
                a = 0
                b = 0

        else:
            pad = 0

            if stride == None:
                s = 1
            else:
                s = stride

            dim = int(((img.shape[0] - k_l + 2 * pad) // s) + 1)

            output_image = np.zeros((dim, dim, num_filters))

            img = img.reshape((3, img.shape[0], img.shape[1]))

            a = 0
            b = 0
            for k in range(num_filters):
                filter = self.filters[k]
                filter = np.flipud(np.fliplr(filter))
                for i in range(output_image.shape[0]):
                    for j in range(output_image.shape[1]):
                        output_image[i, j, k] = np.multiply(filter, img[i+a: i+a + k_l, j+b: j+b + k_h, :]).sum()
                        b = b + s - 1
                    b = 0
                    a = a + s - 1
                a = 0
                b = 0

        if activation == 'relu':
            output_image = relu.activation(output_image)

        return output_image

    def Maxpooling3D(self, img, pool_size=None, stride=None):

        if stride == None:
            s = 1
        else:
            s = stride

        pad = 0
        pool_dim = pool_size[0]

        output_dim = int(((img.shape[0] - pool_dim + 2 * pad) // s) + 1)
        output_image = np.zeros((output_dim, output_dim, img.shape[2]))

        a = 0
        b = 0
        for k in range(img.shape[2]):
            image = img[:, :, k]
            for i in range(output_image.shape[0]):
                for j in range(output_image.shape[1]):
                    output_image[i, j, k] = image[i + a: i + a + pool_dim, j + b: j + b + pool_dim].max()
                    b = b + s - 1
                b = 0
                a = a + s - 1
            a = 0
            b = 0

        return output_image

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    image = X_train[0]
    image = image / 255

    plt.imshow(image, cmap='Greys')
    plt.show()

    model = CNN()

    print(X_train.shape)
    print(image.shape)

    filterned_Image_1 = model.convolve2D(image, num_filters=2, Kernel_size=(3, 3), padding='SAME', stride=1, activation='relu')

    print(filterned_Image_1.shape)
    plt.imshow(filterned_Image_1[:, :, 0], cmap='Greys')
    plt.show()

    plt.imshow(filterned_Image_1[:, :, 1], cmap='Greys')
    plt.show()

    filterned_Image_2 = model.Maxpooling2D(filterned_Image_1, pool_size=(2, 2), stride=2)

    print(filterned_Image_2.shape)
    plt.imshow(filterned_Image_2[:, :, 0], cmap='Greys')
    plt.show()

    plt.imshow(filterned_Image_2[:, :, 1], cmap='Greys')
    plt.show()