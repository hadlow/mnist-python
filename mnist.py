import numpy as np
import struct as st

def process_image_file(image_file):
    image_file.seek(0)
    magic = st.unpack('>4B', image_file.read(4))

    n_images = st.unpack('>I', image_file.read(4))[0]
    n_rows = st.unpack('>I', image_file.read(4))[0]
    n_columns = st.unpack('>I', image_file.read(4))[0]
    n_bytes = n_images * n_rows * n_columns

    images = np.zeros((n_images, n_rows * n_columns))
    images = np.asarray(st.unpack('>' + 'B' * n_bytes, image_file.read(n_bytes))).reshape((n_images, n_rows * n_columns))

    return images

def process_label_file(label_file):
    label_file.seek(0)
    magic = st.unpack('>4B', label_file.read(4))

    n_labels = st.unpack('>I', label_file.read(4))[0]

    labels = np.zeros((n_labels))
    labels = np.asarray(st.unpack('>' + 'B' * n_labels, label_file.read(n_labels)))

    return labels

def one_hot(number, size):
    onehot = np.zeros(size)
    onehot[number] = 1

    return onehot

def dataset():
    home = './dataset/'

    test_images = open(home + 't10k-images-idx3-ubyte', 'rb')
    test_labels = open(home + 't10k-labels-idx1-ubyte', 'rb')
    train_images = open(home + 'train-images-idx3-ubyte', 'rb')
    train_labels = open(home + 'train-labels-idx1-ubyte', 'rb')
    
    train_images = process_image_file(train_images)
    test_images = process_image_file(test_images)
    train_labels = process_label_file(train_labels)
    test_labels = process_label_file(test_labels)
    
    return ((train_images, test_images), (train_labels, test_labels))
