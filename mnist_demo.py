########################################################################
#
# File:   mnist_demo.py
# Author: Matt Zucker
# Date:   April 2021
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# Download, manipulate, and visualize the MNIST dataset.

import os, sys, struct, shutil, urllib.request, zipfile
import datetime
import cv2
import numpy as np

MNIST_IMAGE_SIZE = 28
MNIST_DIMS = 784

MNIST_NUM_CLASSES = 10

WINDOW_NAME = 'MNIST Demo'

######################################################################
# Read a single 4-byte integer from a data file

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Load MNIST data from original file format

def parse_mnist(labels_file, images_file):

    labels = open(labels_file, 'rb')
    images = open(images_file, 'rb')

    lmagic = read_int(labels)
    assert lmagic == 2049

    lcount = read_int(labels)

    imagic = read_int(images)
    assert imagic == 2051

    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)

    assert rows == cols
    assert rows == MNIST_IMAGE_SIZE

    assert icount == lcount 

    l = np.fromfile(labels, dtype='uint8')
    i = np.fromfile(images, dtype='uint8')

    i = i.reshape((icount,rows,cols))

    return l, i

######################################################################
# Download and parse MNIST data

def get_mnist_data():

    filenames = [
        'train-labels-idx1-ubyte', 
        'train-images-idx3-ubyte', 
        't10k-images-idx3-ubyte', 
        't10k-labels-idx1-ubyte'
    ]

    if not all([os.path.exists(name) for name in filenames]):

        downloaded = False

        if not os.path.exists('mnist.zip'):
            print('downloading mnist.zip...')
            url = 'http://mzucker.github.io/swarthmore/mnist.zip'
            req = urllib.request.Request(url)
            f = urllib.request.urlopen(req)
            with open('mnist.zip', 'wb') as ostr:
                shutil.copyfileobj(f, ostr)
            print('done\n')
            downloaded = True

        z = zipfile.ZipFile('mnist.zip', 'r')

        names = z.namelist()

        assert set(names) == set(filenames)

        print('extracting mnist.zip...')

        for name in names:
            print(' ', name)
            with z.open(name) as f:
                with open(name, 'wb') as ostr:
                    shutil.copyfileobj(f, ostr)

        if downloaded:
            os.unlink('mnist.zip')

        print('done\n')

    print('loading MNIST data...')

    train_labels, train_images = parse_mnist('train-labels-idx1-ubyte',
                                             'train-images-idx3-ubyte')

    test_labels, test_images = parse_mnist('t10k-labels-idx1-ubyte',
                                           't10k-images-idx3-ubyte')

    print('done\n')

    return train_labels, train_images, test_labels, test_images

######################################################################

def one_hot_encoded(labels, num_classes):

    n = len(labels)

    output = np.zeros((n, num_classes), dtype=np.float32)

    output[np.arange(n), labels] = 1.0

    return output

######################################################################

def main():

    train_labels, train_images, test_labels, test_images = get_mnist_data()

    train_images = train_images.astype(np.float32) / 255.0 # map to [0, 1] intensity
    train_images = train_images.reshape(-1, MNIST_DIMS) # make row vectors 

    train_label_vecs = one_hot_encoded(train_labels, MNIST_NUM_CLASSES)

    test_images = test_images.astype(np.float32) / 255.0 # map to [0, 1] intensity
    test_images = test_images.reshape(-1, MNIST_DIMS) # make row vectors

    test_label_vecs = one_hot_encoded(test_labels, MNIST_NUM_CLASSES)

    print('train_images:', train_images.shape, train_images.dtype)
    print('train_labels:', train_labels.shape, train_labels.dtype)
    print('train_label_vecs:', train_label_vecs.shape, train_label_vecs.dtype)
    print()

    print('first 5 train_labels:', train_labels[:5])
    print()

    print('first 5 train_label_vecs:')
    print(train_label_vecs[:5])
    print()
    
    print('click in the window and press any key to quit')
    
    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(WINDOW_NAME, wflags)
    cv2.moveWindow(WINDOW_NAME, 50, 50)

    for img_as_vec in train_images:

        img = img_as_vec.reshape(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        img_inverted = 1 - img
        img_uint8 = np.clip(img_inverted*255, 0, 255).astype(np.uint8)
        
        display = cv2.resize(img_uint8, (0, 0), 
                             fx=4, fy=4, 
                             interpolation=cv2.INTER_NEAREST)

        cv2.imshow(WINDOW_NAME, display)

        k = cv2.waitKey(100)

        if k >= 0:
            break

if __name__ == '__main__':
    main()
