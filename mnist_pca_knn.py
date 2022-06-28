########################################################################
#
# File:   mnist_pca_knn.py
# Author: Matt Zucker
# Date:   March 2021
#
# Written for ENGR 27 - Computer Vision
#
# UPDATED BY PATRICK KYAW: April 23, 2021

########################################################################
#
# Shows how to do kNN classification (plus optional PCA) on MNIST
# dataset.

import os, sys, struct, shutil, urllib.request, zipfile
import cv2
from datetime import datetime
import numpy as np

MNIST_IMAGE_SIZE = 28
MNIST_DIMS = 784

TARGET_DISPLAY_SIZE = 340

WINDOW_NAME = 'MNIST PCA + kNN Demo'

BATCH_SIZE = 500


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
# For majority voting step of k nearest neighbors
# https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays

def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)
    return count

######################################################################
# Construct an object we can use for fast nearest neighbor queries.
# See https://github.com/mariusmuja/flann
# And https://docs.opencv.org/master/dc/de2/classcv_1_1FlannBasedMatcher.html

def get_knn_matcher():

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    return matcher

######################################################################
# Use the matcher object to match query vectors to training vectors.
#
# Parameters:
#
#   * query_vecs is p-by-n
#   * train_vecs is m-by-n
#   * train_labels is flat array of length m (optional)
#
# Returns:
#
#   * match_indices p-by-k indices of closest rows in train_vecs
#   * labels_pred flat array of length p (if train_labels is provided)

def match_knn(matcher, k, query_vecs, train_vecs, train_labels=None):

    knn_result = matcher.knnMatch(query_vecs, train_vecs, k)

    match_indices = np.full((len(query_vecs), k), -1, int)

    for i, item_matches in enumerate(knn_result):
        match_indices[i] = [ match.trainIdx for match in item_matches ]

    if train_labels is None:
        return match_indices

    match_labels = train_labels[match_indices]

    bcount = bincount2d(match_labels, bins=10)

    labels_pred = bcount.argmax(axis=1)

    return match_indices, labels_pred

######################################################################
# Load precomputed PCA mean and eigenvectors from an .npz file
# (or create the file if it doesn't exist)

def load_precomputed_pca(train_images, k):

    try:

        d = np.load('mnist_pca.npz')
        mean = d['mean']
        eigenvectors = d['eigenvectors']
        print('loaded precomputed PCA from mnist_pca.npz')

    except:

        print('precomputing PCA one time only for train_images...')

        ndim = train_images.shape[1]
        mean, eigenvectors = cv2.PCACompute(train_images,
                                            mean=None,
                                            maxComponents=train_images.shape[1])

        print('done\n')

        np.savez_compressed('mnist_pca.npz',
                            mean=mean,
                            eigenvectors=eigenvectors)


    eigenvectors = eigenvectors[:k]

    return mean, eigenvectors

######################################################################

def main():

    if len(sys.argv) != 3:
        print('usage: python mnist_pca_knn.py PCA_K KNN_K')
        print()
        print('note: set PCA_K to 0 to disable PCA')
        print()
        sys.exit(1)

    args = sys.argv[1:]

    assert len(args) == 2

    pca_k = int(args[0])
    assert pca_k >= 0 and pca_k <= MNIST_DIMS

    knn_k = int(args[1])
    assert knn_k > 0 and knn_k < 12

    train_labels, train_images, test_labels, test_images = get_mnist_data()

    train_images = train_images.astype(np.float32)
    train_images = train_images.reshape(-1, MNIST_DIMS) # make row vectors

    test_images = test_images.astype(np.float32)
    test_images = test_images.reshape(-1, MNIST_DIMS)

    begin_pca = datetime.now()
    if pca_k > 0:


        # note we could use cv2.PCACompute to do this but
        # instead we use a pre-computed eigen-decomposition
        # of the data if available
        mean, eigenvectors = load_precomputed_pca(train_images, pca_k)
        print('reducing dimensionality of training set...')

        train_vecs = cv2.PCAProject(train_images, mean, eigenvectors)

        print('reducing dimensionality of test set...')
        test_vecs = cv2.PCAProject(test_images, mean, eigenvectors)

        end_pca = datetime.now()

        print("PCA time taken: ", (end_pca - begin_pca).total_seconds())

    else:

        mean = None
        eigenvectors = None
        train_vecs = train_images
        test_vecs = test_images



    print('test_images:', test_images.shape)
    print('test_vecs:', test_vecs.shape)
    print()

    matcher = get_knn_matcher()

    num_test = len(test_images)

    total_errors = 0

    start = datetime.now()

    print(f'evaluating knn accuracy with k={knn_k}...')

    for start_idx in range(0, num_test, BATCH_SIZE):

        end_idx = min(start_idx + BATCH_SIZE, num_test)
        cur_batch_size = end_idx - start_idx

        idx, labels_pred = match_knn(matcher, knn_k,
                                     test_vecs[start_idx:end_idx],
                                     train_vecs, train_labels)

        labels_true = test_labels[start_idx:end_idx]

        total_errors += (labels_true != labels_pred).sum()
        error_rate = 100.0 * total_errors / end_idx

        print(f'{total_errors:4d} errors after {end_idx:5d} test examples (error rate={error_rate:.2f}%)')

    elapsed = (datetime.now() - start).total_seconds()
    print("Total testing time: ", (datetime.now() - start).total_seconds())

if __name__ == '__main__':
    main()
