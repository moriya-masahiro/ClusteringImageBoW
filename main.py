# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import functools
import pathlib
import shutil

from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy


input_dir = '/path/to/images/'
output_dir = '/path/to/save/'

akaze = cv2.AKAZE_create()
images = tuple(pathlib.Path(input_dir).glob('*.jpg'))


@functools.lru_cache(maxsize=1024)
def read_image(path, size=(320, 240)):
    img = cv2.imread(str(path))
    if img.shape[0] > img.shape[1]:
        return cv2.resize(img, (size[1], size[1]*img.shape[0]//img.shape[1]))
    else:
        return cv2.resize(img, (size[0]*img.shape[1]//img.shape[0], size[0]))


@functools.lru_cache(maxsize=None)
def load_kps(path):
    return akaze.detectAndCompute(read_image(path), None)[1]


def detect_all(verbose=False):
    for i, path in enumerate(images):
        if verbose:
            print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/len(images), path))

        try:
            yield from load_kps(path)
        except TypeError as e:
            print(e)


def make_visual_words(verbose=False):
    features = numpy.array(tuple(detect_all(verbose=verbose)))
    return MiniBatchKMeans(n_clusters=128, verbose=verbose).fit(features).cluster_centers_


def make_hist(vws, path):
    hist = numpy.zeros(vws.shape[0])
    for kp in load_kps(path):
        hist[((vws - kp)**2).sum(axis=1).argmin()] += 1
    return hist


def find_nears(vws, hist, n=5, verbose=False):
    nears = []
    for i, path in enumerate(images):
        if verbose:
            print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/len(images), path))

        try:
            h = make_hist(vws, path)
        except TypeError:
            continue

        nears.append((((h - hist)**2).sum(), h, path))
        nears.sort(key=lambda x:x[0])
        nears = nears[:n]
    return nears


if __name__ == '__main__':
    vws = make_visual_words(True)

    path = images[0]
    img = read_image(path)
    hist = make_hist(vws, path)

    nears = find_nears(vws, hist, n=20, verbose=True)
    for x in nears:
        print('{0:.2f} - {2}'.format(*x))
        shutil.copy(str(x[2]), '{0}{1:.2f}.jpg'.format(output_dir, x[0]))