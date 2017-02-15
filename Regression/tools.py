import matplotlib as plt

def vizualize_image(vector):
    im = vector.reshape((256, 256), order='F')
    plt.imshow(im, cmap='gray')
    plt.show()