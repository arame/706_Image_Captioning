class NoneTransform(object):
    """ Does nothing to the image. To be used instead of None """

    def __call__(self, image):
        return image
