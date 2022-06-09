class InitWithDict(object):
    """
    Base class which can be initialized by reading properties from dict
    """

    def __init__(self, init=None):
        """
        :param init:
        """

        if init:
            for key, value in init.iteritems():
                setattr(self, key, value)