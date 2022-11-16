import numpy as np


class RingBuffer:
    """
    Simple RingBuffer object to hold values to average (useful for, e.g.: filtering D component in PID control)
    Note that the buffer object is a 2D numpy array, where each row corresponds to
    individual entries into the buffer
    Args:
        dim (int): Size of entries being added. This is, e.g.: the size of a state vector that is to be stored
        length (int): Size of the ring buffer
    """

    def __init__(self, dim, length):
        # Store input args
        self.dim = dim
        self.length = length

        # Variable so that initial average values are accurate
        self._size = 0

        # Save pointer to end of buffer
        self.ptr = self.length - 1

        # Construct ring buffer
        self.buf = np.zeros((length, dim))

    def push(self, value):
        """
        Pushes a new value into the buffer
        Args:
            value (int or float or array): Value(s) to push into the array (taken as a single new element)
        """
        # Increment pointer, then add value (also increment size if necessary)
        self.ptr = (self.ptr + 1) % self.length
        self.buf[self.ptr] = np.array(value)
        if self._size < self.length:
            self._size += 1

    def clear(self):
        """
        Clears buffer and reset pointer
        """
        self.buf = np.zeros((self.length, self.dim))
        self.ptr = self.length - 1
        self._size = 0

    @property
    def current(self):
        """
        Gets the most recent value pushed to the buffer
        Returns:
            float or np.array: Most recent value in buffer
        """
        return self.buf[self.ptr]

    @property
    def average(self):
        """
        Gets the average of components in buffer
        Returns:
            float or np.array: Averaged value of all elements in buffer
        """
        return np.mean(self.buf[: self._size], axis=0)