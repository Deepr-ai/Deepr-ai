import libc.math as math
cdef class Activation:

    cpdef double tanh(self, double n):
        return math.tanh(n)

def add(a):
    return a+3