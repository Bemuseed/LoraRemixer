import numpy

def matrix_flatten(matrix):
    if matrix.shape == (1,):
        result = numpy.array([matrix[0]])
        shapes = [(1,)]
        offsets = numpy.array([0,1])
    else:
        shapes = [a.shape for a in matrix]
        offsets = [a.size for a in matrix]
        offsets = numpy.cumsum([0] + offsets)
        result = numpy.concatenate([a.flat for a in matrix])
    return result, shapes, offsets

def matrix_unflatten(flattened, shapes, offsets):
    restored = numpy.array([numpy.reshape(flattened[offsets[i]:offsets[i + 1]], shape) for i, shape in enumerate(shapes)])
    return restored

def model_flatten(matrix_list):
    result = numpy.array([], dtype=numpy.float16)
    shapes = []
    offsets = []
    limits = [0]

    counter = 0
    for a in matrix_list:
        counter += 1
        r, s, o = matrix_flatten(a)
        result = numpy.append(result, r)
        shapes.append(s)
        offsets.append(o)
        limits.append(len(result))
    return result, shapes, offsets, limits

def model_unflatten(flattened, shapes, offsets, limits):
    unflattened = []
    for i in range(len(limits) - 1):
        section = flattened[limits[i]:limits[i+1]]
        unflattened.append(matrix_unflatten(section, shapes[i], offsets[i]))
    return unflattened
