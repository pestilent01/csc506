'''This module measures time and space complexity of functions.'''
def measureTime(function, *args):
    '''This function measures the time taken by a function to execute.'''
    import time
    start = time.time()
    result = function(*args)
    end = time.time()
    duration = end - start
    return result, duration

def measureSpace(function, *args):
    '''This function measures the space taken by a function to execute.'''
    import tracemalloc
    tracemalloc.start()
    result = function(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current, peak

def getCompressionRatio(input_size, output_size):
    '''This function calculates the compression ratio.'''
    if input_size == 0:
        return 0
    return (output_size / input_size)