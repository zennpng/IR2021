# NOTE: READ THIS LINK
# https://stackoverflow.com/questions/42602584/how-to-use-multiprocessing-pool-in-an-imported-module

#########################################################
# import multiprocessing
  
# # square function
# def square(x):
#     return x * x

# if __name__ == '__main__':

  
#     # multiprocessing pool object
#     pool = multiprocessing.Pool()
  
#     # pool object with number of element
#     pool = multiprocessing.Pool(processes=4)
  
#     # input list
#     inputs = [0, 1, 2, 3, 4]
  
#     # map the function to the list and pass
#     # function and input list as arguments
#     outputs = pool.map(square, inputs)
  
#     # Print input list
#     print("Input: {}".format(inputs))
  
#     # Print output list
#     print("Output: {}".format(outputs))

####################################################
# example
import multiprocessing as mp
import numpy as np

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]

def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


if __name__ == '__main__':
    # Parallelizing using Pool.apply()

    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(4) # use 4 processors

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

    # Step 3: Don't forget to close
    pool.close()    

    print(results[:10])
    #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]