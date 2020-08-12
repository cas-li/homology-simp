import numpy
import numpy as np
import numpy.linalg

# problem: keep getting 0 for homology. prob bc we need to not do it in F2 sad


def isSmallSimpInBigSimp(small, big):
    last_index = 0
    for i in range(len(small)):
        if small[i] in big[last_index:]:
            last_index = big.index(small[i])
        else:
            return False
    return True


def rowSwap(A, i, j):
    temp = numpy.copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp


def colSwap(A, i, j):
    temp = numpy.copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp


def scaleCol(A, i, c):
    A[:, i] *= int(c) * numpy.ones(A.shape[0], dtype=np.int64)


def scaleRow(A, i, c):
    A[i, :] = np.array(A[i, :], dtype=numpy.float64) * c * \
        numpy.ones(A.shape[1], dtype=numpy.float64)


def colCombine(A, addTo, scaleCol, scaleAmt):
    A[:, addTo] += scaleAmt * A[:, scaleCol]


def rowCombine(A, addTo, scaleRow, scaleAmt):
    A[addTo, :] += scaleAmt * A[scaleRow, :]


def simultaneousReduce(A, B):
    if A.shape[1] != B.shape[0]:
        raise Exception("Matrices have the wrong shape.")

    numRows, numCols = A.shape

    i, j = 0, 0
    while True:
        if i >= numRows or j >= numCols:
            break

        if A[i, j] == 0:
            nonzeroCol = j
            while nonzeroCol < numCols and A[i, nonzeroCol] == 0:
                nonzeroCol += 1

            if nonzeroCol == numCols:
                i += 1
                continue

            colSwap(A, j, nonzeroCol)
            rowSwap(B, j, nonzeroCol)

        pivot = A[i, j]
        scaleCol(A, j, 1.0 / pivot)
        scaleRow(B, j, 1.0 / pivot)

        for otherCol in range(0, numCols):
            if otherCol == j:
                continue
            if A[i, otherCol] != 0:
                scaleAmt = -A[i, otherCol]
                colCombine(A, otherCol, j, scaleAmt)
                rowCombine(B, j, otherCol, -scaleAmt)

        i += 1
        j += 1

    return A, B


def finishRowReducing(B):
    numRows, numCols = B.shape

    i, j = 0, 0
    while True:
        if i >= numRows or j >= numCols:
            break

        if B[i, j] == 0:
            nonzeroRow = i
            while nonzeroRow < numRows and B[nonzeroRow, j] == 0:
                nonzeroRow += 1

            if nonzeroRow == numRows:
                j += 1
                continue

            rowSwap(B, i, nonzeroRow)

        pivot = B[i, j]
        scaleRow(B, i, 1.0 / pivot)

        for otherRow in range(0, numRows):
            if otherRow == i:
                continue
            if B[otherRow, j] != 0:
                scaleAmt = -B[otherRow, j]
                rowCombine(B, otherRow, i, scaleAmt)

        i += 1
        j += 1

    return B


def numPivotCols(A):
    z = numpy.zeros(A.shape[0])
    return [numpy.all(A[:, j] == z) for j in range(A.shape[1])].count(False)


def numPivotRows(A):
    z = numpy.zeros(A.shape[1])
    return [numpy.all(A[i, :] == z) for i in range(A.shape[0])].count(False)


def bettiNumber(d_k, d_kplus1):
    A, B = numpy.copy(d_k), numpy.copy(d_kplus1)
    simultaneousReduce(A, B)
    finishRowReducing(B)

    dimKChains = A.shape[1]
    # print(dimKChains)
    kernelDim = dimKChains - numPivotCols(A)
    # print(kernelDim)
    imageDim = numPivotRows(B)
    # print(imageDim)

    return kernelDim - imageDim


# bd0 = numpy.array([[0, 0, 0, 0]])
# bd1 = numpy.array([[-1, -1, -1, 0, 0, 0], [1, 0, 0, -1, -1, 0],
#                    [0, 1, 0, 1, 0, -1], [0, 0, 1, 0, 1, 1]])

# bd2 = numpy.array([[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 0], [1, 0, 0, 1], [0, 1, 0, -1],
#                    [0, 0, 1, 1]])
# bd3 = numpy.array([[-1], [1], [-1], [1]])

# print("3-simplex")
# print(f"0th homology: {bettiNumber(bd0, bd1)}")
# print(f"1st homology: {bettiNumber(bd1, bd2)}")
# print(f"2nd homology: {bettiNumber(bd2, bd3)}")

# bd0_1 = numpy.array([[0, 0, 0, 0]])
# bd1_1 = numpy.array([[-1, -1, -1, 0, 0, 0], [1, 0, 0, -1, -1, 0],
#                      [0, 1, 0, 1, 0, -1], [0, 0, 1, 0, 1, 1]])

# bd2_1 = numpy.array([[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 0], [1, 0, 0, 1], [0, 1, 0, -1],
#                      [0, 0, 1, 1]])
# bd3_1 = numpy.array([[0], [0], [0], [0]])

# print("boundary 2-sphere")
# print(f"0th homology: {bettiNumber(bd0_1, bd1_1)}")
# print(f"1st homology: {bettiNumber(bd1_1, bd2_1)}")
# print(f"2nd homology: {bettiNumber(bd2_1, bd3_1)}")

# bd0_2 = numpy.array([[0, 0, 0, 0]])
# bd1_2 = numpy.array([[-1, -1, -1, 1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, -1, -1, -1, 0, 1, 0, 0, 1, 0],
#                      [0, 1, 0, 0, 1, 0, -1, -1, -1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, -1, -1, -1]])

# bd2_2 = numpy.array([[1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1],
#                      [0, 0, 0, 0, -1, -1, 0, 0],
#                      [0, 0, 1, 0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0, 0, -1, -1],
#                      [0, 0, 0, 0, 0, 1, 1, 0],
#                      [1, 0, 0, 1, 0, 0, 0, 0],
#                      [-1, -1, 0, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 1, 0, 0, 0],
#                      [0, 0, -1, -1, 0, 0, 0, 0],
#                      [0, 1, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 0, 0, 1, 0]
#                      ])
# bd3_2 = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0]])

# print("torus")
# print(f"0th homology: {bettiNumber(bd0_2, bd1_2)}")
# print(f"1st homology: {bettiNumber(bd1_2, bd2_2)}")
# print(f"2nd homology: {bettiNumber(bd2_2, bd3_2)}")


# ask: input biggest simplices (without vertex set)
# powerset on big simps?? https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
# if latter:
# build vertex set: iterate thru simplices, record each unique character into list
# decompose each big simp into little simps; if a little simp is already in the list, then don't add it to the list

# make a list of size n: for loop that iterates through 1 through n, for each number, find the number of simps that are size i; O(N)
# make n matrices, where n is the size of the biggest simplex

# simp is currently 3-simplex
simps = ["0", "1", "2", "3", "01", "02", "03", "12",
         "13", "23", "012", "013", "023", "123", "0123"]
grouped_simps = []
temp_list = []

biggest_simp_size = 0

# finding the size of the biggest simp
for i in simps:
    if (len(i) > biggest_simp_size):
        biggest_simp_size = len(i)

print(biggest_simp_size)

# grouping the simps by size
for size in range(1, biggest_simp_size + 1):
    for i in simps:
        if (len(i) == size):
            temp_list.append(i)
    grouped_simps.append(temp_list.copy())
    temp_list.clear()


print("this is the grouped simps")
print(grouped_simps)


boundaries = []
temp_boundary = []
temp_row = []

# add bd0
for i in range(len(grouped_simps[0])):
    temp_row.append(0)

temp_boundary.append(temp_row.copy())
boundaries.append(temp_boundary.copy())
temp_row.clear()
temp_boundary.clear()

print(boundaries)

# make bdi, iterate through i-1 size simplices, iterate through the i simps: check the i-1 length strings in the i-simp for match (only 2 things to check): if yes, put 1; else put 0; O(N^2)
for size in range(1, len(grouped_simps)):
    #print("size", size)
    for small_simp in range(len(grouped_simps[size - 1])):
        #print("grouped_simps[size - 1]", grouped_simps[size - 1])
        for big_simp in range(len(grouped_simps[size])):
            if isSmallSimpInBigSimp(grouped_simps[size - 1][small_simp], grouped_simps[size][big_simp]):
                temp_row.append(1)
            else:
                temp_row.append(0)
        # print(temp_row)
        temp_boundary.append(temp_row.copy())
        # print(temp_boundary)
        temp_row.clear()
    boundaries.append(temp_boundary.copy())
    temp_boundary.clear()


print(boundaries)

# print(boundaries[0])
# print(boundaries[1])

print("sphere")
print(
    f"0th homology: {bettiNumber(numpy.array(boundaries[0]), numpy.array(boundaries[1]))}")
print(
    f"1st homology: {bettiNumber(numpy.array(boundaries[1]), numpy.array(boundaries[2]))}")
print(
    f"2nd homology: {bettiNumber(numpy.array(boundaries[2]), numpy.array(boundaries[3]))}")
