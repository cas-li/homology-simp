import numpy
import numpy as np
import numpy.linalg

from itertools import combinations


def isSmallSimpInBigSimp(small, big):
    last_index = 0
    for i in range(len(small)):
        if small[i] in big[last_index:]:
            last_index = big.index(small[i])
        else:
            return False
    return True


def print_powerset(string):
    for i in range(0, len(string) + 1):
        for element in combinations(string, i):
            print(''.join(element))


def simplexDifference(small, big):
    for i in range(len(big)):
        if big[i] not in small:
            return big[i]


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

# ask: input biggest simps (without vertex set)


big_simps = ["01234"]
temp_decompose = []
decomposed_simps = []

# getting the smaller simps from big simps
for i in big_simps:
    for vertex in i:
        temp_decompose.append(vertex)
    for j in range(1, len(temp_decompose) + 1):
        for element in combinations(temp_decompose, j):
            subsimp = (''.join(element))
            if subsimp not in decomposed_simps:
                decomposed_simps.append(subsimp)
    temp_decompose.clear()

# simps = ["0", "1", "2", "3", "01", "02", "03", "12",
#          "13", "23", "012", "013", "023", "123", "0123"]


# simps = ["0", "1", "2", "3", "01", "02", "03", "12",
#          "13", "23", "012", "013", "023", "123"]

simps = decomposed_simps

grouped_simps = []
temp_list = []

biggest_simp_size = 0

# finding the size of the biggest simp
for i in simps:
    if (len(i) > biggest_simp_size):
        biggest_simp_size = len(i)

# grouping the simps by size
for size in range(1, biggest_simp_size + 1):
    for i in simps:
        if (len(i) == size):
            temp_list.append(i)
    grouped_simps.append(temp_list.copy())
    temp_list.clear()

print("these are the grouped simps", grouped_simps)

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

# generate n-1 boundaries: iterate through i-1 size simplices, iterate through the i simps: check the i-1 length strings in the i-simp for match (only 2 things to check)
for size in range(1, len(grouped_simps)):
    #print("size", size)
    for small_simp_index in range(len(grouped_simps[size - 1])):
        #print("grouped_simps[size - 1]", grouped_simps[size - 1])
        small_simp = grouped_simps[size - 1][small_simp_index]
        for big_simp_index in range(len(grouped_simps[size])):
            big_simp = grouped_simps[size][big_simp_index]
            # checking if small simp in big simp
            if isSmallSimpInBigSimp(small_simp, big_simp):
                position = grouped_simps[size][big_simp_index].index(
                    simplexDifference(small_simp, big_simp))
                #print("position", position)
                # decide what number to put
                if (position % 2 == 0):
                    temp_row.append(1)
                else:
                    temp_row.append(-1)
            else:
                temp_row.append(0)
        # print(temp_row)
        temp_boundary.append(temp_row.copy())
        # print(temp_boundary)
        temp_row.clear()
    boundaries.append(temp_boundary.copy())
    temp_boundary.clear()

# add last boundary
for i in range(len(grouped_simps[len(grouped_simps) - 1])):
    temp_boundary.append([0])

boundaries.append(temp_boundary.copy())
temp_row.clear()
temp_boundary.clear()

print("these are the boundaries", boundaries)

# calculate and print all homologies possible
for i in range(len(boundaries) - 1):
    print(i, "homology:",
          f"{bettiNumber(numpy.array(boundaries[i]), numpy.array(boundaries[i+1]))}")
