""" some algorithms for Divide and Conquer """
from typing import List, TypeVar
from utility import get_next_power_of_two_greater_than, is_power_of_2
''' 
Maximum Subarray problem 
'''


def find_max_cross_mid(a: list[float], low: int, mid: int, high: int) -> tuple[int, int, float]:
    max_left_sum = -float('inf')
    current_sum, left_flag = 0, 0
    for i in range(mid, low - 1, -1):
        current_sum += a[i]
        if current_sum > max_left_sum:
            max_left_sum = current_sum
            left_flag = i

    max_right_sum = -float('inf')
    current_sum, right_flag = 0, 0
    for j in range(mid + 1, high + 1):
        current_sum += a[j]
        if current_sum > max_right_sum:
            max_right_sum = current_sum
            right_flag = j

    return left_flag, right_flag, (max_left_sum + max_right_sum)


def find_max_subarray(a: list[float], low: int, high: int) -> tuple[int, int, float]:
    if low == high:
        return low, high, a[low]
    else:
        mid = (low + high) // 2
        left_low, left_high, max_left_sum = find_max_subarray(a, low, mid)
        right_low, right_high, max_right_sum = find_max_subarray(a, mid + 1, high)
        cross_low, cross_high, max_cross_mid = find_max_cross_mid(a, low, mid, high)

    max_subarray = max(max_left_sum, max_right_sum, max_cross_mid)
    if max_subarray == max_left_sum:
        return left_low, left_high, max_left_sum
    elif max_subarray == max_right_sum:
        return right_low, right_high, max_right_sum
    else:
        return cross_low, cross_high, max_cross_mid


test_arr = [1, 2, 3, 4, 5, 6, -7, 8, -9]


def find_kth_smallest(a: list[int], b: list[int], k: int,
                      a_high: int, b_high: int, a_low: int = 0, b_low: int = 0) -> int:
    """ Finds the kth smallest element among elements of two a and b lists
    which contains unique elements in both of them and are sorted ascending """

    if k == 1:
        return min((a[a_low], b[b_low]))
    elif a_low >= a_high:
        return b[b_low + k - 1]
    elif b_low >= b_high:
        return a[a_low + k - 1]

    else:
        half_k = k // 2
        target_a = min((a_high, (half_k + a_low - 1)))
        target_b = min((b_high, (half_k + b_low - 1)))

        if a[target_a] < b[target_b]:
            return find_kth_smallest(a, b, (k - half_k), a_high, b_high, (target_a + 1), b_low)
        else:
            return find_kth_smallest(a, b, (k - half_k), a_high, b_high, a_low, (target_b + 1))


test_a = [10, 12, 19, 28, 35, 90, 100, 108]
test_b = [7, 15, 16, 29, 38, 77, 80, 99, 120, 199]

numeric = TypeVar('numeric', *(int, float, complex, ))


class SizeError(Exception):

    def __init__(self, *args):
        super().__init__(*args)


class Matrix:

    matrix: List[List[numeric]]

    def __init__(self, size_of_rows: int = 0, size_of_columns: int = 0, matrix: List[List[numeric]] = None):
        if matrix is None:
            self.matrix = [[0] * size_of_columns for _ in range(size_of_rows)]
        else:
            self.matrix = matrix

    @property
    def num_rows(self) -> int:
        return len(self.matrix)

    @property
    def num_columns(self) -> int:
        return len(self.matrix[0])

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
            raise SizeError(f'Matrix must have the same number of rows and columns \n '
                            f'but got {self.num_rows} rows and {self.num_columns} columns'
                            f'against {other.num_rows} rows and {other.num_columns} columns.')

        new_matrix = Matrix(matrix=[])
        for row_self, row_other in zip(self.matrix, other.matrix):
            new_row: list = []
            for self_element, other_element in zip(row_self, row_other):
                new_row.append(self_element + other_element)

            new_matrix.matrix.append(new_row)

        return new_matrix

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        add with negation of other matrix
        """
        return self + other.negate()
        pass

    def negate(self) -> 'Matrix':
        new_matrix = Matrix(0, 0)
        for row in self.matrix:
            new_row = []
            for i in range(len(row)):
                new_row.append(-(row[i]))

            new_matrix.matrix.append(new_row)
        return new_matrix

    @staticmethod
    def negate_element(element: numeric) -> numeric:
        if isinstance(element, (int, float, complex)):
            return -element
        elif isinstance(element, bool):
            return not element

    def __mul__(self, other: 'Matrix') -> 'Matrix':

        if not self.are_adjustable_for_multiplication_with_other_matrix(other):
            raise SizeError('multiplication not performable because size of other matrix is not adjustable')

        max_size_between_both_matrix = max(self.num_rows, self.num_columns, other.num_rows, other.num_columns)
        if not (self.is_square() and self.is_size_power_of_2()):
            self._fill_matrix_with_zero_to_make_size_power_of_two_and_make_square(max_size_between_both_matrix)
        if not (other.is_square() and other.is_size_power_of_2()):
            other._fill_matrix_with_zero_to_make_size_power_of_two_and_make_square(max_size_between_both_matrix)

        return Matrix(matrix=Matrix._strassen_multiply_recursive(self, other))

    def are_adjustable_for_multiplication_with_other_matrix(self, other: 'Matrix') -> bool:
        return self.num_columns == other.num_rows

    def is_square(self) -> bool:
        return self.num_rows == self.num_columns

    def is_size_power_of_2(self) -> bool:
        return is_power_of_2(self.num_rows)
        pass

    def _fill_matrix_with_zero_to_make_size_power_of_two_and_make_square(self, max_size: int):
        power_of_two_size = get_next_power_of_two_greater_than(max_size)
        number_of_zero_elements_to_fill_existing_rows = (power_of_two_size - self.num_columns)
        for row in self.matrix:
            row += [0] * number_of_zero_elements_to_fill_existing_rows

        for _ in range(power_of_two_size - self.num_rows):
            self.matrix.append([0] * power_of_two_size)

    @staticmethod
    def _strassen_multiply_recursive(a: 'Matrix', b: 'Matrix') -> list[list[numeric]]:
        n = a.num_rows
        result: list[list[numeric]] = [[0] * n for _ in range(n)]
        if n == 1:
            result.append([a.matrix[0][0] * b.matrix[0][0]])
            return result
        else:
            # Partition a, b

            a_parts = []
            b_parts = []
            for i in range(1, 5):
                a_parts.append(Matrix(matrix=Matrix.__get_partition(a.matrix, part=i)))
                b_parts.append(Matrix(matrix=Matrix.__get_partition(b.matrix, part=i)))

            m1 = Matrix(
                matrix=Matrix._strassen_multiply_recursive((a_parts[0] + a_parts[3]), (b_parts[0] + b_parts[3]))
            )
            m2 = Matrix(
                matrix=Matrix._strassen_multiply_recursive((a_parts[2] + b_parts[3]), b_parts[0])
            )
            m3 = Matrix(
                matrix=Matrix._strassen_multiply_recursive(a_parts[0], (b_parts[1] - b_parts[3]))
            )
            m4 = Matrix(
                matrix=Matrix._strassen_multiply_recursive(a_parts[3], (b_parts[2] - b_parts[0]))
            )
            m5 = Matrix(
                matrix=Matrix._strassen_multiply_recursive((a_parts[0] + a_parts[1]), b_parts[3])
            )
            m6 = Matrix(
                matrix=Matrix._strassen_multiply_recursive((a_parts[2] - a_parts[0]), (b_parts[0] + b_parts[1]))
            )
            m7 = Matrix(
                matrix=Matrix._strassen_multiply_recursive((a_parts[1] - a_parts[3]), (b_parts[2] + b_parts[3]))
            )

            c11 = m1 + m4 - m5 + m7
            c12 = m3 + m5
            c21 = m2 + m4
            c22 = m1 + m3 - m2 + m6

            half_n = n // 2
            for i in range(half_n):
                for j in range(half_n):
                    result[i][j] = c11.matrix[i][j]
                    result[i][j + half_n] = c12.matrix[i][j]
                    result[i + half_n][j] = c21.matrix[i][j]
                    result[i + half_n][j + half_n] = c22.matrix[i][j]
            #
            # for row_c11, row_c12 in zip(c11.matrix, c12.matrix):
            #     result.append(row_c11 + row_c12)
            # for row_c21, row_c22 in zip(c21.matrix, c22.matrix):
            #     result.append(row_c21 + row_c22)

            return result

    @staticmethod
    def __get_partition(matrix_array: list[list[numeric]], part: int = 1) -> list[list[numeric]]:
        """
        to give a quarter partition of the square matrix array
        :param matrix_array: matrix to partition
        :param part: Part number of a square matrix divided into four parts like: [[part1, part2], [part3, part4]].
                    Should be between 1 and 4 if not a ValueError will be raised.
        :return: a quarter sub matrix .
        """
        if not (1 <= part <= 4):
            raise ValueError(f'part is out of bound should be between 1 to 4, but got {part}')
        n = len(matrix_array)
        half_n = n // 2
        start_rows, start_cols = 0, 0
        end_rows, end_cols = half_n, half_n
        partition: list[list[numeric]] = []

        match part:
            case 1:
                start_rows = 0
                end_rows = half_n
                start_cols = 0
                end_cols = half_n

            case 2:
                start_rows = 0
                end_rows = half_n
                start_cols = half_n
                end_cols = n

            case 3:
                start_rows = half_n
                end_rows = n
                start_cols = 0
                end_cols = half_n

            case 4:
                start_rows = half_n
                end_rows = n
                start_cols = half_n
                end_cols = n

        for i in range(start_rows, end_rows):
            partition.append([matrix_array[i][j] for j in range(start_cols, end_cols)])

        return partition

    @staticmethod
    def get_merge_partition(part_a: list[list[numeric]], part_b: list[list[numeric]]):
        for row_a, row_b in zip(part_a, part_b):
            yield row_a + row_b


if __name__ == "__main__":
    # test of find_kth_smallest()
    #
    # smallest_kth_number_between_two_arrays = find_kth_smallest(test_a, test_b, 7,
    #                                                            (len(test_a) - 1),
    #                                                            (len(test_b) - 1)
    #                                                            )
    #
    # print(f' kth smallest element between {sorted(test_a + test_b) = } \n ',
    #       f'{smallest_kth_number_between_two_arrays = }')
    #
    #

    matrix1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    matrix2 = [[1,], [1,], [1,], [1,]]
    m1 = Matrix(matrix=matrix1)
    m2 = Matrix(matrix=matrix2)

    print(*(m1 * m2).matrix, sep='\n')
