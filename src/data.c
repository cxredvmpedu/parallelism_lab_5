#define _XOPEN_SOURCE 800

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "data.h"

int64_t min(int64_t a, int64_t b) { return a < b ? a : b; }

// Інізіалізує стан для ГПВЧ
void state_init(unsigned short seed[3], int rank) {
    time_t t = time(nullptr);

    seed[0] = t + rank;
    seed[1] = t >> 16;
    seed[2] = t >> 32;
}

// Генерує псевдовипадкове число від `low` до `high`
int64_t random_in(unsigned short state[3], int64_t low, int64_t high) {
    return low + nrand48(state) % (high - low + 1);
}

static inline const int64_t*
matrix_at(const Matrix* matrix, size_t row, size_t col) {
    return &matrix->elems[row * matrix->col_n + col];
}

static inline int64_t* matrix_at_mut(Matrix* matrix, size_t row, size_t col) {
    return &matrix->elems[row * matrix->col_n + col];
}

void Matrix_init(Matrix* matrix, size_t row_n, size_t col_n) {
    matrix->elems = malloc(row_n * col_n * sizeof(int64_t));
    matrix->row_n = row_n;
    matrix->col_n = col_n;
}

void Matrix_destroy(Matrix* matrix) { free(matrix->elems); }

void Matrix_init_clone(Matrix* clone, const Matrix* matrix) {
    Matrix_init(clone, matrix->row_n, matrix->col_n);
    memcpy(
        clone->elems,
        matrix->elems,
        matrix->row_n * matrix->col_n * sizeof(int64_t)
    );
}

void Matrix_init_random(
    Matrix* matrix,
    size_t row_n,
    size_t col_n,
    unsigned short state[3],
    int64_t low,
    int64_t high
) {
    Matrix_init(matrix, row_n, col_n);
    for (size_t i = 0; i < row_n; i++) {
        for (size_t j = 0; j < col_n; j++) {
            *matrix_at_mut(matrix, i, j) = random_in(state, low, high);
        }
    }
}

void Matrix_init_slice_cols(
    Matrix* slice,
    const Matrix* matrix,
    size_t start_col,
    size_t col_n
) {
    Matrix_init(slice, matrix->row_n, col_n);

    for (size_t i = 0; i < slice->row_n; i++) {
        const int64_t* src_row =
            matrix->elems + (i * matrix->col_n) + start_col;
        int64_t* dst_row = slice->elems + (i * slice->col_n);
        memcpy(dst_row, src_row, col_n * sizeof(int64_t));
    }
}

void Matrix_init_product(
    Matrix* product,
    const Matrix* left,
    const Matrix* right
) {
    assert(left->col_n == right->row_n);

    Matrix_init(product, left->row_n, right->col_n);

    for (size_t i = 0; i < product->row_n; i++) {
        for (size_t j = 0; j < product->col_n; j++) {
            int64_t sum = 0;
            for (size_t k = 0; k < left->col_n; k++) {
                sum += *matrix_at(left, i, k) * *matrix_at(right, k, j);
            }
            *matrix_at_mut(product, i, j) = sum;
        }
    }
}

void Matrix_fmt(char** str, const Matrix* matrix) {
    // Number: 1 + floor(log10(2^63-1)) = 19 chars
    // Space / newline: 1 char
    // \0: 1 char
    size_t size = (matrix->row_n * matrix->col_n * 19) + 1;
    *str = malloc(size);
    (*str)[0] = '\0';

    char* ptr = *str;
    for (size_t i = 0; i < matrix->row_n; i++) {
        for (size_t j = 0; j < matrix->col_n; j++) {
            ptr += sprintf(ptr, "%li", *matrix_at(matrix, i, j));

            if (j != matrix->col_n - 1) {
                ptr += sprintf(ptr, " ");
            }
        }

        if (i != matrix->row_n - 1) {
            ptr += sprintf(ptr, "\n");
        }
    }
}

const int64_t* vector_at(const Vector* vector, size_t i) {
    return &vector->elems[i];
}

int64_t* vector_at_mut(Vector* vector, size_t i) { return &vector->elems[i]; }

void Vector_init(Vector* vector, size_t n) {
    vector->elems = malloc(n * sizeof(int64_t));
    vector->n = n;
}

void Vector_destroy(Vector* vector) { free(vector->elems); }

void Vector_init_clone(Vector* clone, const Vector* vector) {
    Vector_init(clone, vector->n);
    memcpy(clone->elems, vector->elems, vector->n * sizeof(int64_t));
}

void Vector_init_random(
    Vector* vector,
    size_t n,
    unsigned short state[3],
    int64_t low,
    int64_t high
) {
    Vector_init(vector, n);
    for (size_t i = 0; i < n; i++) {
        *vector_at_mut(vector, i) = random_in(state, low, high);
    }
}

void Vector_init_slice(
    Vector* slice,
    const Vector* vector,
    size_t start,
    size_t n
) {
    Vector_init(slice, n);
    const int64_t* src = vector->elems + start;
    int64_t* dst = slice->elems;
    memcpy(dst, src, n * sizeof(int64_t));
}

void Vector_init_merged(
    Vector* merged,
    const Vector* left,
    const Vector* right
) {
    Vector_init(merged, left->n + right->n);
    size_t left_size = left->n * sizeof(int64_t);
    size_t right_size = right->n * sizeof(int64_t);
    memcpy(merged->elems, left->elems, left_size);
    memcpy(merged->elems + left->n, right->elems, right_size);
}

void Vector_init_matrix_product(
    Vector* product,
    const Vector* left,
    const Matrix* right
) {
    assert(left->n == right->row_n);

    Vector_init(product, right->col_n);

    for (size_t i = 0; i < right->col_n; i++) {
        int64_t sum = 0;
        for (size_t j = 0; j < left->n; j++) {
            sum += *vector_at(left, j) * *matrix_at(right, j, i);
        }
        *vector_at_mut(product, i) = sum;
    }
}

void Vector_add(Vector* left, const Vector* right) {
    assert(left->n == right->n);

    for (size_t i = 0; i < left->n; i++) {
        *vector_at_mut(left, i) += *vector_at(right, i);
    }
}

void Vector_scale(Vector* vector, int64_t scalar) {
    for (size_t i = 0; i < vector->n; i++) {
        *vector_at_mut(vector, i) *= scalar;
    }
}

int64_t Vector_min(const Vector* vector) {
    int64_t min = *vector_at(vector, 0);
    for (size_t i = 0; i < vector->n; i++) {
        int64_t curr = *vector_at(vector, i);
        if (curr < min) {
            min = curr;
        }
    }
    return min;
}

void Vector_fmt(char** str, const Vector* vector) {
    // Number: 1 + floor(log10(2^63-1)) = 19 chars
    // Space / newline: 1 char
    // \0: 1 char
    size_t size = (vector->n * 19) + 1;
    *str = malloc(size);
    (*str)[0] = '\0';

    char* ptr = *str;
    for (size_t i = 0; i < vector->n; i++) {
        ptr += sprintf(ptr, "%li", *vector_at(vector, i));

        if (i != vector->n - 1) {
            ptr += sprintf(ptr, " ");
        }
    }
}
