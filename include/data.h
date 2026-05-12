#pragma once

#include <stddef.h>
#include <stdint.h>

#define Cleanup(Type) __attribute__((cleanup(Type##_cleanup))) Type

#define PRINT_VECTOR(name_str, vector_ptr)     \
    do {                                       \
        char* _str = nullptr;                  \
        Vector_fmt(&_str, (vector_ptr));       \
        printf("%s:\n%s\n", (name_str), _str); \
        free(_str);                            \
    } while (0)

#define PRINT_MATRIX(name_str, matrix_ptr)     \
    do {                                       \
        char* _str = nullptr;                  \
        Matrix_fmt(&_str, (matrix_ptr));       \
        printf("%s:\n%s\n", (name_str), _str); \
        free(_str);                            \
    } while (0)

int64_t min(int64_t a, int64_t b);

// ГПВЧ
void state_init(unsigned short seed[3], int rank);
int64_t random_in(unsigned short state[3], int64_t low, int64_t high);

// Матриця
typedef struct {
    int64_t* elems;
    size_t row_n; // Кількість рядків
    size_t col_n; // Кількість стовпців
} Matrix;

void Matrix_init(Matrix* matrix, size_t row_n, size_t col_n);
void Matrix_destroy(Matrix* matrix);
static inline void Matrix_cleanup(Matrix* m) { Matrix_destroy(m); }

void Matrix_init_clone(Matrix* clone, const Matrix* matrix);
void Matrix_init_random(
    Matrix* matrix,
    size_t row_n,
    size_t col_n,
    unsigned short state[3],
    int64_t low,
    int64_t high
);
void Matrix_init_slice_cols(
    Matrix* slice,
    const Matrix* matrix,
    size_t start_col,
    size_t cols_n
);
void Matrix_init_product(
    Matrix* product,
    const Matrix* left,
    const Matrix* right
);

void Matrix_fmt(char** str, const Matrix* matrix);

// Вектор
typedef struct {
    int64_t* elems;
    size_t n; // Кількість елементів
} Vector;

void Vector_init(Vector* vector, size_t n);
void Vector_destroy(Vector* vector);
static inline void Vector_cleanup(Vector* v) { Vector_destroy(v); }

void Vector_init_clone(Vector* clone, const Vector* vector);
void Vector_init_random(
    Vector* vector,
    size_t n,
    unsigned short state[3],
    int64_t low,
    int64_t high
);
void Vector_init_slice(
    Vector* slice,
    const Vector* vector,
    size_t start,
    size_t n
);
void Vector_init_merged(
    Vector* merged,
    const Vector* left,
    const Vector* right
);
void Vector_init_matrix_product(
    Vector* product,
    const Vector* left,
    const Matrix* right
);

void Vector_add(Vector* left, const Vector* right);
void Vector_scale(Vector* vector, int64_t scalar);
int64_t Vector_min(const Vector* vector);

void Vector_fmt(char** str, const Vector* vector);
