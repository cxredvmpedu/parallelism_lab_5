/*
 * Лабораторна робота 5
 * на тему "Повідомлення. Бібліотека Open MPI"
 * з дисципліни "Програмне забезпечення високопродуктивних комп’ютерних систем"
 * Варіант: 2
 * Вираз:   A = min(C) * Z + D * (MX * MR)
 * Виконав: Чигрин Віталій Сергійович, студент групи ІМ-32
 * Дата:    12.05.2026
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"

constexpr int N = 3000;
constexpr int TASK_N = 6;

static_assert(N % TASK_N == 0, "N must be divisible by TASK_N");
constexpr int H = N / TASK_N;

constexpr int LOW = 0;
constexpr int HIGH = 10;

// Введення вхідних змінних
void input(int rank, Vector* C, Vector* D, Vector* Z, Matrix* MR, Matrix* MX);
void input_T1(Vector* C, Vector* Z);
void input_T2(Matrix* MX);
void input_T6(Vector* D, Matrix* MR);

// Обчислює вираз за алгоритмом
void compute(int rank, Vector* C, Vector* D, Vector* Z, Matrix* MR, Matrix* MX);
void compute_T1(Vector* C, Vector* Z);
void compute_T2(Matrix* MX);
void compute_T3();
void compute_T4();
void compute_T5();
void compute_T6(Vector* D, Matrix* MR);

void lab_5() {
    MPI_Init(nullptr, nullptr);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != TASK_N) {
        if (world_rank == 1) {
            fprintf(stderr, "Need %d proccesses, got %d\n", TASK_N, world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Task T%d started\n", world_rank + 1);

    // Вхідні змінні
    Cleanup(Vector) C = {}, D = {}, Z = {};
    Cleanup(Matrix) MR = {}, MX = {};
    input(world_rank, &C, &D, &Z, &MR, &MX);

    compute(world_rank, &C, &D, &Z, &MR, &MX);

    printf("Task T%d finished\n", world_rank + 1);

    MPI_Finalize();
}

int main() {
    lab_5();
    return EXIT_SUCCESS;
}

void input(int rank, Vector* C, Vector* D, Vector* Z, Matrix* MR, Matrix* MX) {
    switch (rank) {
    case 0:
        input_T1(C, Z);
        break;
    case 1:
        input_T2(MX);
        break;
    case 5:
        input_T6(D, MR);
        break;
    }
}

void input_T1(Vector* C, Vector* Z) {
    unsigned short state[3];
    state_init(state, 0);

    // Введення
    Vector_init_random(C, N, state, LOW, HIGH);
    Vector_init_random(Z, N, state, LOW, HIGH);

    // Виведення
    PRINT_VECTOR("C", C);
    PRINT_VECTOR("Z", Z);
}

void input_T2(Matrix* MX) {
    unsigned short state[3];
    state_init(state, 1);

    // Введення
    Matrix_init_random(MX, N, N, state, LOW, HIGH);

    // Виведення
    PRINT_MATRIX("MX", MX);
}

void input_T6(Vector* D, Matrix* MR) {
    unsigned short state[3];
    state_init(state, 5);

    // Введення
    Vector_init_random(D, N, state, LOW, HIGH);
    Matrix_init_random(MR, N, N, state, LOW, HIGH);

    // Виведення
    PRINT_VECTOR("D", D);
    PRINT_MATRIX("MR", MR);
}

void compute(
    int rank,
    Vector* C,
    Vector* D,
    Vector* Z,
    Matrix* MR,
    Matrix* MX
) {
    switch (rank) {
    case 0:
        compute_T1(C, Z);
        break;
    case 1:
        compute_T2(MX);
        break;
    case 2:
        compute_T3();
        break;
    case 3:
        compute_T4();
        break;
    case 4:
        compute_T5();
        break;
    case 5:
        compute_T6(D, MR);
        break;
    }
}

// Допоміжна функція обчислення частини виразу A_H = a * Z_H + D * (MX * MR_H)
void compute_A_H(
    Vector* A_H,
    int64_t a,
    const Vector* Z_H,
    const Vector* D_local,
    const Matrix* MX_local,
    const Matrix* MR_H
) {
    Cleanup(Vector) X_H = {};
    Vector_init_clone(&X_H, Z_H);
    Vector_scale(&X_H, a);

    Cleanup(Matrix) MX_MR_H = {};
    Matrix_init_product(&MX_MR_H, MX_local, MR_H);

    Cleanup(Vector) Y_H = {};
    Vector_init_matrix_product(&Y_H, D_local, &MX_MR_H);

    Vector_init_clone(A_H, &X_H);
    Vector_add(A_H, &Y_H);
}

void compute_T1(Vector* C, Vector* Z) {
    // 2. Передати C2, Z2, C3, Z3 до T2
    MPI_Send(C->elems + H, 2 * H, MPI_INT64_T, 1, 201, MPI_COMM_WORLD);
    MPI_Send(Z->elems + H, 2 * H, MPI_INT64_T, 1, 202, MPI_COMM_WORLD);

    // 3. Передати C4, Z4, C5, Z5, C6, Z6 до T6
    MPI_Send(C->elems + 3 * H, 3 * H, MPI_INT64_T, 5, 601, MPI_COMM_WORLD);
    MPI_Send(Z->elems + 3 * H, 3 * H, MPI_INT64_T, 5, 602, MPI_COMM_WORLD);

    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init_slice(&C_H, C, 0, H);
    Vector_init_slice(&Z_H, Z, 0, H);

    // 4. Прийняти MX від T2
    Cleanup(Matrix) MX = {};
    Matrix_init(&MX, N, N);
    MPI_Recv(
        MX.elems,
        N * N,
        MPI_INT64_T,
        1,
        101,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 5. Прийняти MR1, MR2, MR3 та D від T6
    Cleanup(Matrix) MR123 = {};
    Matrix_init(&MR123, N, 3 * H);
    Cleanup(Vector) D = {};
    Vector_init(&D, N);
    MPI_Recv(
        MR123.elems,
        N * 3 * H,
        MPI_INT64_T,
        5,
        102,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        D.elems,
        N,
        MPI_INT64_T,
        5,
        103,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 6. Передати MX до T6
    MPI_Send(MX.elems, N * N, MPI_INT64_T, 5, 603, MPI_COMM_WORLD);

    // 7. Передати MR2, MR3, D до T2
    Cleanup(Matrix) MR23 = {};
    Matrix_init_slice_cols(&MR23, &MR123, H, 2 * H);
    MPI_Send(MR23.elems, N * 2 * H, MPI_INT64_T, 1, 203, MPI_COMM_WORLD);
    MPI_Send(D.elems, N, MPI_INT64_T, 1, 204, MPI_COMM_WORLD);

    Cleanup(Matrix) MR_H = {};
    Matrix_init_slice_cols(&MR_H, &MR123, 0, H);

    // 8. Обчислити локальний мінімум a1 = min(C1)
    int64_t a1 = Vector_min(&C_H);

    // 9. Прийняти a56 від T6
    int64_t a56;
    MPI_Recv(&a56, 1, MPI_INT64_T, 5, 104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 10. Прийняти a234 від T2
    int64_t a234;
    MPI_Recv(&a234, 1, MPI_INT64_T, 1, 105, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 11. Обчислити глобальний мінімум a = min(a1, a234, a56)
    int64_t a = min(a1, min(a234, a56));

    // 12. Передати a до T2
    MPI_Send(&a, 1, MPI_INT64_T, 1, 206, MPI_COMM_WORLD);

    // 13. Передати a до T6
    MPI_Send(&a, 1, MPI_INT64_T, 5, 604, MPI_COMM_WORLD);

    // 14. Обчислити A1
    Cleanup(Vector) A1 = {};
    compute_A_H(&A1, a, &Z_H, &D, &MX, &MR_H);

    // 15. Прийняти A5, A6 від T6
    Cleanup(Vector) A5 = {}, A6 = {};
    Vector_init(&A5, H);
    Vector_init(&A6, H);
    MPI_Recv(
        A5.elems,
        H,
        MPI_INT64_T,
        5,
        106,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        A6.elems,
        H,
        MPI_INT64_T,
        5,
        107,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 16. Передати A1, A5, A6 до T2
    MPI_Send(A1.elems, H, MPI_INT64_T, 1, 209, MPI_COMM_WORLD);
    MPI_Send(A5.elems, H, MPI_INT64_T, 1, 210, MPI_COMM_WORLD);
    MPI_Send(A6.elems, H, MPI_INT64_T, 1, 211, MPI_COMM_WORLD);
}

void compute_T2(Matrix* MX) {
    // 2. Передати MX до T1
    MPI_Send(MX->elems, N * N, MPI_INT64_T, 0, 101, MPI_COMM_WORLD);

    // 3. Передати MX до T3
    MPI_Send(MX->elems, N * N, MPI_INT64_T, 2, 301, MPI_COMM_WORLD);

    // 4. Прийняти C2, Z2, C3, Z3 від T1
    Cleanup(Vector) C23 = {}, Z23 = {};
    Vector_init(&C23, 2 * H);
    Vector_init(&Z23, 2 * H);
    MPI_Recv(
        C23.elems,
        2 * H,
        MPI_INT64_T,
        0,
        201,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        Z23.elems,
        2 * H,
        MPI_INT64_T,
        0,
        202,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 5. Передати C3, Z3 до T3
    MPI_Send(C23.elems + H, H, MPI_INT64_T, 2, 302, MPI_COMM_WORLD);
    MPI_Send(Z23.elems + H, H, MPI_INT64_T, 2, 303, MPI_COMM_WORLD);

    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init_slice(&C_H, &C23, 0, H);
    Vector_init_slice(&Z_H, &Z23, 0, H);

    // 6. Прийняти MR2, MR3 та D від T1
    Cleanup(Matrix) MR23 = {};
    Matrix_init(&MR23, N, 2 * H);
    Cleanup(Vector) D = {};
    Vector_init(&D, N);
    MPI_Recv(
        MR23.elems,
        N * 2 * H,
        MPI_INT64_T,
        0,
        203,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        D.elems,
        N,
        MPI_INT64_T,
        0,
        204,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 7. Передати MR3, D до T3
    Cleanup(Matrix) MR3 = {};
    Matrix_init_slice_cols(&MR3, &MR23, H, H);
    MPI_Send(MR3.elems, N * H, MPI_INT64_T, 2, 304, MPI_COMM_WORLD);
    MPI_Send(D.elems, N, MPI_INT64_T, 2, 305, MPI_COMM_WORLD);

    Cleanup(Matrix) MR_H = {};
    Matrix_init_slice_cols(&MR_H, &MR23, 0, H);

    // 8. Обчислити a2
    int64_t a2 = Vector_min(&C_H);

    // 9. Прийняти a34 від T3
    int64_t a34;
    MPI_Recv(&a34, 1, MPI_INT64_T, 2, 205, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 10. Обчислити a234
    int64_t a234 = min(a2, a34);

    // 11. Передати a234 до T1
    MPI_Send(&a234, 1, MPI_INT64_T, 0, 105, MPI_COMM_WORLD);

    // 12. Прийняти a від T1
    int64_t a;
    MPI_Recv(&a, 1, MPI_INT64_T, 0, 206, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 13. Передати a до T3
    MPI_Send(&a, 1, MPI_INT64_T, 2, 306, MPI_COMM_WORLD);

    // 14. Обчислити A2
    Cleanup(Vector) A2 = {};
    compute_A_H(&A2, a, &Z_H, &D, MX, &MR_H);

    // 15. Прийняти A3, A4 від T3
    Cleanup(Vector) A3 = {}, A4 = {};
    Vector_init(&A3, H);
    Vector_init(&A4, H);
    MPI_Recv(
        A3.elems,
        H,
        MPI_INT64_T,
        2,
        207,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        A4.elems,
        H,
        MPI_INT64_T,
        2,
        208,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 16. Прийняти A1, A5, A6 від T1
    Cleanup(Vector) A1 = {}, A5 = {}, A6 = {};
    Vector_init(&A1, H);
    Vector_init(&A5, H);
    Vector_init(&A6, H);
    MPI_Recv(
        A1.elems,
        H,
        MPI_INT64_T,
        0,
        209,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        A5.elems,
        H,
        MPI_INT64_T,
        0,
        210,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        A6.elems,
        H,
        MPI_INT64_T,
        0,
        211,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 17. Об'єднання A
    Cleanup(Vector) A = {};
    Vector_init(&A, N);
    memcpy(A.elems + 0 * H, A1.elems, H * sizeof(int64_t));
    memcpy(A.elems + 1 * H, A2.elems, H * sizeof(int64_t));
    memcpy(A.elems + 2 * H, A3.elems, H * sizeof(int64_t));
    memcpy(A.elems + 3 * H, A4.elems, H * sizeof(int64_t));
    memcpy(A.elems + 4 * H, A5.elems, H * sizeof(int64_t));
    memcpy(A.elems + 5 * H, A6.elems, H * sizeof(int64_t));

    // Виведення A
    PRINT_VECTOR("A", &A);
}

void compute_T3() {
    // 1. Прийняти MX від T2
    Cleanup(Matrix) MX = {};
    Matrix_init(&MX, N, N);
    MPI_Recv(
        MX.elems,
        N * N,
        MPI_INT64_T,
        1,
        301,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 2. Передати MX до T4
    MPI_Send(MX.elems, N * N, MPI_INT64_T, 3, 410, MPI_COMM_WORLD);

    // 3. Прийняти C3, Z3 від T2
    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init(&C_H, H);
    Vector_init(&Z_H, H);
    MPI_Recv(
        C_H.elems,
        H,
        MPI_INT64_T,
        1,
        302,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        Z_H.elems,
        H,
        MPI_INT64_T,
        1,
        303,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 4. Прийняти MR3, D від T2
    Cleanup(Matrix) MR_H = {};
    Matrix_init(&MR_H, N, H);
    Cleanup(Vector) D = {};
    Vector_init(&D, N);
    MPI_Recv(
        MR_H.elems,
        N * H,
        MPI_INT64_T,
        1,
        304,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        D.elems,
        N,
        MPI_INT64_T,
        1,
        305,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 5. a3 = min(C3)
    int64_t a3 = Vector_min(&C_H);

    // 6. Прийняти a4 від T4
    int64_t a4;
    MPI_Recv(&a4, 1, MPI_INT64_T, 3, 307, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 7. a34 = min(a3, a4)
    int64_t a34 = min(a3, a4);

    // 8. Передати a34 до T2
    MPI_Send(&a34, 1, MPI_INT64_T, 1, 205, MPI_COMM_WORLD);

    // 9. Прийняти a від T2
    int64_t a;
    MPI_Recv(&a, 1, MPI_INT64_T, 1, 306, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 10. Передати a до T4
    MPI_Send(&a, 1, MPI_INT64_T, 3, 405, MPI_COMM_WORLD);

    // 11. Обчислити A3
    Cleanup(Vector) A3 = {};
    compute_A_H(&A3, a, &Z_H, &D, &MX, &MR_H);

    // 12. Прийняти A4 від T4
    Cleanup(Vector) A4 = {};
    Vector_init(&A4, H);
    MPI_Recv(
        A4.elems,
        H,
        MPI_INT64_T,
        3,
        308,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 13. Передати A3, A4 до T2
    MPI_Send(A3.elems, H, MPI_INT64_T, 1, 207, MPI_COMM_WORLD);
    MPI_Send(A4.elems, H, MPI_INT64_T, 1, 208, MPI_COMM_WORLD);
}

void compute_T4() {
    // 1. Прийняти MX від T3
    Cleanup(Matrix) MX = {};
    Matrix_init(&MX, N, N);
    MPI_Recv(
        MX.elems,
        N * N,
        MPI_INT64_T,
        2,
        410,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 2. Прийняти C4, Z4 від T5
    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init(&C_H, H);
    Vector_init(&Z_H, H);
    MPI_Recv(
        C_H.elems,
        H,
        MPI_INT64_T,
        4,
        401,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        Z_H.elems,
        H,
        MPI_INT64_T,
        4,
        402,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 3. Прийняти MR4, D від T5
    Cleanup(Matrix) MR_H = {};
    Matrix_init(&MR_H, N, H);
    Cleanup(Vector) D = {};
    Vector_init(&D, N);
    MPI_Recv(
        MR_H.elems,
        N * H,
        MPI_INT64_T,
        4,
        403,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        D.elems,
        N,
        MPI_INT64_T,
        4,
        404,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 4. a4 = min(C4)
    int64_t a4 = Vector_min(&C_H);

    // 5. Передати a4 до T3
    MPI_Send(&a4, 1, MPI_INT64_T, 2, 307, MPI_COMM_WORLD);

    // 6. Прийняти a від T3
    int64_t a;
    MPI_Recv(&a, 1, MPI_INT64_T, 2, 405, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 7. Обчислити A4 = X4 + Y4
    Cleanup(Vector) A4 = {};
    compute_A_H(&A4, a, &Z_H, &D, &MX, &MR_H);

    // 8. Передати A4 до T3
    MPI_Send(A4.elems, H, MPI_INT64_T, 2, 308, MPI_COMM_WORLD);
}

void compute_T5() {
    // 1. Прийняти MR4, MR5 та D від T6
    Cleanup(Matrix) MR45 = {};
    Matrix_init(&MR45, N, 2 * H);
    Cleanup(Vector) D = {};
    Vector_init(&D, N);
    MPI_Recv(
        MR45.elems,
        N * 2 * H,
        MPI_INT64_T,
        5,
        501,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        D.elems,
        N,
        MPI_INT64_T,
        5,
        502,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 2. Прийняти C4, Z4, C5, Z5 від T6
    Cleanup(Vector) C45 = {}, Z45 = {};
    Vector_init(&C45, 2 * H);
    Vector_init(&Z45, 2 * H);
    MPI_Recv(
        C45.elems,
        2 * H,
        MPI_INT64_T,
        5,
        503,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        Z45.elems,
        2 * H,
        MPI_INT64_T,
        5,
        504,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 3. Прийняти MX від T6
    Cleanup(Matrix) MX = {};
    Matrix_init(&MX, N, N);
    MPI_Recv(
        MX.elems,
        N * N,
        MPI_INT64_T,
        5,
        506,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 4. Передати C4, Z4 до T4
    MPI_Send(C45.elems, H, MPI_INT64_T, 3, 401, MPI_COMM_WORLD);
    MPI_Send(Z45.elems, H, MPI_INT64_T, 3, 402, MPI_COMM_WORLD);

    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init_slice(&C_H, &C45, H, H);
    Vector_init_slice(&Z_H, &Z45, H, H);

    // 5. Передати MR4, D до T4
    Cleanup(Matrix) MR4 = {};
    Matrix_init_slice_cols(&MR4, &MR45, 0, H);
    MPI_Send(MR4.elems, N * H, MPI_INT64_T, 3, 403, MPI_COMM_WORLD);
    MPI_Send(D.elems, N, MPI_INT64_T, 3, 404, MPI_COMM_WORLD);

    Cleanup(Matrix) MR_H = {};
    Matrix_init_slice_cols(&MR_H, &MR45, H, H);

    // 6. a5 = min(C5)
    int64_t a5 = Vector_min(&C_H);

    // 7. Передати a5 до T6
    MPI_Send(&a5, 1, MPI_INT64_T, 5, 605, MPI_COMM_WORLD);

    // 8. Прийняти a від T6
    int64_t a;
    MPI_Recv(&a, 1, MPI_INT64_T, 5, 505, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 9. Обчислити A5 = X5 + Y5
    Cleanup(Vector) A5 = {};
    compute_A_H(&A5, a, &Z_H, &D, &MX, &MR_H);

    // 10. Передати A5 до T6
    MPI_Send(A5.elems, H, MPI_INT64_T, 5, 606, MPI_COMM_WORLD);
}

void compute_T6(Vector* D, Matrix* MR) {
    // 2. Передати MR1, MR2, MR3 та D до T1
    Cleanup(Matrix) MR123 = {};
    Matrix_init_slice_cols(&MR123, MR, 0, 3 * H);
    MPI_Send(MR123.elems, N * 3 * H, MPI_INT64_T, 0, 102, MPI_COMM_WORLD);
    MPI_Send(D->elems, N, MPI_INT64_T, 0, 103, MPI_COMM_WORLD);

    // 3. Передати MR4, MR5 та D до T5
    Cleanup(Matrix) MR45 = {};
    Matrix_init_slice_cols(&MR45, MR, 3 * H, 2 * H);
    MPI_Send(MR45.elems, N * 2 * H, MPI_INT64_T, 4, 501, MPI_COMM_WORLD);
    MPI_Send(D->elems, N, MPI_INT64_T, 4, 502, MPI_COMM_WORLD);

    Cleanup(Matrix) MR_H = {};
    Matrix_init_slice_cols(&MR_H, MR, 5 * H, H);

    // 4. Прийняти C4, C5, C6, Z4, Z5, Z6 від T1
    Cleanup(Vector) C456 = {}, Z456 = {};
    Vector_init(&C456, 3 * H);
    Vector_init(&Z456, 3 * H);
    MPI_Recv(
        C456.elems,
        3 * H,
        MPI_INT64_T,
        0,
        601,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
    MPI_Recv(
        Z456.elems,
        3 * H,
        MPI_INT64_T,
        0,
        602,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 5. Передати C4, C5, Z4, Z5 до T5
    MPI_Send(C456.elems, 2 * H, MPI_INT64_T, 4, 503, MPI_COMM_WORLD);
    MPI_Send(Z456.elems, 2 * H, MPI_INT64_T, 4, 504, MPI_COMM_WORLD);

    Cleanup(Vector) C_H = {}, Z_H = {};
    Vector_init_slice(&C_H, &C456, 2 * H, H);
    Vector_init_slice(&Z_H, &Z456, 2 * H, H);

    // 6. Прийняти MX від T1
    Cleanup(Matrix) MX = {};
    Matrix_init(&MX, N, N);
    MPI_Recv(
        MX.elems,
        N * N,
        MPI_INT64_T,
        0,
        603,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 7. Передати MX до T5
    MPI_Send(MX.elems, N * N, MPI_INT64_T, 4, 506, MPI_COMM_WORLD);

    // 8. Обчислити a6 = min(C6)
    int64_t a6 = Vector_min(&C_H);

    // 9. Прийняти a5 від T5
    int64_t a5;
    MPI_Recv(&a5, 1, MPI_INT64_T, 4, 605, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 10. Обчислити a56 = min(a5, a6)
    int64_t a56 = min(a5, a6);

    // 11. Передати a56 до T1
    MPI_Send(&a56, 1, MPI_INT64_T, 0, 104, MPI_COMM_WORLD);

    // 12. Прийняти a від T1
    int64_t a;
    MPI_Recv(&a, 1, MPI_INT64_T, 0, 604, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 13. Передати a до T5
    MPI_Send(&a, 1, MPI_INT64_T, 4, 505, MPI_COMM_WORLD);

    // 14. Обчислити A6 = X6 + Y6
    Cleanup(Vector) A6 = {};
    compute_A_H(&A6, a, &Z_H, D, &MX, &MR_H);

    // 15. Прийняти A5 від T5
    Cleanup(Vector) A5 = {};
    Vector_init(&A5, H);
    MPI_Recv(
        A5.elems,
        H,
        MPI_INT64_T,
        4,
        606,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    // 16. Передати A5, A6 до T1
    MPI_Send(A5.elems, H, MPI_INT64_T, 0, 106, MPI_COMM_WORLD);
    MPI_Send(A6.elems, H, MPI_INT64_T, 0, 107, MPI_COMM_WORLD);
}
