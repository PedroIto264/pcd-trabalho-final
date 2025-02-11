#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define GRID_SIZE 2000
#define TIME_STEPS 500
#define DIFFUSION_COEFF 0.1
#define DELTA_TIME 0.01
#define DELTA_SPACE 1.0
#define BLOCK_SIZE 16

typedef struct {
    int grid_size;
    int time_steps;
    double diffusion_coeff;
    double delta_time;
    double delta_space;
} SimulationConfig;

__global__ void diffusion_kernel(double* C, double* C_new, int size, double diffusion_coeff,
                               double delta_time, double delta_space) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        int idx = i * size + j;
        C_new[idx] = C[idx] + diffusion_coeff * delta_time * (
            (C[(i + 1) * size + j] + C[(i - 1) * size + j] +
             C[i * size + (j + 1)] + C[i * size + (j - 1)] - 4 * C[idx])
            / (delta_space * delta_space)
        );
    }
}

void solve_diffusion_equation_cuda(double* C, double* C_new, int size, double diffusion_coeff,
                                  double delta_time, double delta_space, float* cuda_time) {
    double *d_C, *d_C_new;
    size_t array_size = size * size * sizeof(double);

    cudaMalloc(&d_C, array_size);
    cudaMalloc(&d_C_new, array_size);

    cudaMemcpy(d_C, C, array_size, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    diffusion_kernel<<<grid_dim, block_dim>>>(d_C, d_C_new, size, diffusion_coeff, delta_time, delta_space);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(cuda_time, start, stop);

    cudaMemcpy(C_new, d_C_new, array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_C);
    cudaFree(d_C_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void solve_diffusion_equation_mpi_hybrid(double* local_C, double* local_C_new, int local_size,
                                        const SimulationConfig* config, int rank, MPI_Comm cart_comm,
                                        double* total_mpi_time, double* total_cuda_time) {
    int i, j, t;
    int left, right, top, bottom;
    MPI_Status status;

    // Calcular ranks vizinhos
    MPI_Cart_shift(cart_comm, 0, 1, &top, &bottom); // Vizinhos acima e abaixo
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // Vizinhos à esquerda e à direita

    float cuda_time = 0.0; // Alterado para float
    double mpi_time = 0.0;

    for (t = 0; t < config->time_steps; t++) {
        double start_mpi = MPI_Wtime();

        // Troca de informações nas bordas
        MPI_Sendrecv(&local_C[1], local_size, MPI_DOUBLE, top, 0,
                     &local_C[local_size + 1], local_size, MPI_DOUBLE, bottom, 0, cart_comm, &status);
        MPI_Sendrecv(&local_C[local_size * (local_size - 2) + 1], local_size, MPI_DOUBLE, bottom, 0,
                     &local_C[1], local_size, MPI_DOUBLE, top, 0, cart_comm, &status);
        MPI_Sendrecv(&local_C[local_size], local_size, MPI_DOUBLE, left, 0,
                     &local_C[local_size * (local_size - 1) + 1], local_size, MPI_DOUBLE, right, 0, cart_comm, &status);
        MPI_Sendrecv(&local_C[local_size * 2 - 1], local_size, MPI_DOUBLE, right, 0,
                     &local_C[local_size], local_size, MPI_DOUBLE, left, 0, cart_comm, &status);

        mpi_time += MPI_Wtime() - start_mpi;

        // Cálculo da difusão usando CUDA
        float cuda_step_time = 0.0; // Alterado para float
        solve_diffusion_equation_cuda(local_C, local_C_new, local_size, config->diffusion_coeff,
                                     config->delta_time, config->delta_space, &cuda_step_time);
        cuda_time += cuda_step_time;

        // Atualização da grade
        #pragma omp parallel for collapse(2)
        for (i = 1; i < local_size - 1; i++) {
            for (j = 1; j < local_size - 1; j++) {
                local_C[i * local_size + j] = local_C_new[i * local_size + j];
            }
        }
    }

    *total_mpi_time = mpi_time;
    *total_cuda_time = (double)cuda_time / 1000.0; // Convertendo de milissegundos para segundos
}

int main(int argc, char** argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    SimulationConfig config = {
        .grid_size = GRID_SIZE,
        .time_steps = TIME_STEPS,
        .diffusion_coeff = DIFFUSION_COEFF,
        .delta_time = DELTA_TIME,
        .delta_space = DELTA_SPACE
    };

    // Criar uma grade 2D de processos
    int dims[2] = {0, 0};
    MPI_Dims_create(num_procs, 2, dims); // Cria uma grade 2D equilibrada
    int periods[2] = {0, 0}; // Não usar condições periódicas
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int local_size = GRID_SIZE / dims[0]; // Tamanho local da grade
    double* local_C = (double*)calloc(local_size * local_size, sizeof(double));
    double* local_C_new = (double*)calloc(local_size * local_size, sizeof(double));

    if (rank == 0) {
        local_C[local_size / 2 * local_size + local_size / 2] = 1.0;
    }

    double start_time = MPI_Wtime();
    double total_mpi_time = 0.0, total_cuda_time = 0.0;
    solve_diffusion_equation_mpi_hybrid(local_C, local_C_new, local_size, &config, rank, cart_comm,
                                       &total_mpi_time, &total_cuda_time);
    double elapsed_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        printf("Tempo de execução MPI Híbrido: %.6f segundos\n", elapsed_time);
        printf("Tempo gasto em comunicação MPI: %.6f segundos\n", total_mpi_time);
        printf("Eficiência MPI: %.2f%%\n", (1 - (total_mpi_time / elapsed_time)) * 100);
    }

    free(local_C);
    free(local_C_new);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}