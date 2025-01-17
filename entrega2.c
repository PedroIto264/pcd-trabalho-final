#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

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

double** allocate_2d_array(int size) {
    double** array = (double**)malloc(size * sizeof(double*));
    if (!array) {
        fprintf(stderr, "Falha na alocação de memória\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        array[i] = (double*)calloc(size, sizeof(double));
        if (!array[i]) {
            fprintf(stderr, "Falha na alocação de memória\n");
            exit(EXIT_FAILURE);
        }
    }
    return array;
}

void free_2d_array(double** array, int size) {
    for (int i = 0; i < size; i++) {
        free(array[i]);
    }
    free(array);
}

void initialize_grid(double** grid, int size) {
    grid[size / 2][size / 2] = 1.0;
}

double* convert_to_1d(double** array_2d, int size) {
    double* array_1d = (double*)malloc(size * size * sizeof(double));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            array_1d[i * size + j] = array_2d[i][j];
        }
    }
    return array_1d;
}

void convert_to_2d(double* array_1d, double** array_2d, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            array_2d[i][j] = array_1d[i * size + j];
        }
    }
}

void solve_diffusion_equation_cuda(double** C, double** C_new, const SimulationConfig* config) {
    int size = config->grid_size;
    size_t array_size = size * size * sizeof(double);
    
    double* h_C = convert_to_1d(C, size);
    double* h_C_new = convert_to_1d(C_new, size);
    
    double *d_C, *d_C_new;
    cudaMalloc(&d_C, array_size);
    cudaMalloc(&d_C_new, array_size);
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaMemcpy(d_C, h_C, array_size, cudaMemcpyHostToDevice);
    
    for (int t = 0; t < config->time_steps; t++) {
        diffusion_kernel<<<grid_dim, block_dim>>>(d_C, d_C_new, size,
            config->diffusion_coeff, config->delta_time, config->delta_space);
        
        double* temp = d_C;
        d_C = d_C_new;
        d_C_new = temp;
    }
    
    cudaMemcpy(h_C, d_C, array_size, cudaMemcpyDeviceToHost);
    
    convert_to_2d(h_C, C, size);
    
    cudaFree(d_C);
    cudaFree(d_C_new);
    free(h_C);
    free(h_C_new);
}

void solve_diffusion_equation_sequential(double** C, double** C_new, const SimulationConfig* config) {
    for (int t = 0; t < config->time_steps; t++) {
        double difmedio = 0.;

        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C_new[i][j] = C[i][j] + config->diffusion_coeff * config->delta_time * (
                    (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) 
                    / (config->delta_space * config->delta_space)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        if ((t % 100) == 0) {
            printf("Iteração %d - Diferença = %g\n", t, difmedio / ((config->grid_size - 2) * (config->grid_size - 2)));
        }
    }
}

void solve_diffusion_equation_parallel(double** C, double** C_new, const SimulationConfig* config) {
    for (int t = 0; t < config->time_steps; t++) {
        double difmedio = 0.;

        #pragma omp parallel for collapse(2) reduction(+ : difmedio)
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C_new[i][j] = C[i][j] + config->diffusion_coeff * config->delta_time * (
                    (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) 
                    / (config->delta_space * config->delta_space)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }
    }
}

double compare_results(double** C1, double** C2, int size) {
    double max_diff = 0.0;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double diff = fabs(C1[i][j] - C2[i][j]);
            max_diff = (diff > max_diff) ? diff : max_diff;
        }
    }
    return max_diff;
}


int main() {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    SimulationConfig config = {
        .grid_size = GRID_SIZE,
        .time_steps = TIME_STEPS,
        .diffusion_coeff = DIFFUSION_COEFF,
        .delta_time = DELTA_TIME,
        .delta_space = DELTA_SPACE
    };
    
    printf("%-10s %-20s %-20s %-20s %-10s %-10s\n", 
           "Threads", "Tempo Sequencial", "Tempo OpenMP", "Tempo CUDA", "Speedup", "Precisão");
    
    for (int test = 0; test < num_tests; test++) {
        int num_threads = thread_coufnts[test];
        omp_set_num_threads(num_threads);
        
        double** C_seq = allocate_2d_array(config.grid_size);
        double** C_new_seq = allocate_2d_array(config.grid_size);
        double** C_omp = allocate_2d_array(config.grid_size);
        double** C_new_omp = allocate_2d_array(config.grid_size);
        double** C_cuda = allocate_2d_array(config.grid_size);
        double** C_new_cuda = allocate_2d_array(config.grid_size);
        
        initialize_grid(C_seq, config.grid_size);
        initialize_grid(C_omp, config.grid_size);
        initialize_grid(C_cuda, config.grid_size);
        
        double start_time = omp_get_wtime();
        solve_diffusion_equation_sequential(C_seq, C_new_seq, &config);
        double sequential_time = omp_get_wtime() - start_time;
        
        start_time = omp_get_wtime();
        solve_diffusion_equation_parallel(C_omp, C_new_omp, &config);
        double openmp_time = omp_get_wtime() - start_time;
        
        start_time = omp_get_wtime();
        solve_diffusion_equation_cuda(C_cuda, C_new_cuda, &config);
        double cuda_time = omp_get_wtime() - start_time;
        
        double precision_omp = compare_results(C_seq, C_omp, config.grid_size);
        double precision_cuda = compare_results(C_seq, C_cuda, config.grid_size);
        double speedup_cuda = sequential_time / cuda_time;
        
        printf("%-10d %-20.6f %-20.6f %-20.6f %-10.2f %-10.2e\n",
               num_threads, sequential_time, openmp_time, cuda_time,
               speedup_cuda, precision_cuda);
        
        free_2d_array(C_seq, config.grid_size);
        free_2d_array(C_new_seq, config.grid_size);
        free_2d_array(C_omp, config.grid_size);
        free_2d_array(C_new_omp, config.grid_size);
        free_2d_array(C_cuda, config.grid_size);
        free_2d_array(C_new_cuda, config.grid_size);
    }
    
    return EXIT_SUCCESS;
}