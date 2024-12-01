#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Constantes de configuração da simulação
#define GRID_SIZE 2000      // Tamanho da grade
#define TIME_STEPS 500      // Número de passos de tempo
#define DIFFUSION_COEFF 0.1 // Coeficiente de difusão
#define DELTA_TIME 0.01     // Incremento de tempo
#define DELTA_SPACE 1.0     // Incremento espacial

// Estrutura para armazenar os parâmetros da simulação
typedef struct {
    int grid_size;          // Tamanho da grade quadrada
    int time_steps;         // Número de iterações
    double diffusion_coeff; // Coeficiente de difusão
    double delta_time;      // Incremento de tempo
    double delta_space;     // Incremento espacial
} SimulationConfig;

// Aloca dinamicamente uma matriz 2D de doubles
double** allocate_2d_array(int size) {
    // Aloca memória para as linhas da matriz
    double** array = (double**)malloc(size * sizeof(double*));
    if (!array) {
        fprintf(stderr, "Falha na alocação de memória\n");
        exit(EXIT_FAILURE);
    }

    // Aloca memória para cada coluna, inicializando com zeros
    for (int i = 0; i < size; i++) {
        array[i] = (double*)calloc(size, sizeof(double));
        if (!array[i]) {
            fprintf(stderr, "Falha na alocação de memória\n");
            exit(EXIT_FAILURE);
        }
    }

    return array;
}

// Libera a memória alocada para uma matriz 2D
void free_2d_array(double** array, int size) {
    // Libera primeiro cada linha
    for (int i = 0; i < size; i++) {
        free(array[i]);
    }
    // Depois libera o ponteiro principal
    free(array);
}

// Inicializa a grade com um ponto de fonte central
void initialize_grid(double** grid, int size) {
    // Define uma concentração unitária no centro da grade
    grid[size / 2][size / 2] = 1.0;
}

// Resolução da equação de difusão em versão paralela
void solve_diffusion_equation_parallel(double** C, double** C_new, const SimulationConfig* config) {
    // Itera sobre todos os passos de tempo
    for (int t = 0; t < config->time_steps; t++) {
        double difmedio = 0.;

        // Região paralela com redução para cálculo da diferença média
        // Usa collapse(2) para paralelizar loops aninhados
        #pragma omp parallel for collapse(2) reduction(+ : difmedio)
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                // Discretização da equação de difusão 2D
                C_new[i][j] = C[i][j] + config->diffusion_coeff * config->delta_time * (
                    (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) 
                    / (config->delta_space * config->delta_space)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        // Atualiza a grade original em paralelo
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }
    }
}

// Resolução da equação de difusão em versão sequencial
void solve_diffusion_equation_sequential(double** C, double** C_new, const SimulationConfig* config) {
    // Itera sobre todos os passos de tempo
    for (int t = 0; t < config->time_steps; t++) {
        double difmedio = 0.;

        // Calcula novos valores para cada ponto da grade
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                // Discretização da equação de difusão 2D
                C_new[i][j] = C[i][j] + config->diffusion_coeff * config->delta_time * (
                    (C[i + 1][j] + C[i - 1][j] + C[i][j + 1] + C[i][j - 1] - 4 * C[i][j]) 
                    / (config->delta_space * config->delta_space)
                );
                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        // Atualiza a grade original
        for (int i = 1; i < config->grid_size - 1; i++) {
            for (int j = 1; j < config->grid_size - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        // Imprime resultados intermediários a cada 100 passos
        if ((t % 100) == 0) {
            printf("Iteração %d - Diferença = %g\n", t, difmedio / ((config->grid_size - 2) * (config->grid_size - 2)));
        }
    }
}

// Compara os resultados entre as versões sequencial e paralela
double compare_results(double** C1, double** C2, int size) {
    double max_diff = 0.0;
    
    // Encontra a máxima diferença absoluta entre as duas grades
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double diff = fabs(C1[i][j] - C2[i][j]);
            max_diff = (diff > max_diff) ? diff : max_diff;
        }
    }
    return max_diff;
}

int main() {
    // Configurações de número de threads para teste
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Configuração da simulação
    SimulationConfig config = {
        .grid_size = GRID_SIZE,
        .time_steps = TIME_STEPS,
        .diffusion_coeff = DIFFUSION_COEFF,
        .delta_time = DELTA_TIME,
        .delta_space = DELTA_SPACE
    };

    // Imprime cabeçalho da tabela
    printf("%-10s %-20s %-20s %-10s %-10s\n", "Threads", "Tempo Sequencial", "Tempo Paralelo", "Speedup", "Precisão");

    // Testes de desempenho para diferentes números de threads
    for (int test = 0; test < num_tests; test++) {
        int num_threads = thread_counts[test];
        omp_set_num_threads(num_threads);

        // Aloca e inicializa arrays para versão sequencial
        double** C_seq = allocate_2d_array(config.grid_size);
        double** C_new_seq = allocate_2d_array(config.grid_size);
        initialize_grid(C_seq, config.grid_size);

        // Aloca e inicializa arrays para versão paralela
        double** C_par = allocate_2d_array(config.grid_size);
        double** C_new_par = allocate_2d_array(config.grid_size);
        initialize_grid(C_par, config.grid_size);
 
        // Mede tempo de computação sequencial
        double start_time = omp_get_wtime();
        solve_diffusion_equation_sequential(C_seq, C_new_seq, &config);
        double sequential_time = omp_get_wtime() - start_time;

        // Mede tempo de computação paralela
        start_time = omp_get_wtime();
        solve_diffusion_equation_parallel(C_par, C_new_par, &config);
        double parallel_time = omp_get_wtime() - start_time;

        // Calcula precisão e speedup
        double precision = compare_results(C_seq, C_par, config.grid_size);
        double speedup = sequential_time / parallel_time;

        // Imprime resultados na tabela
        printf("%-10d %-20.6f %-20.6f %-10.2f %-10.2e\n",
               num_threads, sequential_time, parallel_time,
               speedup, precision);

        // Libera memória alocada
        free_2d_array(C_seq, config.grid_size);
        free_2d_array(C_new_seq, config.grid_size);
        free_2d_array(C_par, config.grid_size);
        free_2d_array(C_new_par, config.grid_size);
    }

    return EXIT_SUCCESS;
}