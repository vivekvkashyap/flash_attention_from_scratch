#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define batch_size 32
#define sequence_length 1024
#define hidden_dim 512
#define n_head 8
#define head_dim 64
#define T_c 32
#define T_r 32
#define Bc 32
#define Br 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define softmax_scale 0.125 // (1 / 64)

__global__ void multi_head_attention(float *d_q, float *d_k, float *d_v, float *d_S, float *d_P, float *d_o){

    int qkv_offset = (blockIdx.y * (n_head * sequence_length * head_dim)) + (blockIdx.x * (sequence_length * head_dim));
    int sp_offset = (blockIdx.y * (n_head * sequence_length * sequence_length)) + (blockIdx.x * (sequence_length * sequence_length));
    d_q += qkv_offset;
    d_k += qkv_offset;
    d_v += qkv_offset;
    d_o += qkv_offset;
    d_S += sp_offset;
    d_P += sp_offset;
    
    for (int w1 = 0; w1 < sequence_length; w1++){
        float tmp = 0.0;
        for (int k1 = 0; k1 < head_dim; k1++){
            tmp += d_q[threadIdx.x * head_dim + k1] * d_k[w1 * head_dim + k1];
        }
        d_S[threadIdx.x * sequence_length + w1] = tmp * softmax_scale;
    }
    __syncthreads();

    for (int h = 0; h < sequence_length; h++){
        if (h > threadIdx.x) {
            d_S[threadIdx.x * sequence_length + h] = -INFINITY;
        }
    }

    float denominator = 0.0;
    for (int l1 = 0; l1 < sequence_length; l1++){
        denominator += exp(d_S[threadIdx.x * sequence_length + l1]);
    }
    __syncthreads();

    for (int d1 = 0; d1 < sequence_length; d1++){
        d_P[threadIdx.x * sequence_length + d1] = exp(d_S[threadIdx.x * sequence_length + d1]) / denominator;
    }
    __syncthreads();

    for (int b1 = 0; b1 < head_dim; b1++){
        float tmp2 = 0.0;
        for (int v1 = 0; v1 < sequence_length; v1++){
            tmp2 += d_P[threadIdx.x * sequence_length + v1] * d_v[v1 * head_dim + b1];
        }
        d_o[threadIdx.x * head_dim + b1] = tmp2;
    }
}

__global__ void flash_attn_kernel_2(float *d_q, float *d_k, float *d_v, float *d_o, float *d_l, float *d_m){
    // extern __shared__ float smem[];
    // float *q_shared = smem;
    // float *k_shared = q_shared + Br * head_dim;
    // float *v_shared = k_shared + Bc * head_dim;
    // float *s_shared = v_shared + Bc * head_dim;
    // float *o_shared = s_shared + Br * Bc;
    // float *l_shared = o_shared + Br * head_dim;
    // float *m_shared = l_shared + Br;

    // extern __shared__ float smem[];
    // float *q_shared = smem;                           // Size: Br * head_dim
    // float *k_shared = q_shared + Br * head_dim;      // Size: Bc * head_dim  
    // float *v_shared = k_shared + Bc * head_dim;      // Size: Bc * head_dim
    // float *s_shared = v_shared + Bc * head_dim;      // Size: Br * Bc (FIXED!)
    // float *o_shared = s_shared + Br * Bc;            // Size: Br * head_dim (FIXED!)
    // float *l_shared = o_shared + Br * head_dim;      // Size: Br (FIXED!)
    // float *m_shared = l_shared + Br;   
    __shared__ float k_shared[Bc * head_dim];
    __shared__ float v_shared[Bc * head_dim];
    __shared__ float q_shared[Br * head_dim];
    __shared__ float o_shared[Br * head_dim];
    __shared__ float l_shared[Br];
    __shared__ float m_shared[Br];
    __shared__ float s_shared[Br * Bc];

    int q_offset = (blockIdx.y * (sequence_length * head_dim)) + (blockIdx.x * (Br * head_dim));
    int kv_offset = (blockIdx.y * (sequence_length * head_dim));
    int lm_offset = (blockIdx.y * (sequence_length)) + (blockIdx.x * (Br));

    float *d_q_orig = d_q + q_offset;
    float *d_k_orig = d_k + kv_offset;
    float *d_v_orig = d_v + kv_offset;
    float *d_o_orig = d_o + q_offset;
    float *d_l_orig = d_l + lm_offset;
    float *d_m_orig = d_m + lm_offset;

    d_q = d_q_orig;
    d_k = d_k_orig;
    d_v = d_v_orig;
    d_o = d_o_orig;
    d_l = d_l_orig;
    d_m = d_m_orig;

    for (int a1 = 0; a1 < head_dim; a1++){
        if (threadIdx.x < Bc) {
            q_shared[threadIdx.x * head_dim + a1] = d_q[threadIdx.x * head_dim + a1]; 
            o_shared[threadIdx.x * head_dim + a1] = d_o[threadIdx.x * head_dim + a1];
        }
    }

    if (threadIdx.x < Bc) {
        l_shared[threadIdx.x] = d_l[threadIdx.x];
        m_shared[threadIdx.x] = d_m[threadIdx.x];
    }

    float prev_m_tilda = -INFINITY;

    for (int j = 0; j < T_c; j++){
        for (int p1 = 0; p1 < head_dim; p1++){
            if (threadIdx.x < Bc) {
                v_shared[threadIdx.x * head_dim + p1] = d_v[threadIdx.x * head_dim + p1];
                k_shared[threadIdx.x * head_dim + p1] = d_k[threadIdx.x * head_dim + p1];
            }
        }

        // float m_tilda = -INFINITY;
        // float S[Bc]; 
        // float S_masked[Bc];
        prev_m_tilda = m_shared[threadIdx.x];
        for (int w1 = 0; w1 < Bc; w1++){
            float tmp = 0.0;
            int i = blockIdx.x;
            if (j < i || (j == i && w1 <= threadIdx.x)){
                for (int e1 = 0; e1 < head_dim; e1 ++){
                    tmp += q_shared[threadIdx.x * head_dim + e1] * k_shared[w1 * head_dim + e1];     
                }
                s_shared[threadIdx.x * Bc + w1] = tmp * softmax_scale;
                m_shared[threadIdx.x] = fmaxf(m_shared[threadIdx.x], s_shared[threadIdx.x * Bc + w1]);
            } else {
                s_shared[threadIdx.x * Bc + w1] = -INFINITY;
            }

        float P_tilda[Bc]; 
        float summ = 0.0;
        for (int f1 =0; f1 < Bc; f1++){
            float diff = s_shared[threadIdx.x * Bc + f1] - m_shared[threadIdx.x] ; 
            P_tilda[f1] = exp(diff); 
            summ += P_tilda[f1];
        }

        __syncthreads();

        l_shared[threadIdx.x] = exp(prev_m_tilda - m_shared[threadIdx.x]) * l_shared[threadIdx.x] + summ;

        for (int g1 = 0; g1 < head_dim; g1++){
            float first_term = 1 / (exp(prev_m_tilda - m_shared[threadIdx.x])) * o_shared[threadIdx.x * head_dim + g1];
            float tmp = 0.0;
            for (int v1 = 0; v1 < Bc; v1++){
                tmp += P_tilda[v1] * v_shared[v1 * head_dim + g1];
            }
        
            o_shared[threadIdx.x * head_dim + g1] = (first_term + tmp);
        }
        __syncthreads();
        if ((j+1) != T_c){
            d_k += Bc * head_dim;
            d_v += Bc * head_dim;
        }
        
        // d_o += Br * head_dim;
        // d_l += Br;
        // d_m += Br;
        
    }
    for (int l1 = 0; l1 < head_dim; l1++){
        if (threadIdx.x < Br) { 
            o_shared[threadIdx.x * head_dim + l1] = (1 / l_shared[threadIdx.x]) * o_shared[threadIdx.x * head_dim + l1];
            d_o[threadIdx.x * head_dim + l1] = o_shared[threadIdx.x * head_dim + l1];
        }
    }

    d_l[threadIdx.x] = m_shared[threadIdx.x] + log(l_shared[threadIdx.x]);
    }
    __syncthreads();
    
    d_o = d_o_orig;
    d_l = d_l_orig;
    d_m = d_m_orig;
}


void init_matrix(float *mat, int bs, int n_h, int s_l, int h_d){
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < n_h; j++) {
            for (int k = 0; k < s_l; k++){
                for (int l = 0; l < h_d; l++){
                    mat[i * n_h * s_l * h_d + j * s_l * h_d + k * h_d + l] = (float)rand() / RAND_MAX;
                }   
            }
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_q, *h_k, *h_v, *h_o, *h_l, *h_m;
    float *d_q, *d_k, *d_v, *d_o, *d_l, *d_m;
    float *h_S, *h_P, *h_o_cpu;
    float *d_S, *d_P, *d_o_gpu;

    size_t q_size = batch_size * n_head * sequence_length * head_dim * sizeof(float);
    size_t k_size = batch_size * n_head * sequence_length * head_dim * sizeof(float);
    size_t v_size = batch_size * n_head * sequence_length * head_dim * sizeof(float);

    size_t o_size = batch_size * n_head * sequence_length * head_dim * sizeof(float);
    size_t l_size = batch_size * n_head * sequence_length * sizeof(float);
    size_t m_size = batch_size * n_head * sequence_length * sizeof(float);

    size_t S_size = batch_size * n_head * sequence_length * sequence_length * sizeof(float);
    size_t P_size = batch_size * n_head * sequence_length * sequence_length * sizeof(float);

    h_q = (float*)malloc(q_size);
    h_k = (float*)malloc(k_size);
    h_v = (float*)malloc(v_size);

    h_o = (float*)malloc(o_size);
    h_l = (float*)malloc(l_size);
    h_m = (float*)malloc(m_size);

    h_S = (float*)malloc(S_size);
    h_P = (float*)malloc(P_size);
    h_o_cpu = (float*)malloc(o_size);

    srand(time(NULL));
    init_matrix(h_q, batch_size, n_head, sequence_length, head_dim);
    init_matrix(h_k, batch_size, n_head, sequence_length, head_dim);
    init_matrix(h_v, batch_size, n_head, sequence_length, head_dim);

    memset(h_o, 0, o_size);
    memset(h_l, 0, l_size); 

    memset(h_S, 0, S_size);
    memset(h_P, 0, P_size);

    size_t actual_m_size = m_size / sizeof(float);
    for (size_t i = 0; i < actual_m_size; i++){
        h_m[i] = -INFINITY;
    }

    CUDA_CHECK(cudaMalloc(&d_q, q_size));
    CUDA_CHECK(cudaMalloc(&d_k, k_size));
    CUDA_CHECK(cudaMalloc(&d_v, v_size));
    CUDA_CHECK(cudaMalloc(&d_o, o_size));
    CUDA_CHECK(cudaMalloc(&d_l, l_size));
    CUDA_CHECK(cudaMalloc(&d_m, m_size));
    CUDA_CHECK(cudaMalloc(&d_S, S_size));
    CUDA_CHECK(cudaMalloc(&d_P, P_size));
    CUDA_CHECK(cudaMalloc(&d_o_gpu, o_size));

    CUDA_CHECK(cudaMemcpy(d_q, h_q, q_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, k_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, v_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_o, h_o, o_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l, h_l, l_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, m_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, S_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, h_P, P_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_o_gpu, h_o_cpu, o_size, cudaMemcpyHostToDevice));

    dim3 dimGrid_mha(n_head, batch_size);
    dim3 dimBlock_mha(sequence_length);

    printf("Performing multi_attn warmup runs...\n");
    for (int i=0; i<2; i++){
        multi_head_attention<<<dimGrid_mha, dimBlock_mha>>>(d_q, d_k, d_v, d_S, d_P, d_o_gpu);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking multi_attn implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 2; i++) {
        double start_time = get_time();
        multi_head_attention<<<dimGrid_mha, dimBlock_mha>>>(d_q, d_k, d_v, d_S, d_P, d_o_gpu);
        cudaDeviceSynchronize();
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    CUDA_CHECK(cudaMemcpy(h_o_cpu, d_o_gpu, o_size, cudaMemcpyDeviceToHost));

    dim3 dimGrid_flash(T_r, batch_size * n_head);
    dim3 dimBlock_flash(Bc);

    // printf("Host: d_o address = %p\n", (void*)d_o);
    // flash_attn_kernel_2<<<dimGrid_flash, dimBlock_flash>>>(d_q, d_k, d_v, d_o, d_l, d_m);
    // CUDA_CHECK(cudaGetLastError());
    // printf("Host: d_o address after launch = %p\n", (void*)d_o);
    // cudaDeviceSynchronize();
    // printf("Host: d_o address after sync = %p\n", (void*)d_o);

    printf("Performing flash_attn warmup runs...\n");
    for (int i=0; i<2; i++){
        flash_attn_kernel_2<<<dimGrid_flash, dimBlock_flash>>>(d_q, d_k, d_v, d_o, d_l, d_m);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking flash_attn implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 2; i++) {
        double start_time = get_time();
        flash_attn_kernel_2<<<dimGrid_flash, dimBlock_flash>>>(d_q, d_k, d_v, d_o, d_l, d_m);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("mutli-head attn avg time: %.2f milliseconds\n", cpu_avg_time * 1000);
    printf("flash attn avg time: %.2f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);

    CUDA_CHECK(cudaMemcpy(h_o, d_o, o_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_l, d_l, l_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m, d_m, m_size, cudaMemcpyDeviceToHost));

    bool correct = true;
    int mismatches = 0;
    for (int i = 0; i < batch_size; i++){
        for (int j = 0; j < n_head; j++){
            for (int k = 0; k < sequence_length; k++){
                for (int l = 0; l < head_dim; l++){
                    int index = (i * (n_head * sequence_length * head_dim)) + (j * (sequence_length * head_dim)) + (k * (head_dim)) + l;
                    if (fabs(h_o_cpu[index] - h_o[index]) > 1e-3) {
                        printf("Mismatch at [%d][%d]: multi_attn=%.6f, flash_attn=%.6f\n", i, j, h_o_cpu[index], h_o[index]);
                        mismatches++;
                        correct = false;
                    }
                    
                }
            }
        }
    }

    if (correct) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect (%d mismatches shown)\n", mismatches);
    }
    

    free(h_q);
    free(h_k);
    free(h_v);
    free(h_o);
    free(h_l);
    free(h_m);
    free(h_S);
    free(h_P);
    free(h_o_cpu);
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_l));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_o_gpu));

    return 0;
}

