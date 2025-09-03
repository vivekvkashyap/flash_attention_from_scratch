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

void cpu_multi_head_attn(float *c_Q, float *c_K, float *c_V, float *c_O){
    int bs = batch_size;
    int n_h = n_head;
    int s_l = sequence_length;
    int h_d = head_dim;
    
    float *S = (float*)malloc(bs * n_h * s_l * s_l * sizeof(float));
    float *P = (float*)malloc(bs * n_h * s_l * s_l * sizeof(float));
    
    for (int i = 0; i < bs; i++){
        for (int j = 0; j < n_h; j++){
            
            for (int k = 0; k < s_l; k++){        
                for (int l = 0; l < s_l; l++){    
                    float tmp = 0.0;
                    for (int d = 0; d < h_d; d++){ 
                        float q_val = c_Q[i * (n_h * s_l * h_d) + j * (s_l * h_d) + k * (h_d) + d];
                        float k_val = c_K[i * (n_h * s_l * h_d) + j * (s_l * h_d) + l * (h_d) + d];
                        tmp += q_val * k_val;
                    }
                    S[i * (n_h * s_l * s_l) + j * (s_l * s_l) + k * (s_l) + l] = tmp;
                }
            }
            
            for (int k = 0; k < s_l; k++){        
                
                float denominator = 0.0;
                for (int l = 0; l < s_l; l++){    
                    denominator += exp(S[i * (n_h * s_l * s_l) + j * (s_l * s_l) + k * (s_l) + l]);
                }
                
                for (int l = 0; l < s_l; l++){    
                    P[i * (n_h * s_l * s_l) + j * (s_l * s_l) + k * (s_l) + l] = 
                        exp(S[i * (n_h * s_l * s_l) + j * (s_l * s_l) + k * (s_l) + l]) / denominator;
                }
            }
            
            for (int k = 0; k < s_l; k++){        
                for (int d = 0; d < h_d; d++){    
                    float tmp2 = 0.0;
                    for (int l = 0; l < s_l; l++){
                        float p_val = P[i * (n_h * s_l * s_l) + j * (s_l * s_l) + k * (s_l) + l];
                        float v_val = c_V[i * (n_h * s_l * h_d) + j * (s_l * h_d) + l * (h_d) + d];
                        tmp2 += p_val * v_val;
                    }
                    c_O[i * (n_h * s_l * h_d) + j * (s_l * h_d) + k * (h_d) + d] = tmp2;
                }
            }
        }
    }

    free(S);
    free(P);
}


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

__global__ void flash_attn_kernel_1(float *d_q, float *d_k, float *d_v, float *d_o, float *d_l, float *d_m){
    __shared__ float k_shared[Bc * head_dim];
    __shared__ float v_shared[Bc * head_dim];
    __shared__ float q_shared[Br * head_dim];
    __shared__ float o_shared[Br * head_dim];
    __shared__ float l_shared[Br];
    __shared__ float m_shared[Br];

    int qkv_offset = (blockIdx.y * (n_head * sequence_length * head_dim)) + (blockIdx.x * (sequence_length * head_dim));
    int lm_offset = (blockIdx.y * (n_head * sequence_length)) + (blockIdx.x * (sequence_length));

    float *d_q_orig = d_q + qkv_offset;
    float *d_k_orig = d_k + qkv_offset;
    float *d_v_orig = d_v + qkv_offset;
    float *d_o_orig = d_o + qkv_offset;
    float *d_l_orig = d_l + lm_offset;
    float *d_m_orig = d_m + lm_offset;

    d_q = d_q_orig;
    d_k = d_k_orig;
    d_v = d_v_orig;
    d_o = d_o_orig;
    d_l = d_l_orig;
    d_m = d_m_orig;

    float l_new = 0.0;
    float m_new = -INFINITY;

    for (int j = 0; j < T_c; j++){

        for (int p1 = 0; p1 < head_dim; p1++){

            if (threadIdx.x < Bc) {
                v_shared[threadIdx.x * head_dim + p1] = d_v[threadIdx.x * head_dim + p1];
                k_shared[threadIdx.x * head_dim + p1] = d_k[threadIdx.x * head_dim + p1];
            }
        }
        __syncthreads();

        d_q = d_q_orig;
        d_o = d_o_orig;
        d_l = d_l_orig;
        d_m = d_m_orig;

        for (int i = 0; i < T_r; i++){

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
            __syncthreads();
            
            float S[Bc]; 
            float S_masked[Bc];
            float m_tilda = -INFINITY;
            for (int w1 = 0; w1 < Bc; w1++){
                float tmp = 0.0;
                for (int e1 = 0; e1 < head_dim; e1 ++){
                    tmp += q_shared[threadIdx.x * head_dim + e1] * k_shared[w1 * head_dim + e1];
                }
                S[w1] = tmp * softmax_scale;

            for (int h = 0; h < Bc; h++){
                if (j < i || (j == i && h <= threadIdx.x)){  
                    S_masked[h] = S[h];
                } else {
                    S_masked[h] = -INFINITY;
                }
                m_tilda = fmaxf(m_tilda, S_masked[h]);
            }

            float P_tilda[Bc]; 
            float l_tilda = 0.0; 
            for (int f1 =0; f1 < Bc; f1++){
                float diff = S_masked[f1] - m_tilda; 
                P_tilda[f1] = exp(diff); 
                l_tilda += P_tilda[f1];

            }
            __syncthreads();

            m_new = MAX(m_shared[threadIdx.x], m_tilda);
            l_new = (exp(m_shared[threadIdx.x] - m_new) * l_shared[threadIdx.x]) + (exp(m_tilda - m_new) * l_tilda);

            float first_term = 0.0;
            float second_term = 0.0;
            
            for (int g1 = 0; g1 < head_dim; g1++){
                first_term = ((l_shared[threadIdx.x] * exp(m_shared[threadIdx.x] - m_new)) * o_shared[threadIdx.x * head_dim + g1]);
                
                float tmp = 0.0;
                for (int v1 = 0; v1 < Bc; v1++){
                    tmp += P_tilda[v1] * v_shared[v1 * head_dim + g1];
                }
                
                second_term = exp(m_tilda - m_new) * tmp;
                d_o[threadIdx.x * head_dim + g1] = (1 / l_new) * (first_term + second_term);
            }
            __syncthreads();
            
            d_l[threadIdx.x] = l_new;
            d_m[threadIdx.x] = m_new; 
            }
            if ((i+1) != T_r){
                d_q += Br * head_dim;
                d_o += Br * head_dim;
                d_l += Br;
                d_m += Br;
            }
            
        }
        __syncthreads();
        if ((j+1) != T_c){
            d_k += Bc * head_dim;
            d_v += Bc * head_dim;
        }
    }
}

//  using way too much shared memory and back_S should be shared and not calculated for each row
// __global__ void flash_attn_backward_kernel_1(float *d_q, float *d_k, float *d_v, float *d_o, float *d_l, float *d_m, float *back_q, float *back_k, float *back_v, float *back_o){

//     __shared__ float k_shared[Bc * head_dim];
//     __shared__ float v_shared[Bc * head_dim];
//     __shared__ float q_shared[Br * head_dim];
//     __shared__ float o_shared[Br * head_dim];
//     __shared__ float l_shared[Br];
//     __shared__ float m_shared[Br];

//     __shared__ float back_k_tilda_shared[Bc * head_dim];
//     __shared__ float back_v_tilda_shared[Bc * head_dim];
//     __shared__ float back_q_shared[Br * head_dim];
//     __shared__ float back_o_shared[Br * head_dim];

//     __shared__ float back_S_shared[Br * Bc];

//     int qkv_offset = (blockIdx.y * (n_head * sequence_length * head_dim)) + (blockIdx.x * (sequence_length * head_dim));
//     int lm_offset = (blockIdx.y * (n_head * sequence_length)) + (blockIdx.x * (sequence_length));

//     float *d_q_orig = d_q + qkv_offset;
//     float *d_k_orig = d_k + qkv_offset;
//     float *d_v_orig = d_v + qkv_offset;
//     float *d_o_orig = d_o + qkv_offset;
//     float *d_l_orig = d_l + lm_offset;
//     float *d_m_orig = d_m + lm_offset;

//     float *back_q_orig = back_q + qkv_offset;
//     float *back_k_orig = back_k + qkv_offset;
//     float *back_v_orig = back_v + qkv_offset;
//     float *back_o_orig = back_o + qkv_offset;

//     for (int j = 0; j < T_c; j++){

//         for (int p1 = 0; p1 < head_dim; p1++){

//             if (threadIdx.x < Bc) {
//                 v_shared[threadIdx.x * head_dim + p1] = d_v[threadIdx.x * head_dim + p1];
//                 k_shared[threadIdx.x * head_dim + p1] = d_k[threadIdx.x * head_dim + p1];

//                 back_k_tilda_shared[threadIdx.x * head_dim + p1] = 0.0;
//                 back_v_tilda_shared[threadIdx.x * head_dim + p1] = 0.0;

//             }
//         }
//         __syncthreads();
        

//         d_q = d_q_orig;
//         d_o = d_o_orig;
//         d_l = d_l_orig;
//         d_m = d_m_orig;

//         back_q = back_q_orig;
//         back_o = back_o_orig;

//         for (int i = 0; i < T_r; i++){

//             for (int a1 = 0; a1 < head_dim; a1++){
//                 if (threadIdx.x < Bc) {
//                     q_shared[threadIdx.x * head_dim + a1] = d_q[threadIdx.x * head_dim + a1]; 
//                     o_shared[threadIdx.x * head_dim + a1] = d_o[threadIdx.x * head_dim + a1];

//                     back_q_shared[threadIdx.x * head_dim + a1] = back_q[threadIdx.x * head_dim + a1];
//                     back_o_shared[threadIdx.x * head_dim + a1] = back_o[threadIdx.x * head_dim + a1];
//                 }

//             }
//             if (threadIdx.x < Bc) {
//                 l_shared[threadIdx.x] = d_l[threadIdx.x];
//                 m_shared[threadIdx.x] = d_m[threadIdx.x];
//             }
//             __syncthreads();

//             float S[Bc]; 
//             float S_masked[Bc];
//             float back_S[Bc];
//             float m_tilda = -INFINITY;
//             for (int w1 = 0; w1 < Bc; w1++){
//                 float tmp = 0.0;
//                 for (int e1 = 0; e1 < head_dim; e1 ++){
//                     tmp += q_shared[threadIdx.x * head_dim + e1] * k_shared[w1 * head_dim + e1];
//                 }
//                 S[w1] = tmp * softmax_scale;

//             for (int h = 0; h < Bc; h++){
//                 if (j < i || (j == i && h <= threadIdx.x)){  
//                     S_masked[h] = S[h];
//                 } else {
//                     S_masked[h] = -INFINITY;
//                 }
//             }
            
//             float P[Bc]; 
//             float P_tilda[Bc];
//             float back_P[Bc];
//             float l_tilda = 0.0; 
//             for (int f1 =0; f1 < Bc; f1++){
//                 float diff = S_masked[f1] - m_shared[threadIdx.x]; 
//                 P_tilda[f1] =  (1 / d_l[threadIdx.x]) * (exp(diff) - d_m[threadIdx.x]); 
//             }
//             __syncthreads();

//             for (int l = 0; l < head_dim; l++){
//                 float tmp2 = 0.0;
//                 for (int t = 0; t < Br; t++){
//                     tmp2 += P_tilda[t * Bc + threadIdx.x] * back_o[t * head_dim + l];
//                 }
//                 back_v_tilda_shared[threadIdx.x * head_dim + l] += tmp2;
//             }

//             for (int n = 0; n < Bc; n++){
//                 float tmp3 = 0.0;
//                 for (int y = 0; y < head_dim; y++){
//                     tmp3 += back_o_shared[threadIdx.x * head_dim + y] * v_shared[n * head_dim + y];
//                 }
//                 // back_P[threadIdx.x * Bc + n] = tmp3;
//                 back_P[n] = tmp3;
//             }

//             float D[Br];
//             float tmp4 = 0.0;
//             for (int h = 0; h < head_dim; h++){
//                 tmp4 += back_o_shared[threadIdx.x * head_dim + h] * o_shared[threadIdx.x * head_dim + h];
//             }
//             D[threadIdx.x] = tmp4;
//             }
            
//             for (int f = 0; f < Bc; f++){
//                 // back_S[threadIdx.x * Bc + f] = d_P[threadIdx.x * Bc + f] * (back_P[threadIdx.x * Bc + f] - D[threadIdx.x]);
//                 back_S[f] = P_tilda[threadIdx.x * Bc + f] * (back_P[threadIdx.x * Bc + f] - D[threadIdx.x]);
//             }
//             __syncthreads();

//             for (int o = 0; o < head_dim; o++){
//                 float tmp5 = 0.0;
//                 for (int g = 0; g < Bc; g++){
//                     // tmp5 += back_S[threadIdx.x * Bc + g] * k_shared[g * d + o];
//                     tmp5 += back_S[g] * k_shared[g * head_dim + o];
//                 }
//                 back_q_shared[threadIdx.x * head_dim + o] += tmp5 * softmax_scale;
//                 back_q[threadIdx.x * head_dim + o] =  back_q_shared[threadIdx.x * head_dim + o];
//             }

//             for (int t = 0; t < head_dim; t++){
//                 float tmp6 = 0.0;
//                 for (int z = 0; z < Bc; z++){
//                     // tmp6 += back_S[z * Bc + threadIdx.x] * q_shared[z * d + t];
//                     tmp6 += back_S[z * Bc + threadIdx.x] * q_shared[z * head_dim + t];
//                 }
//                 back_k_tilda_shared[threadIdx.x * head_dim + t] += tmp6 * softmax_scale;
//             }
//             __syncthreads();

//             if ((i+1) != T_r){
//                 d_q += Br * head_dim;
//                 d_o += Br * head_dim;
//                 d_l += Br;
//                 d_m += Br;

//                 back_q += Br * head_dim;
//                 back_o += Br * head_dim;
//             }


//         }                   
//         for (int g = 0; g < head_dim; g++){
//             back_k[threadIdx.x * head_dim + g] = back_k_tilda_shared[threadIdx.x * head_dim + g];
//             back_v[threadIdx.x * head_dim + g] = back_v_tilda_shared[threadIdx.x * head_dim + g];
//         }   

//         __syncthreads();
//         if ((j+1) != T_c){
//             d_k += Bc * head_dim;
//             d_v += Bc * head_dim;

//             back_k += Bc * head_dim;
//             back_v += Bc * head_dim;
//         }

//     }
// }

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
    for (int i=0; i<3; i++){
        multi_head_attention<<<dimGrid_mha, dimBlock_mha>>>(d_q, d_k, d_v, d_S, d_P, d_o_gpu);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking multi_attn implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        multi_head_attention<<<dimGrid_mha, dimBlock_mha>>>(d_q, d_k, d_v, d_S, d_P, d_o_gpu);
        cudaDeviceSynchronize();
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    CUDA_CHECK(cudaMemcpy(h_o_cpu, d_o_gpu, o_size, cudaMemcpyDeviceToHost));

    dim3 dimGrid_flash(n_head, batch_size);
    dim3 dimBlock_flash(Bc);

    printf("Performing flash_attn warmup runs...\n");
    for (int i=0; i<3; i++){
        flash_attn_kernel_1<<<dimGrid_flash, dimBlock_flash>>>(d_q, d_k, d_v, d_o, d_l, d_m);
        cudaDeviceSynchronize();
    }
    
    printf("Benchmarking flash_attn implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        flash_attn_kernel_1<<<dimGrid_flash, dimBlock_flash>>>(d_q, d_k, d_v, d_o, d_l, d_m);
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
