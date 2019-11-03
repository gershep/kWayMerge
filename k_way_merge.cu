/* 
* Parallel k-way merge algorithm for merging k sorted lists with arbitrary sizes.
* Number of lists is bounded above by 1024, but you can get around this with iterative application
* of the algorithm. Lists are divided in sublists that can be merged independently. 
* Sublist are merged iteratively two at a time using a bitonic network.
* More details: http://algo2.iti.kit.edu/documents/GPUSortingMerging.pdf
*/

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_simple_wrapper.h"

#define SUBLIST_COUNT     		30    	// number of generated sublists using splitters; best set to a multiple of SM count	 	
#define	PARTITION_BLOCK_SIZE	1024	// works well, no need to change it
#define MERGE_BLOCK_SIZE		128		// works well, no need to change it

#define NUM_BANKS 				32
#define LOG_NUM_BANKS 			5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 

__device__ int binary_search_greater_than(double *x, int l, int u, double d) {
    while (l <= u) {
        int i = (l+u) / 2;
        if (d >= x[i]) {
            l = i + 1;
        } else {
            u = i - 1;
        }
    }
    
    return l;	
}

__device__ int binary_search_smaller_than(double *x, int l, int u, double d) {
    while (l <= u) {
        int i = (l+u) / 2;
        if (d <= x[i]) {
            u = i - 1;
        } else {
            l = i + 1;
        }
    }
    
    return u;
}

__device__ void warp_reduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__device__ void exclusive_scan(int *sdata, int n, int thid) {
    int offset = 1; 

    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        
        if (thid < d) { 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            sdata[bi] += sdata[ai];
        }
        
        offset *= 2;
    }
    
    __syncthreads();

    if (thid == 0) { 
        sdata[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        
        if (thid < d) {		 			
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
        
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
}

__device__ int random_integer(curandState *state, int low, int upp) {
    return curand(state) % (upp-low+1) + low;
}

__global__ void setup_kernel(curandState *state, unsigned long long offset) {
    curand_init(offset, 0, 0, state);
}

__global__ void generate_splitters(double *E, int k, int spl_size, int *lower_splitter, int *upper_splitter, curandState *state) {
    // at first, we will only use the first PARTITION_BLOCK_SIZE elements
    // later, we will use offsets to avoid bank conflicts in the scan algorithm
    __shared__ int lower_splitter_copy[(PARTITION_BLOCK_SIZE/NUM_BANKS)*(NUM_BANKS+1)];
    __shared__ int lower_search_boundary[PARTITION_BLOCK_SIZE];
    __shared__ int upper_search_boundary[PARTITION_BLOCK_SIZE];
    __shared__ int first_smaller_than_pivot[PARTITION_BLOCK_SIZE];
    __shared__ int first_greater_than_pivot[PARTITION_BLOCK_SIZE];
    __shared__ int smaller_count[PARTITION_BLOCK_SIZE];
    __shared__ int smaller_equal_count[PARTITION_BLOCK_SIZE];
    __shared__ double pivot;
    
    if (threadIdx.x < k) {
        lower_splitter_copy[threadIdx.x]   = lower_splitter[threadIdx.x];		// we need it multiple times
        lower_search_boundary[threadIdx.x] = lower_splitter_copy[threadIdx.x];
        upper_search_boundary[threadIdx.x] = upper_splitter[threadIdx.x];		// we only need it once
    }

    __syncthreads();
    
    while (1) {
        // let the first thread pick the pivot element
        if (threadIdx.x == 0) {
            while (1) {
                // pick a thread r at random
                int r = random_integer(state, 0, k-1);
    
                if (lower_search_boundary[r] < upper_search_boundary[r]) {
                    // pick a random index from from [lower, upper>
                    int idx = random_integer(state, lower_search_boundary[r], upper_search_boundary[r]-1);
                    pivot = E[idx];
                    break;
                }
            }
        }

        __syncthreads();
        
        smaller_count[threadIdx.x] = 0;
        smaller_equal_count[threadIdx.x] = 0;
    
        if (threadIdx.x < k) {
            first_greater_than_pivot[threadIdx.x] = 
                binary_search_greater_than(E, lower_search_boundary[threadIdx.x],    upper_search_boundary[threadIdx.x]-1, pivot);
            first_smaller_than_pivot[threadIdx.x] = 
                binary_search_smaller_than(E, lower_search_boundary[threadIdx.x], first_greater_than_pivot[threadIdx.x]-1, pivot);
            
            smaller_count[threadIdx.x] 		 += first_smaller_than_pivot[threadIdx.x] - lower_splitter_copy[threadIdx.x] + 1;
            smaller_equal_count[threadIdx.x] += first_greater_than_pivot[threadIdx.x] - lower_splitter_copy[threadIdx.x];
        }
        
        __syncthreads();
            
        // reductions over smaller_count and smaller_equal_count
        for (int step = PARTITION_BLOCK_SIZE >> 1; step > 32; step >>= 1) {	
            if (threadIdx.x < step) {
                smaller_count[threadIdx.x] += smaller_count[threadIdx.x + step];
                smaller_equal_count[threadIdx.x] += smaller_equal_count[threadIdx.x + step];
            }
            
            __syncthreads();
        }
        
        if (threadIdx.x < 32) {
            warp_reduce(smaller_count, threadIdx.x);
            warp_reduce(smaller_equal_count, threadIdx.x);
        }
        
        __syncthreads();
    
        // if the right pivot is found, stop searching
        if (smaller_count[0] <= spl_size && spl_size < smaller_equal_count[0]) {
            break;
        }
        
        // else, update boundaries and search again
        if (threadIdx.x < k) {
            if (smaller_count[0] > spl_size) {
                upper_search_boundary[threadIdx.x] = first_smaller_than_pivot[threadIdx.x] + 1;
            } else {
                lower_search_boundary[threadIdx.x] = first_greater_than_pivot[threadIdx.x]; 
            }
        }
        
        __syncthreads();		
    }

    int gap = spl_size - smaller_count[0];
    
    // we reuse the lower_splitter_copy buffer to keep the number of pivot elements per list
    int *pivot_count_per_list = lower_splitter_copy;
    
    // to optimize the scan, we use offsets
    int bank_offset = CONFLICT_FREE_OFFSET(threadIdx.x);
    
    // reset everything so we can use the scan algorithm like k = PARTITION_BLOCK_SIZE
    pivot_count_per_list[threadIdx.x + bank_offset] = 0;	
    
    if (threadIdx.x < k) {
        pivot_count_per_list[threadIdx.x + bank_offset] = 
            first_greater_than_pivot[threadIdx.x] - first_smaller_than_pivot[threadIdx.x] - 1;
    }
    
    // exclusive_scan
    exclusive_scan(pivot_count_per_list, PARTITION_BLOCK_SIZE, threadIdx.x);
    
    __syncthreads();
    
    // where to save new splitters
    int *new_splitter = lower_splitter + k;
    
    // create new splitters
    if (threadIdx.x < k) {
        // elements from scan are not necessary adjacent
        int position_now = threadIdx.x + bank_offset; 
        int position_next = threadIdx.x + 1 + CONFLICT_FREE_OFFSET(threadIdx.x+1);
        
        new_splitter[threadIdx.x] = first_smaller_than_pivot[threadIdx.x] + 1;
        
        if (threadIdx.x < k-1 && pivot_count_per_list[position_next] <= gap) {
            new_splitter[threadIdx.x] += pivot_count_per_list[position_next] - pivot_count_per_list[position_now];
        } 
        else if (gap - pivot_count_per_list[position_now] > 0) {
            new_splitter[threadIdx.x] += gap - pivot_count_per_list[position_now];
        }
    }   
}

__device__ void bitonic_merge(double *sdata, int thid) {
    int step = MERGE_BLOCK_SIZE;
    
    while (step > 0) {
        int idx = (thid / step) * step + thid;
        if (sdata[idx] > sdata[idx+step]) {
            double temp = sdata[idx];
            sdata[idx] = sdata[idx+step];
            sdata[idx+step] = temp;
        }
        
        step /= 2;

        __syncthreads();
    }
}

__global__ void gpu_merge(double *E, int k, int *splitters, double *W0, double *W1, int n) {
    __shared__ int lower_splitter[PARTITION_BLOCK_SIZE];
    __shared__ int upper_splitter[PARTITION_BLOCK_SIZE];
    __shared__ double merge_buffer[2*MERGE_BLOCK_SIZE];

    int lists_to_merge = k;
    
    // start position of merged sublist
    int sublist_start_position = 
        blockIdx.x * (n/SUBLIST_COUNT) + (blockIdx.x <= n % SUBLIST_COUNT ? blockIdx.x : n % SUBLIST_COUNT);
    
    // input and output buffer
    double *buff_in  = E;
    double *buff_out = W0;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        lower_splitter[i] = splitters[i+k*blockIdx.x];
        upper_splitter[i] = splitters[i+k*blockIdx.x+k];
    }
    
    __syncthreads();
    
    while (lists_to_merge > 1) {
        for (int i = 0; i < lists_to_merge; i += 2) {
            if (i+1 < lists_to_merge) {
                // merge list_A and list_B into list_C
                double *buff_A = buff_in + lower_splitter[i];
                double *buff_B = buff_in + lower_splitter[i+1];
                
                int size_A = upper_splitter[i]-lower_splitter[i];
                int size_B = upper_splitter[i+1]-lower_splitter[i+1];
            
                __syncthreads();
                
                // start updating splitters for next iteration
                lower_splitter[0] = sublist_start_position;
                upper_splitter[i/2] = lower_splitter[i/2] + size_A + size_B;
                lower_splitter[i/2+1] = upper_splitter[i/2];
    
                double *buff_C = buff_out + lower_splitter[i/2];
                
                // fill merge_buffer with dummy elements 
                merge_buffer[blockDim.x-1-threadIdx.x] = DBL_MAX;
                merge_buffer[threadIdx.x+blockDim.x] = DBL_MAX;
        
                if (threadIdx.x < size_A) {
                    merge_buffer[blockDim.x-1-threadIdx.x] = buff_A[threadIdx.x];
                }
                    
                if (threadIdx.x < size_B) {
                    merge_buffer[threadIdx.x+blockDim.x] = buff_B[threadIdx.x];
                }

                __syncthreads();
                        
                int loaded_A = min(blockDim.x, size_A);			// how many did I load from buffer A
                int loaded_B = min(blockDim.x, size_B);			// how many did I load from buffer B
                int loaded_shared = loaded_A + loaded_B; 		// loaded in shared memory

                buff_A += loaded_A;
                buff_B += loaded_B;
                size_A -= loaded_A;
                size_B -= loaded_B;
                
                while (loaded_shared > 0) {
                    // bitonic merge
                    bitonic_merge(merge_buffer, threadIdx.x);	
                        
                    if (threadIdx.x < loaded_shared) {
                        buff_C[threadIdx.x] = merge_buffer[threadIdx.x];
                    }

                    buff_C += min(blockDim.x, loaded_shared);
                    loaded_shared -= min(blockDim.x, loaded_shared);
                    
                    // choose next block of elements; default = B
                    int next = 0;
                    
                    if (0 < size_A && 0 < size_B) {
                        next = (buff_A[0] <= buff_B[0]);
                    }					
                    else if (0 < size_A) {
                        next = 1;
                    }					

                    __syncthreads();

                    if (next) {
                        if (threadIdx.x < size_A) {
                            merge_buffer[blockDim.x-1-threadIdx.x] = buff_A[threadIdx.x];
                        }
                        else { 
                            merge_buffer[blockDim.x-1-threadIdx.x] = DBL_MAX;
                        }

                        loaded_A = min(blockDim.x, size_A);
                        loaded_shared += loaded_A;
                        buff_A += loaded_A;
                        size_A -= loaded_A;
                    }
                    else {
                        if (threadIdx.x < size_B) {
                            merge_buffer[blockDim.x-1-threadIdx.x] = buff_B[threadIdx.x];
                        }
                        else {
                            merge_buffer[blockDim.x-1-threadIdx.x] = DBL_MAX;
                        }
                        
                        loaded_B = min(blockDim.x, size_B);
                        loaded_shared += loaded_B;
                        buff_B += loaded_B;
                        size_B -= loaded_B;
                    }
    
                    __syncthreads();
                }
            }
            else {	
                double *buff_A = buff_in  + lower_splitter[i];
                double *buff_C = buff_out + lower_splitter[i/2];
                
                int size_A = upper_splitter[i]-lower_splitter[i];
                
                lower_splitter[i/2+1] = lower_splitter[i/2] + size_A;
                upper_splitter[i/2] = lower_splitter[i/2+1];

                // simple memory copy
                for (int j = threadIdx.x; j < size_A; j += blockDim.x) {
                    buff_C[j] = buff_A[j];
                }

                __syncthreads();
            }
        }
        
        if (lists_to_merge == k) {
            buff_in = W1;
        }
        
        // swap buffers
        double *temp = buff_in;
        buff_in = buff_out;
        buff_out = temp;
        
        lists_to_merge = (lists_to_merge+1)/2;
    }
}

// iterative 2-way merging
void cpu_merge(double *A, int k, int *S, double *B) {
    while (k > 1) {
        for (int i = 0; i < k-1; i += 2) {
            int x = S[i];
            int y = S[i+1];
            int z = S[i];
            
            while (x < S[i+1] && y < S[i+2]) {
                if (A[x] <= A[y]) {
                    B[z++] = A[x++];
                } else {
                    B[z++] = A[y++];
                }
            }
                    
            while (x < S[i+1]) {
                B[z++] = A[x++];
            }
            
            while (y < S[i+2]) {
                B[z++] = A[y++];
            }

            S[(i+2)/2] = S[i+2];
        }
        
        if (k % 2) {
            for (int i = S[k-1]; i < S[k]; i++) {
                B[i] = A[i];
            }
            
            S[(k+1)/2] = S[k];
        }
        
        // switch buffers
        double *temp = A;
        A = B;
        B = temp;
        
        k = (k+1)/2;
    }
}
    
int main(int argc, char **argv) {
    FILE *fS;					// binary file for list sizes
    FILE *fE;					// binary file for list elements
    FILE *fR;					// binary file for merged list

    int k;						// number of lists
    int n;	 					// number of elements
    int p = SUBLIST_COUNT;		// number of sublists per list

    int 	*hst_S = NULL;		// starting positions of lists
    double 	*hst_E = NULL;		// elements of lists
    double  *hst_W = NULL;		// working buffer

    double *dev_E  = NULL;		// elements of lists
    double *dev_W0 = NULL;		// working buffer
    double *dev_W1 = NULL;		// working buffer
    
    int	*dev_splitters = NULL;	// splitters

    double gpu_time1 = 0.0;		// time to generate splitters
    double gpu_time2 = 0.0;		// time to merge sublists
    double cpu_time  = 0.0;		// time to merge lists
    
    curandState	*state;			// needed for generating random numbers
    
    if (argc != 5) {                                                                 
        fprintf(stderr, "Usage: %s k sizes.dat elements.dat result.dat\n", argv[0]); 
        exit(1);
    }
    
    if ((k = atoi(argv[1])) < 2 || k > PARTITION_BLOCK_SIZE) {
        fprintf(stderr, "Error: k must be between 2 and %d\n", PARTITION_BLOCK_SIZE);
        exit(1);
    }

    host_alloc(hst_S, int, k+1);
    hst_S[0] = 0;

    open_file(fS, argv[2], "rb");
    read_file(hst_S+1, sizeof(int), k, fS);
        
    for (int i = 0; i < k; ++i) {
        hst_S[i+1] += hst_S[i];
    }

    n = hst_S[k];

    host_alloc(hst_E, double, n);		
    host_alloc(hst_W, double, n);

    open_file(fE, argv[3], "rb");
    open_file(fR, argv[4], "wb");
    
    read_file(hst_E, sizeof(double), n, fE);
    
    cuda_exec(cudaMalloc(&state, sizeof(curandState)));

    cuda_exec(cudaMalloc(&dev_E,  n * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_W0, n * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_W1, n * sizeof(double)));
    
    cuda_exec(cudaMalloc(&dev_splitters, k * (p+1) * sizeof(int))); 

    cuda_exec(cudaMemcpy(dev_E, hst_E, n * sizeof(double), cudaMemcpyHostToDevice));
    
    // initialize starting and ending splitters
    cuda_exec(cudaMemcpy(dev_splitters, 	hst_S,   k * sizeof(int), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_splitters+k*p, hst_S+1, k * sizeof(int), cudaMemcpyHostToDevice));

    cpu_time -= timer();
    
    // merge the lists on CPU
    cpu_merge(hst_E, k, hst_S, hst_W);
    
    cpu_time += timer();
    
    setup_kernel<<<1,1>>>(state, timer());
    cudaDeviceSynchronize();
    
    gpu_time1 -= timer();
    
    // we need p-1 splitters to partition a list into p parts
    for (int i = 0; i < p-1; i++) {			
        int spl_size = n/p + (i < (n%p) ? 1 : 0);
        generate_splitters<<<1, PARTITION_BLOCK_SIZE>>>(dev_E, k, spl_size, dev_splitters + i*k, dev_splitters + k*p, state);
        cudaDeviceSynchronize();
    }

    gpu_time1 += timer();

    gpu_time2 -= timer();
    
    gpu_merge<<<p, MERGE_BLOCK_SIZE>>>(dev_E, k, dev_splitters, dev_W0, dev_W1, n);
    cudaDeviceSynchronize();
    
    gpu_time2 += timer();
    
    // find the results from CPU and GPU
    double 	*hst_C;		// result from cpu, merged lists
    double 	*hst_G;		// result from gpu, merged lists 
    
    int buffer_switch_count = 0;
    
    while (k > 1) {
        k = (k+1)/2;
        ++buffer_switch_count;
    }

    if (buffer_switch_count % 2) {
        hst_C = hst_W;
        hst_G = hst_E;
        cuda_exec(cudaMemcpy(hst_G, dev_W0, n * sizeof(double), cudaMemcpyDeviceToHost));
    }	
    else {
        hst_C = hst_E;
        hst_G = hst_W;
        cuda_exec(cudaMemcpy(hst_G, dev_W1, n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // compare CPU and GPU results
    for (int i = 0; i < n; i++) {
        if (hst_C[i] != hst_G[i]) {
            printf("CPU and GPU results don't match!\n");
            break;
        } 
        else if (i == n-1) {
            printf("CPU and GPU results match!\n");
            write_file(hst_C, sizeof(double), n, fR);
        }		
    }
    
    printf("CPU time:  %5dms\n", (int) (1000 * cpu_time));
    printf("GPU time1: %5dms\n", (int) (1000 * gpu_time1));
    printf("GPU time2: %5dms\n", (int) (1000 * gpu_time2));

    free(hst_S);
    free(hst_E);
    free(hst_W);

    cudaFree(state);
    cudaFree(dev_E);
    cudaFree(dev_W0);
    cudaFree(dev_W1);
    
    cudaFree(dev_splitters);

    close_file(fS);
    close_file(fE);
    close_file(fR);

    return 0;
}
