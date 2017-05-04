#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>



__global__ void array_set_kernel(float *output, float value, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = value;
  
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)arr->data;
  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  array_set_kernel<<<blocks, threads>>>(output_data, value, size);
  
  return 0;
}

__global__ void broadcast_kernel(const float *input, float *output, size_t in_size, size_t out_size){
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= in_size) return;
  for (size_t i=ind; i<out_size; i+=in_size) output[i] = input[ind];
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert (output->ndim - input->ndim == 1);
  size_t in_size = 1;
  size_t out_size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    in_size *= input->shape[i];
  }
  for (index_t i = 0; i < output->ndim; ++i) {
    out_size *= output->shape[i];
  }
  assert (out_size % in_size == 0);
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  if (in_size <= 1024) {
    threads.x = in_size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (in_size + 1023)/1024;;
  }
  broadcast_kernel<<<blocks, threads>>>(input_data, output_data, in_size, out_size);
  return 0;
}

__global__ void reduce_sum_axis_zero_kernel(const float *input, float *output, size_t in_size, size_t out_size){
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= out_size) return;
  output[ind] = 0;
  for (size_t i=ind; i<in_size; i+=out_size) output[ind] += input[i];
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert (input->ndim - output->ndim == 1);
  size_t in_size = 1;
  size_t out_size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    in_size *= input->shape[i];
  }
  for (index_t i = 0; i < output->ndim; ++i) {
    out_size *= output->shape[i];
  }
  assert (in_size % out_size == 0);
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  if (out_size <= 1024) {
    threads.x = out_size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (out_size + 1023)/1024;
  }
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(input_data, output_data, in_size, out_size);
  return 0;
}

__global__ void ele_add_kernel(const float *matA, const float *matB, float *output, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = matA[ind] + matB[ind];
  
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < matA->ndim; ++i) {
    size *= matA->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  ele_add_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data, size);
  
  return 0;
}

__global__ void add_const_kernel(const float *input, float *output, float value, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = input[ind] + value;
  
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;

  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  add_const_kernel<<<blocks, threads>>>(input_data, output_data, val, size);
  
  return 0;
}

__global__ void ele_mult_kernel(const float *matA, const float *matB, float *output, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = matA[ind] * matB[ind];
  
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < matA->ndim; ++i) {
    size *= matA->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  ele_mult_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data, size);
  return 0;
}

__global__ void mult_const_kernel(const float *input, float *output, float value, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = input[ind] * value;
  
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;

  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  mult_const_kernel<<<blocks, threads>>>(input_data, output_data, val, size);
  
  return 0;
}

__global__ void MatMulKernel(const float *A, const float *B, float *C, int rowA, int colA, int rowB, int colB, bool transA, bool transB) {
  float Cvalue = 0.0;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(!transA && !transB){
    assert(colA == rowB);
    if(r >= rowA || c >= colB) return;
    for (int e = 0; e < colA; ++e)
      Cvalue += (A[r * colA + e]) * (B[e * colB + c]);
    C[r * colB + c] = Cvalue;
  }else if(transA && !transB){
    if(r >= colA || c >= colB) return;
    for (int e = 0; e < rowA; ++e)
      Cvalue += (A[e * colA + r]) * (B[e * colB + c]);
    C[r * colB + c] = Cvalue;
  }else if(!transA && transB){
    if(r >= rowA || c >= rowB) return;
    for (int e = 0; e < colA; ++e)
      Cvalue += (A[r * colA + e]) * (B[c * colB + e]);
    C[r * rowB + c] = Cvalue;
  }else if(transA && transB){
    if(r >= colA || c >= rowB) return;
    for (int e = 0; e < rowA; ++e)
      Cvalue += (A[e * colA + r]) * (B[c * colB + e]);
    C[r * rowB + c] = Cvalue;
  }

}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  
  int rowA = matA->shape[0];
  int colA = matA->shape[1];
  int rowB = matB->shape[0];
  int colB = matB->shape[1];
  //int u = matC->shape[0];
  //int v = matC->shape[1];
  //cublasStatus_t stat; // CUBLAS functions status
  //cublasHandle_t handle;
  //stat = cublasCreate(&handle); 
  float *matC_data = (float *)matC->data;
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  //float al = 1.0f;
  //float bet = 0.0f;

  //int m = rowA;
  //int k = colA;
  //int n = colB;
  //int lda = k;
  //int ldb = n;
  //int ldc = n;
  //cublasOperation_t transA = CUBLAS_OP_N;
  //cublasOperation_t transB = CUBLAS_OP_N;
  //cublasDestroy(handle);
  const int BLOCK_SIZE = 16;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((std::max(rowB, colB) + dimBlock.x - 1) / dimBlock.x,
            (std::max(rowA, colA) + dimBlock.y - 1) / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(matA_data, matB_data, matC_data, rowA, colA, rowB, colB, transposeA, transposeB);
  return 0;
}




__global__ void relu_kernel(float *input, float *output, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  output[ind] = max(input[ind], 0.);
  
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  float *input_data = (float *)input->data;

  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  relu_kernel<<<blocks, threads>>>(input_data, output_data, size);
  
  return 0;
}

__global__ void relu_grad_kernel(const float *input, const float *in_grad, float *output, size_t size) {
  
  // Two dimensional thread blocks.
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size) {
    return;
  }
  float s = 0;
  if (input[ind] > 0) s = 1;
  if (input[ind] < 0) s = -1;
  output[ind] = (s + 1) * in_grad[ind] * 0.5;
  
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  size_t size = 1;
  for (index_t i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }
  dim3 blocks;
  dim3 threads;
  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  if (size <= 1024) {
    threads.x = size;
    blocks.x = 1;
  }else{
    threads.x = 1024;
    blocks.x = (size + 1023)/1024;
  }
  relu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data, output_data, size);
  return 0;
}

__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output) {
  
  // Two dimensional thread blocks.
  int y = threadIdx.y * blockDim.x + threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  output += y * ncol;
 
  float maxval = *input;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input[x] - maxval);
  }
  
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval) / sum;
  }
  
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);  
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;  
  float *output_data = (float *)output->data;

  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  softmax_kernel<<<1, threads>>>(
      nrow, ncol, input_data, output_data);
  
  return 0;
}

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
