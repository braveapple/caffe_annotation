#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_); // 释放CPU内存空间
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_)); // 释放GPU显存空间
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED: // 如果未初始化
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); // 分配CPU内存空间
    caffe_memset(size_, 0, cpu_ptr_); // 初始化为全0
    head_ = HEAD_AT_CPU; // 设置状态为 HEAD_AT_CPU
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU: // 如果GPU数据有效
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); // 分配CPU内存空间
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_); // 将GPU显存中数据拷贝到CPU内存中
    head_ = SYNCED; // 设置状态为 SYNCED
#else
    NO_GPU;
#endif
    break;
  // 如果CPU数据有效或数据已同步，则不用做多余操作
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED: // 如果数据为初始化
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_)); // 分配GPU显存空间
    caffe_gpu_memset(size_, 0, gpu_ptr_); // 初始化为全0
    head_ = HEAD_AT_GPU; // 状态设置为 HEAD_AT_GPU
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU: // CPU数据有效
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_)); // 分配GPU显存空间
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_); // 将CPU内存数据拷贝至GPU显存中
    head_ = SYNCED; // 设置状态为 SYNCED
    break;
  // 如果GPU数据有效或数据已同步，则不用做多余操作
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

// 获取只读 cpu data 的指针
const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu(); // 将数据拷贝至CPU内存中
  return (const void*)cpu_ptr_;
}

// 设置 cpu data
void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) { // CPU对数据拥有所有权（非共享数据）
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_); // 释放CPU内存空间
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU; // 状态设为 HEAD_AT_CPU
  own_cpu_data_ = false; // 因为是共享数据，所以CPU数据所有权设为 false
}

// 获取只读 gpu data 的指针
const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu(); // 将数据拷贝至GPU显存中
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// 设置 gpu data
void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) { // 如果GPU对数据拥有所有权（非共享数据）
    CUDA_CHECK(cudaFree(gpu_ptr_)); // 释放GPU显存空间
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU; // 状态设置为 HEAD_AT_GPU
  own_gpu_data_ = false; // 因为是共享数据，所以GPU数据所有权设为 false
#else
  NO_GPU;
#endif
}

// 获取读写 cpu data 的指针
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu(); // 将数据拷贝CPU内存中
  head_ = HEAD_AT_CPU; // 状态设置为 HEAD_AT_CPU
  return cpu_ptr_;
}

// 获取读写 gpu data 的指针
void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu(); // 将数据拷贝至GPU显存中
  head_ = HEAD_AT_GPU; // 状态设置为 HEAD_AT_GPU
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
// 异步同步数据
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_)); // 分配GPU显存空间
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_); // 判断当前GPU设备号是否与 device_ 一致
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_); // 判断当前数据存储在的GPU设备号是否与 device_一致
  }
#endif
#endif
}

}  // namespace caffe

