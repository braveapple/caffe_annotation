#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

// 如果机器是支持GPU的并且安装了cuda，通过cudaMallocHost分配的host memory将会被pinned，
// 这里我谷歌了一下，pinned的意思就是内存不会被paged out，我们知道内存里面是由页作为基本的管理单元。
// 分配的内存可以常驻在内存空间中对效率是有帮助的，空间不会被别的进程所抢占。同样如果内存越大，能被分配的Pinned内存自然也越大。
// 还有一点是，对于单一的GPU而言提升并不会太显著，但是对于多个GPU的并行而言可以显著提高稳定性。

// 全局函数，分配内存空间
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size)); // GPU模式下CUDA分配内存
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64); // 若使用MKL矩阵库，CPU模式下使用MKL的 mkl_malloc() 函数分配空间 
#else
  *ptr = malloc(size); // 若没有使用MKL矩阵库，CPU模式下使用最原始的 malloc() 函数分配空间
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// 全局函数，释放内存空间函数
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */

// 该类负责存储分配以及CPU和GPU之间的数据同步 
class SyncedMemory {
 public:
  // 构造函数，负责初始化
  SyncedMemory();
  // 显示构造函数，禁止隐式转换
  explicit SyncedMemory(size_t size);
  // 析构函数，内部调用 CaffeFreeHost() 函数
  ~SyncedMemory();
  const void* cpu_data(); // 获取只读 cpu data 的指针
  void set_cpu_data(void* data); // 设置 cpu data ，并且释放原始空间
  const void* gpu_data(); // 获取只读 gpu data 的指针
  void set_gpu_data(void* data); // 设置 gpu data ，并且释放原始空间
  void* mutable_cpu_data(); // 获取读写 cpu data 的指针
  void* mutable_gpu_data(); // 获取读写 gpu data 的指针
  // 状态机变量，表示四种状态：未初始化，CPU数据有效，GPU数据有效，已同步
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; } // 获取数据当前状态
  size_t size() { return size_; } // 获取数据当前存储空间大小

#ifndef CPU_ONLY
  // 这是一个cuda拷贝的异步传输函数，从数据从cpu拷贝到gpu，异步传输是已经假定caller会在使用之前做同步操作。
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu(); // 数据同步至CPU
  void to_gpu(); // 数据同步至GPU
  void* cpu_ptr_; // 位于CPU的数据指针
  void* gpu_ptr_; // 位于GPU的数据指针
  size_t size_; // 数据存储空间大小
  SyncedHead head_; // 数据当前状态
  bool own_cpu_data_; // 标志是否拥有CPU数据所有权（否，即从别的数据共享）
  bool cpu_malloc_use_cuda_; // 标志是否使用CUDA的内存分配和释放函数
  bool own_gpu_data_; // 标志是否拥有GPU数据所有权
  int device_; // GPU设备编号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory); // 禁止该类的拷贝与赋值
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
