#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

// 定义 sigmoid 函数
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 获取只读 bottom_data 指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // 获取读写 top_data 指针
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    // 将每一个元素输入到 sigmoid 函数中
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // 获取只读 top_data 的指针
    const Dtype* top_data = top[0]->cpu_data();
    // 获取只读 top_diff 的指针
    const Dtype* top_diff = top[0]->cpu_diff();
    // 获取读写 bottom_diff 的指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 获取 bottom[0] 的元素个数
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // top_data 存储了计算 sigmoid 函数之后的值
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
