#include <algorithm>
#include <vector>

#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 获取只读 bottom_data 指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // 获取读写 bottom_diff 指针
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // 获取只读 label 指针（ground_truth）
  const Dtype* label = bottom[1]->cpu_data();
  // 获取输入的 batch_size
  int num = bottom[0]->num();
  // 获取 bottom_data 元素个数
  int count = bottom[0]->count();
  // 单个元素的维度
  int dim = count / num;

  // 将 bottom_data 的数据拷贝到 bottom_diff 中
  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    // label[i] 表示第 i 个样本的 ground_truth，范围是 [0, 1, 2, ..., K - 1]（K 分类问题）
    // 此处将第 i 个样本的 K 维预测值的第 label[i] 处的值乘以 -1
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      // 计算每个样本的每一类的损失值
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  
  // 获取读写 loss 的指针
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1: // 采用 L1 范数
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2: // 采用 L2 范数
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // ground_truth 是不能进行反向传播
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // 获取读写 bottom_diff 的指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 获取只读 label 的指针
    const Dtype* label = bottom[1]->cpu_data();
    // 获取 batch_size 个数
    int num = bottom[0]->num();
    // 获取 bottom_data 元素个数
    int count = bottom[0]->count();
     // 单个元素的维度
    int dim = count / num;
 
    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    // 获取 loss_weight 值
    const Dtype loss_weight = top[0]->cpu_diff()[0]; 
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      // caffe_cpu_sign 是符号函数
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
