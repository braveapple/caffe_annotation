#include <algorithm>
#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->sigmoid_bottom_vec_.clear();
  this->sigmoid_bottom_vec_.push_back(bottom[0]);
  this->sigmoid_top_vec_.clear();
  this->sigmoid_top_vec_.push_back(this->sigmoid_output_.get());
  this->sigmoid_layer_->SetUp(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);

  this->has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (this->has_ignore_label_) {
    this->ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) { 
    // 如果指明了 normalization，那么直接读取 normalization 的模式
    this->normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    // 如果没有指明 normalization，但是有 normalize 这项，
    // 那么我们可以选择 VALID 和 BATCH_SIZE 两者之一 
    this->normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    // 如果上述两者都没有指明，那么我们默认的 normalization 的模式是 BATCH_SIZE
    this->normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 调用父类的 Reshape 函数
  LossLayer<Dtype>::Reshape(bottom, top);
  // 获取 batch_size
  this->outer_num_ = bottom[0]->shape(0);  // batch size
  // 获取单个实例的尺寸大小
  this->inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  this->sigmoid_layer_->Reshape(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(this->outer_num_ * this->inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(this->outer_num_ * this->inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(this->outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  this->sigmoid_bottom_vec_[0] = bottom[0];
  this->sigmoid_layer_->Forward(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  // 获取只读 input_data 的指针
  const Dtype* input_data = bottom[0]->cpu_data();
  // 获取只读 target 的指针
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (this->has_ignore_label_ && this->target_value == this->ignore_label_) {
      continue;
    }
    // 将 x >= 0 和 x < 0 两个部分分开计算，这样使得 loss 值更加稳定
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    ++valid_count;
  }
  // 获取归一化系数
  this->normalizer_ = get_normalizer(this->normalization_, valid_count);
  // 归一化 loss
  top[0]->mutable_cpu_data()[0] = loss / this->normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // 我们认为 label 项是不能进行反向传播的
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    // 首先我们计算 diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = this->sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 实现 element-wise 减法： bottom_diff[i] = sigmoid_output_data[i] - target[i]
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Zero out gradient of ignored targets.
    // 如果忽略标签，那么 bottom_diff 设置为 0
    if (this->has_ignore_label_) {
      for (int i = 0; i < count; ++i) {
        const int target_value = static_cast<int>(target[i]);
        if (target_value == this->ignore_label_) {
          bottom_diff[i] = 0;
        }
      }
    }
    // Scale down gradient
    // 将 loss_weight 归一化
    Dtype loss_weight = top[0]->cpu_diff()[0] / this->normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
