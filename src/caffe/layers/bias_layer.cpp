#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 如果输入只有一个 bottom blob，那么 bias 就一个可学习的参数（标量），
  // 并且存储在 BiasLayer 层中的 this->blobs_ 中
  // 接下来，我们可以通过判断 this->blobs_ 的向量长度
  // 是否为 0，来判断 bias 是否被初始化 
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    // 获取 Bias 参数
    const BiasParameter& param = this->layer_param_.bias_param();
    // 将负数的索引值转换为正数索引值
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    // 我们感兴趣的 axes 数目
    const int num_axes = param.num_axes();
    // num_axes 大于等于 -1，其中 -1 代表所有的 bottom[0] 的轴
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      // bias blob 扩展维数不能超过 bottom[0] 的维数
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }

    this->blobs_.resize(1); // 将 bias 参数个数设置为 1
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end); // 设置 bias 的 shape
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape)); // 为 this->blobs_ 分配空间 
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.filler())); // 获取指向 filler 的指针
    filler->Fill(this->blobs_[0].get()); // 填充 bias 参数
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true); // bias参数需要进行反向传播
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BiasParameter& param = this->layer_param_.bias_param();
  // 如果输入 bottom 的个数大于 1，那么 bias 就是 bottom[1]
  // 否则 bias 就是可学习参数 this->blobs_[0]
  Blob<Dtype>* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis == 0 in special case where bias is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis == 0 and (therefore) outer_dim_ == 1.
  // 如果 bias 是一个可学习的参数（标量），那么 axis = 0
  // 否则 axis 等于 param.axis()
  const int axis = (bias->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  // bias blob 扩展维数不能超过 bottom[0] 的维数
  CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
      << "bias blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis;
  // 判断 bias blob 维度是否与 bottom[0] 对应的维度一致
  for (int i = 0; i < bias->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis + i
        << ") and bias->shape(" << i << ")";
  }

  this->outer_dim_ = bottom[0]->count(0, axis); // 获取 outer 维度
  this->bias_dim_ = bias->count(); // 获取 bias 维度
  this->inner_dim_ = bottom[0]->count(axis + bias->num_axes()); // 获取 inner 维度
  this->dim_ = this->bias_dim_ * this->inner_dim_;
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
  if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
    caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  for (int n = 0; n < outer_dim_; ++n) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Dtype(1), bias_data,
        bias_multiplier_.cpu_data(), Dtype(1), top_data);
    top_data += dim_;
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_cpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasLayer);
#endif

INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe
