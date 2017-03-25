#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  // 如果输入只有一个 bottom blob，那么 scale factor 就一个可学习的参数（标量），
  // 并且存储在 ScaleLayer 层中的 this->blobs_ 中
  // 接下来，我们可以通过判断 this->blobs_ 的向量长度
  // 是否为 0，来判断 scale factor 是否被初始化 
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) { // 如果 scale factor 没有初始化
    // scale is a learned parameter; initialize it
    this->axis_ = bottom[0]->CanonicalAxisIndex(param.axis()); // 将带有负数的轴转化成正数
    const int num_axes = param.num_axes();
    // num_axis 必须大于 -1，其中 -1 表示扩展 bottom[0] 的所有维度
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      // 判断 scale factor 扩展为 blob 的维数不超过 bottom[0]
      CHECK_GE(bottom[0]->num_axes(), this->axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << this->axis_;
    }
    this->blobs_.resize(1); // 我们将参数的个数设置为 1
    // 获取可扩展 shape 的 start axis 的迭代器
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + this->axis_;
    // 获取可扩展 shape 的 end axis 的迭代器
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    // 获取可扩展的 shape 维度
    vector<int> scale_shape(shape_start, shape_end);
    // 为可学习参数 scale factor 分配空间
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
    FillerParameter filler_param(param.filler()); // 填充 scale factor
    // 如果没有 filler 填充器，那么我们就填充全 1
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    // 获取 filler 指针
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    // 为可学习参数的 scale factor 填充初始化数据
    filler->Fill(this->blobs_[0].get());
  }

  // 如果有 bias 项
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    this->bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    this->bias_bottom_vec_.resize(1);
    this->bias_bottom_vec_[0] = bottom[0];
    this->bias_layer_->SetUp(this->bias_bottom_vec_, top);
    if (this->blobs_.size() + bottom.size() < 3) {
      // case: blobs.size == 1 && bottom.size == 1
      // or blobs.size == 0 && bottom.size == 2
      this->bias_param_id_ = this->blobs_.size();
      this->blobs_.resize(this->bias_param_id_ + 1);
      this->blobs_[this->bias_param_id_] = this->bias_layer_->blobs()[0];
    } else {
      // bias param already initialized
      this->bias_param_id_ = this->blobs_.size() - 1;
      this->bias_layer_->blobs()[0] = this->blobs_[this->bias_param_id_];
    }
    this->bias_propagate_down_.resize(1, false);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  this->axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  // 判断 bottom[0] 的 axis 数目是否大于或等于 this->axis_ + scale->num_axes()
  CHECK_GE(bottom[0]->num_axes(), this->axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;

  for (int i = 0; i < scale->num_axes(); ++i) {
    // bottom[0] 的尺寸是否与 scale 的尺寸一致
    CHECK_EQ(bottom[0]->shape(this->axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, this->axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] == top[0]) {  // in-place computation
    this->temp_.ReshapeLike(*bottom[0]);
  } else {
    top[0]->ReshapeLike(*bottom[0]);
  }
  this->sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  this->sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (this->sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
    caffe_set(sum_mult_size, Dtype(1), this->sum_multiplier_.mutable_cpu_data());
  }
  if (this->bias_layer_) {
    this->bias_bottom_vec_[0] = top[0];
    this->bias_layer_->Reshape(this->bias_bottom_vec_, top);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 获取只读 bottom data 的指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (bottom[0] == top[0]) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    // 如果我们进行的是 in-place 计算，我们需要在覆盖它之前，保存 bootom data
    // 这个操作只对反向传播有用。如果我们不进行反向传播，那么就可以跳过该操作。
    // 但是目前 Caffe 在进行前向传播的时候，并不知道我们是否需要反向传播
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
               this->temp_.mutable_cpu_data());
  }
  // 获取只读 scale_data 的指针
  const Dtype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  // 获取读写 top data 的指针
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      // 获取扩展因子
      const Dtype factor = scale_data[d];
      // 使用对应的扩展因子扩展 bottom_data，并且输出到 top_data
      caffe_cpu_scale(this->inner_dim_, factor, bottom_data, top_data);
      bottom_data += this->inner_dim_;
      top_data += this->inner_dim_;
    }
  }
  if (this->bias_layer_) {
    this->bias_layer_->Forward(this->bias_bottom_vec_, top);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    this->bias_layer_->Backward(top, this->bias_propagate_down_, this->bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    // 获取只读 top_diff 的指针 
    const Dtype* top_diff = top[0]->cpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->cpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Dtype* product = (is_eltwise ? scale->mutable_cpu_diff() :
        (in_place ? this->temp_.mutable_cpu_data() : bottom[0]->mutable_cpu_diff()));
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (this->sum_result_.count() == 1) {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result = caffe_cpu_dot(this->inner_dim_, product, sum_mult);
          *scale_diff += result;
        } else {
          *scale_diff = caffe_cpu_dot(this->inner_dim_, product, sum_mult);
        }
      } else {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_cpu_diff() : this->sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, this->sum_result_.count(), this->inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (this->outer_dim_ != 1) {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (this->scale_dim_ == 1) {
          if (scale_param) {
            Dtype result = caffe_cpu_dot(this->outer_dim_, sum_mult, sum_result);
            *scale_diff += result;
          } else {
            *scale_diff = caffe_cpu_dot(this->outer_dim_, sum_mult, sum_result);
          }
        } else {
          caffe_cpu_gemv(CblasTrans, this->outer_dim_, this->scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scale_data = scale->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < this->outer_dim_; ++n) {
      for (int d = 0; d < this->scale_dim_; ++d) {
        const Dtype factor = scale_data[d];
        caffe_cpu_scale(this->inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += this->inner_dim_;
        top_diff += this->inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleLayer);
#endif

INSTANTIATE_CLASS(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
