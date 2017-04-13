#include <algorithm>
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  // 参数 axis 和 slice_dim 至少出现一个，但是不能同时出现
  CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";
  this->slice_point_.clear();
  // 将参数分片位置保存在向量 slice_point_ 中
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(this->slice_point_));
}

template <typename Dtype>
void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  if (slice_param.has_slice_dim()) {
    this->slice_axis_ = static_cast<int>(slice_param.slice_dim());
    // Don't allow negative indexing for slice_dim, a uint32 -- almost
    // certainly unintended.
    // 参数 slice_dim 不能是负数
    CHECK_GE(this->slice_axis_, 0) << "casting slice_dim from uint32 to int32 "
        << "produced negative result; slice_dim must satisfy "
        << "0 <= slice_dim < " << kMaxBlobAxes;
    // 参数 slice_dim 不能超过 bottom blob 的轴数
    CHECK_LT(this->slice_axis_, num_axes) << "slice_dim out of range.";
  } else {
    this->slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
  }
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_slice_axis = bottom[0]->shape(this->slice_axis_);
  this->num_slices_ = bottom[0]->count(0, this->slice_axis_); // 计算每类分片的个数
  this->slice_size_ = bottom[0]->count(this->slice_axis_ + 1); // 计算单位分片的尺寸
  int count = 0;
  // 如果在参数中指明了参数 slice_point_
  if (this->slice_point_.size() != 0) {
    CHECK_EQ(this->slice_point_.size(), top.size() - 1);
    CHECK_LE(this->top.size(), bottom_slice_axis);
    int prev = 0;
    vector<int> slices; // 存储每个分片的分片长度
    for (int i = 0; i < this->slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(bottom_slice_axis - prev); // 处理最后一个分片
    // 对每一个 top blob 进行 reshape
    for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  } else { // 如果没有指明参数 slice_point_ ,默认为按 top blob 的个数均匀分割
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  }
  CHECK_EQ(count, bottom[0]->count());
  // 如果 top blob 的个数为 1，那么我们不需要进行分片
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 如果 top blob 的个数为 1，那么我们不需要进行分片
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  // 获取只读 bottom_data 的指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_slice_axis = bottom[0]->shape(this->slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    // 获取读写 top_data 的指针
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int top_slice_axis = top[i]->shape(this->slice_axis_);
    for (int n = 0; n < this->num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * this->slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * this->slice_size_;
      // 考虑到比 this->slice_axis_ 更低的维度，存储的空间是连续的
      caffe_copy(top_slice_axis * this->slice_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // 如果不需进行反向传播或者 top blob 的个数为 1，那么直接返回。
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0; // slice 维度的偏置
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(this->slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const int top_slice_axis = top[i]->shape(this->slice_axis_);
    for (int n = 0; n < this->num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * this->slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * this->slice_size_;
      caffe_copy(top_slice_axis * this->slice_size_,
          top_diff + top_offset, bottom_diff + bottom_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);
REGISTER_LAYER_CLASS(Slice);

}  // namespace caffe
