#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
// 类PoolingLayer
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 层设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 返回类型
  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  // 因为最大值采样可以额外输出一个掩码的Blob, 所以要多返回一个blob
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_; // 池化核的高和宽
  int stride_h_, stride_w_; // 池化核的以高和宽方向的平移步宽
  int pad_h_, pad_w_; // 池化核的扩展的高和宽
  int channels_; // 输入通道数
  int height_, width_; // 输入Feature Map 尺寸
  int pooled_height_, pooled_width_; // 池化后的尺寸
  bool global_pooling_; // 是否全区域池化（将整幅图像降采样为1x1）
  Blob<Dtype> rand_idx_; // 随机采样点的索引
  Blob<int> max_idx_; // 最大采样点的索引
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
