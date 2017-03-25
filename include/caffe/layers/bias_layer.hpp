#ifndef CAFFE_BIAS_LAYER_HPP_
#define CAFFE_BIAS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes a sum of two input Blobs, with the shape of the latter Blob
 *        "broadcast" to match the shape of the former. Equivalent to tiling
 *        the latter Blob, then computing the elementwise sum.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer. Note: in case bias and scaling are desired, both operations can
 * be handled by `ScaleLayer` configured with `bias_term: true`.
 */

// 计算两个输入 blob 的和，并且要将 latter Blob 的尺寸扩展与 former Blob 的尺寸一致
// 均匀地扩展 latter Blob，然后计算 elementwise 和
// 注意：第二个输入 blob 可能会被忽略，如果它是该层的学习参数
template <typename Dtype>
class BiasLayer : public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit BiasLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 层设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 层变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Bias"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  Blob<Dtype> bias_multiplier_; 
  int outer_dim_, bias_dim_, inner_dim_, dim_;
};



}  // namespace caffe

#endif  // CAFFE_BIAS_LAYER_HPP_
