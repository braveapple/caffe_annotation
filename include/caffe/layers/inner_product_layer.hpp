#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  // 显示构造函数，避免隐式转化
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // Layer 初始化函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Layer 变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // 获取 Layer 类型
  virtual inline const char* type() const { return "InnerProduct"; }
  // 获取输入 Blob 个数
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // 获取输出 Blob 个数
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  // CPU前向传播
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // GPU前向传播
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPU反向传播
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // GPU反向传播
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; // 样本个数
  int K_; // 单个样本的特征长度
  int N_; // 输出神经元个数
  bool bias_term_; // 标记是否有偏置项
  Blob<Dtype> bias_multiplier_; // 偏置项数乘因子
  // 标记权重矩阵是否已转置
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
