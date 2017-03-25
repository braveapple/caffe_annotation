#ifndef CAFFE_SCALE_LAYER_HPP_
#define CAFFE_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

/**
 * @brief Computes the elementwise product of two input Blobs, with the shape of
 *        the latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product. Note: for efficiency and convenience, this layer can
 *        additionally perform a "broadcast" sum too when `bias_term: true`
 *        is set.
 *
 * The latter, scale input may be omitted, in which case it's learned as
 * parameter of the layer (as is the bias, if it is included).
 */

// 计算两个输入的 Blob 的 elementwise 内积，并将后面一个 Blob 的尺寸大小
// 扩展为前一个 Blob 的尺寸大小。使用相应的元素填充后面一个 Blob，然后再使用 
// elementwise 内积
// 注意：考虑到效率和方便，当设置 "bias_term: true"，该层也可以执行扩展求和

template <typename Dtype>
class ScaleLayer: public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit ScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 层设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 层变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 返回层的类型
  virtual inline const char* type() const { return "Scale"; }
  // Scale
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 定义 ScaleLayer 层内嵌的 BiasLayer 层
  shared_ptr<Layer<Dtype> > bias_layer_;
  // 定义 BiasLayer 层的输入的 bottom blob 向量
  vector<Blob<Dtype>*> bias_bottom_vec_;
  // 指定 BiasLayer 层是否需要反向传播
  vector<bool> bias_propagate_down_;
  // 指定 BiasLayer 的参数的索引值
  int bias_param_id_;

  Blob<Dtype> sum_multiplier_; // 乘积的结果
  Blob<Dtype> sum_result_; // 求和的结果
  Blob<Dtype> temp_; // 中间变量
  int axis_; // 轴的索引值
  int outer_dim_, scale_dim_, inner_dim_; // 用于存储输出、扩展、输入的维度
};


}  // namespace caffe

#endif  // CAFFE_SCALE_LAYER_HPP_
