#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */

// 这里提供一个公共接口，即输入一个 blob，然后输出一个相同尺寸的 blob
// 输出元素只依赖于对应输入元素 
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  // 变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 输入与输出的 blob 数目都是 1
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
