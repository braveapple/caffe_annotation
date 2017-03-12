#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */

// 损失层有两个输入 blob，分别存储 prediction 和 ground-truth 标签，输出是一个单一的 blob 来表示 loss
// 损失层一般只对 prediction blob 进行反向传播 
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  // 显示定义构造函数
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  // 配置层的相关参数
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  // 变形函数
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // 输入一般为 2 个 blob
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  // 为了便于反向传播，建议 Net 设置自动为损失层分配一个 blob，其中该 blob 存储损失值
  // （即使用户没有在 prototxt 明确给出输出 blob）
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  // 通常我们不能对标签进行反向传播，所以我们不对标签进行前置反向传播
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
