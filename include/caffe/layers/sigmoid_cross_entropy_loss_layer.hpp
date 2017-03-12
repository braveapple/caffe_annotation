#ifndef CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * @brief Computes the cross-entropy (logistic) loss @f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n +
 *                  (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *        @f$, often used for predicting targets interpreted as probabilities.
 *
 * This layer is implemented rather than separate
 * SigmoidLayer + CrossEntropyLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SigmoidLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the scores @f$ x \in [-\infty, +\infty]@f$,
 *      which this layer maps to probability predictions
 *      @f$ \hat{p}_n = \sigma(x_n) \in [0, 1] @f$
 *      using the sigmoid function @f$ \sigma(.) @f$ (see SigmoidLayer).
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [0, 1] @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy loss: @f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n + (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *      @f$
 */
template <typename Dtype>
class SigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  // 显示构造函数，内嵌了 Sigmoid Layer
  explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  // 层初始化设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 返回该层的类型名称
  virtual inline const char* type() const { return "SigmoidCrossEntropyLoss"; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the sigmoid cross-entropy loss error gradient w.r.t. the
   *        predictions.
   *
   * Gradients cannot be computed with respect to the target inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as gradient computation with respect
   *      to the targets is not implemented.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$x@f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (\hat{p}_n - p_n)
   *      @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.

  // 读取 normalization 模式的参数，根据 blob 的尺寸计算归一化
  // 如果 normalization_mode 是 VALID，那么函数就会读取 VALID 个数据
  // 如果 normalization_mode 设置为 -1, 那么就会读取所有的数据
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  // 内嵌的 Sigmoid Layer 用将预测值映射为概率值
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;

  /// sigmoid_output stores the output of the SigmoidLayer.
  // 存储 Sigmoid Layer 的输出值
  shared_ptr<Blob<Dtype> > sigmoid_output_;

  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  // 用于存储所有的 bottom 的值，输入 SigmoildLayer::Forward 函数
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;

  /// top vector holder to call the underlying SigmoidLayer::Forward
  // 用于存储所有的 top 的值，是 SigmoidLayer::Forward 函数的输出
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  /// Whether to ignore instances with a certain label.
  // 是否要忽略有确切的 label 的实例
  bool has_ignore_label_;

  /// The label indicating that an instance should be ignored.
  // 该标签 label 表示这个 instance 需要被忽略
  int ignore_label_;

  /// How to normalize the loss.
  // 指明如何归一化损失值
  LossParameter_NormalizationMode normalization_;
  Dtype normalizer_;
  int outer_num_, inner_num_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
