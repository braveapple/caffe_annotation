#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

// 主要实现的是将上一层的数据归一化为均值为 0，方差为 1
// 在默认情况下，在训练过程中，网络通过计算整个训练数据集的均值和方差，然后将这两个
// 参数用于测试阶段输出确切的结果。你可以手动设置 "use_global_stats" 选项来决定是否
// 使用全局的统计特性。这些统计特性存储在三个 blob 中 (0) mean, (1) variance, 
// (2) moving average factor
// 注意的是， BatchNormLayer 只实现了归一化的阶段，要在该层后面跟上 ScaleLayer 实现 
// Scale 和 Shift (设置 bias_term: true) 
template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
 public:
  // 显示构造函数
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 层设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 层变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 返回层的类型名称
  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // 均值、方差、中间值、归一化值
  Blob<Dtype> mean_, variance_, temp_, x_norm_;
  // 如果为假，采用滑动平均对全局的均值和方差进行累加。如果为真，则使用全局的均值
  // 和方差，而不是使用每个 batch 的均值和方差
  bool use_global_stats_;
  // 滑动平均的衰减系数，默认为 0.999
  Dtype moving_average_fraction_;
  int channels_;
  // 分母的附加值，防止除 0 的情况，默认值为 1e-5
  Dtype eps_;

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype> batch_sum_multiplier_;
  Blob<Dtype> num_by_chans_;
  Blob<Dtype> spatial_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
