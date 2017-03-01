#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
// 类 BaseConvolutionLayer 继承于类 Layer
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  // 构造函数，实现之前先实现 Layer 的构造函数
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 层设置函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // 内存分配与数据变形函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // 返回最少的输入blob个数
  virtual inline int MinBottomBlobs() const { return 1; }
  // 返回最少的输出blob个数
  virtual inline int MinTopBlobs() const { return 1; }
  // 判断输入与输出blob个数是否一致
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  // CPU实现前向传播的卷积操作
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  // CPU实现前向传播的卷积操作后加上bias
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  // CPU实现后向传播求数据导数
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  // CPU实现后向传播求权重导数
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  // CPU实现后向传播求偏置导数
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  // GPU实现前向传播的卷积操作
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  // GPU实现前向传播的卷积操作后加上bias
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  // GPU实现后向传播求数据导数
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  // GPU实现后向传播求权重导数
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  // GPU实现后向传播求偏置导数
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  // 获取输入数据的空间维度的第 i 个 维度
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // 如果我们使用反卷积操作，reverse_dimensions() 函数返回 true，因此帮助卷积操作知道哪一维是哪一维
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  // 从其他参数计算出 height_out_ 和 width_out_
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_; // kernel的形状 = [kernel_h, kernel_w]
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_; // 步长形状 = [stride_h, stride_w] 
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_; // pad的形状 = [pad_h, pad_w]
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_; // 扩展参数 = [diation_h, diation_w]
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_; // 卷积的输入形状 = [输入图像通道数, 输入图像h, 输入图像w]
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_; // col_buffer的形状 = [kernel_dim_, conv_out_spatial_dim_ ]
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_; // 输出形状
  const vector<int>* bottom_shape_; // 输入形状

  int num_spatial_axes_; // 空间轴个数
  int bottom_dim_; // 输入度维度 = 输入图像通道数 * 输入图像的h * 输入图像w
  int top_dim_; // 输出维度 = 输出通道数 * 输出h * 输出w

  int channel_axis_; // 输入图像的第几个轴是通道
  int num_; // 输入图像的数目，即 batch size
  int channels_; // 输入图像的通道数
  int group_; // 卷积组的大小
  int out_spatial_dim_; // 输出空间维度 = 卷积之后的图像h * 卷积之后图像w
  int weight_offset_; // 使用卷积组用到的权重偏置
  int num_output_; // 卷积后的图像的通道数
  bool bias_term_; // 是否启用偏置
  bool is_1x1_; // 是不是1x1卷积
  bool force_nd_im2col_; // 是否强制使用N维通用卷积

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  // im2col 将图像拉成一个列向量
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      // 如果图像是二维的
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      // 如果图像是N维的（N > 2）
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  // col2im 将一个列向量变成一个图片
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      // 如果图像是二维的
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      // 如果图像是N维的（N > 2）
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }

#ifndef CPU_ONLY
  // im2col 将图像拉成一个列向量
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      // 如果图像是二维的
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      // 如果图像是N维的（N > 2）
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  // col2im 将一个列向量变成一个图片
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      // 如果图像是二维的
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      // 如果图像是N维的（N > 2）
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_; // num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_
  int num_kernels_col2im_; // num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_
  int conv_out_channels_; // 卷积的输出通道数，在参数配置文件中设置
  int conv_in_channels_; // 卷积的输入通道数，即输入图像的通道数
  int conv_out_spatial_dim_; // 卷积的输出的空间维度 = 卷积后图像h * 卷积后图像w
  int kernel_dim_; // 卷积核的维度 = 输入图像的维度 * 卷积核h * 卷积核w 
  int col_offset_; // 在使用gropu参数的时候使用的offset
  int output_offset_;

  Blob<Dtype> col_buffer_; // im2col的时候使用的存储空间
  Blob<Dtype> bias_multiplier_; // 将偏置扩展成矩阵
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
