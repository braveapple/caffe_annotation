#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// 层设置函数
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  // 获取卷积参数
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  //im2col,一般情况下 num_spatial_axes_ == 2,即将2维图像拉成向量，但 force_nd_im2col_ 针对的是更general的情况N维图像
  force_nd_im2col_ = conv_param.force_nd_im2col();
  // 输入图像的第几个轴是通道，对输入(N, C, H, W)，那么 axis() = 1，我们可以对输入(H, W)单独进行卷积操作 
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis()); 
  // (H, W)，即 axis() = 2 或 3 可以看成是 spatial_axis
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis; // spatial axis个数
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  //当num_spatial_axes_==2时，spatial_dim_blob_shape这个vector只包含一个元素且值为2 
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  // 以 spatial_dim_blob_shape 为参数来构造一个 Blob，即 kernel_shape_
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  // 如果采用二维卷积，且存在 kernel_h 和 kernel_w 参数
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else { // 不存在 kernel_h 和 kernel_w 参数
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      // 此时卷积核一个二维方阵
      // 如果模型描述文件中的卷积参数有 kernel_size : 4，那么卷积核维度是 4 * 4 
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  // 步长形状 = [stride_h, stride_w]
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  // 如果采用二维卷积，且存在 stride_h 和 stride_w 参数
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else { // 不存在 stride_h 和 stride_w 参数
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    // 如果在模型描述文件中没有stride，默认 stride = [1, 1]
    // 如果在模型描述文件中有 stride = 2，那么 stride = [2, 2]
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

  // Setup pad dimensions (pad_).
  // pad的形状 = [pad_h, pad_w]
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  // 如果采用二维卷积，且存在 pad_h 和 pad_w 参数
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else { // 不存在 pad_h 和 pad_w 参数
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    // 如果在模型描述文件中没有pad，默认 pad = [1, 1]
    // 如果在模型描述文件中有pad = 2，那么 pad = [2, 2]
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  // 扩展参数 = [diation_h, diation_w]
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  // 如果采用二维卷积，且存在 dilation_h 和 dilation_w 参数
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  // 如果在模型描述文件中没有dilation，默认 dilation = [1, 1]
  // 如果在模型描述文件中有dilation = 2，那么 dilation = [2, 2]
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  // 判断是否是 1x1 卷积 （stride = 1 且 pad = 0）
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_); // 输入图像的通道个数
  num_output_ = this->layer_param_.convolution_param().num_output(); // 卷积后图像的通道个数
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group(); // 卷积组的大小
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) { // 判断是否要反转输入输出维度
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_; // 卷积输出通道数
  weight_shape[1] = conv_in_channels_ / group_; // 卷积输入通道数
  // 最后 weight_shape = [conv_out_channels_, conv_in_channels_ / group_, H, W]
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term(); // 是否启用偏置
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    // 判断权重和偏置项对应的 blob 个数是否与 blobs_ 个数一致
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    // 如果 weight_shape 与 权重blob的维度 不一致
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    // 如果存在偏置项且 bias_shape 与 偏置维度不一致
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2); // 存在偏置项，将 blobs_ 长度重置为 2
    } else {
      this->blobs_.resize(1); // 不存在偏置项，将 blobs_ 长度重置为 1
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    // blobs_[0]的维度信息是四个维度，count_ 为四个维度的值相乘
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // 获取卷积核指针
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    // 初始化卷积核
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    // 初始化偏置项
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // 获取卷积核维度，输入通道数 * 卷积核高度 * 卷积核宽度
  kernel_dim_ = this->blobs_[0]->count(1);
  // 使用卷积组用到的权重偏置
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  // 卷积层需要后向传播
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

// 卷积层应该默认有多少个bottom 就有多少个top输出 
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_); // 输入图像数目，即 batch size
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  // top_shape 的维度为 [N, num_output_, H, W]
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  // reshape 每一个 top blob
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) { // 如果需要反转维度
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else { // 如果不需要反转维度
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  // group分组，对 conv_in_channels 分组; 卷积窗口在输入“图像”上按步长滑动，
  //（可以想象）形成了多个子图;然后将所有子图拉成一列，列的长度就是 col_offset_
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  // 卷积的输入形状 = [输入图像通道数, 输入图像h, 输入图像w]
  conv_input_shape_.Reshape(bottom_dim_blob_shape); 
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) { // 如果需要反转维度 
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else { // 如果不需要反转维度
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.

  // 一般情况下，col_buffer_ 的维度信息为三个维度。
  // col_buffer_shape_ 的存储的元素为：kernel_dim_ * group_， 
  // 输出特征图的H， 输出特征图的W。
  // 可以认为 col_buffer_ 内所存储的数据的维度为：(kernel_dim_ * group_) × H × W，
  // 且与 kernel_dim_ * conv_out_spatial_dim_ 有密切关系.

  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) { // 如果需要反转维度
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else { // 如果不需要反转维度
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  // 设置偏置项
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

/*
* 只是对一张图像进行前向传播！
* 与全连接层类比，conv_out_channels_ / group_ 相当与全连接层的输出神经元个数;
* conv_out_spatial_dim_ 相当于全连接层中的样本个数;
* kernel_dim_ 相当与全连接层中每个样本特征向量的维数。 
*/

// 实现前向传播卷积操作
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      // 如果没有1x1卷积，也没有 skip_im2col    
      // 则使用 conv_im2col_cpu 对使用卷积核滑动过程中的每一个 kernel 大小的图像块    
      // 变成一个列向量，形成一个 height = kernel_dim_ 的    
      // width = 卷积后图像height * 卷积后图像width   
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }

  // 使用caffe的 cpu_gemm 来进行计算  
  // 假设输入是20个feature map，输出是10个feature map，group_= 2
  // 那么我们就会把这个训练网络分解成两个 10 -> 5 的网络，由于两个网络结构是
  // 一模一样的，那么就可以利用多个GPU完成训练加快训练速度
  for (int g = 0; g < group_; ++g) {
    // weights <--- blobs_[0]->cpu_data()。类比全连接层，
    // weights 为权重，col_buff 相当与数据，矩阵相乘 weights × col_buff. 
    // 其中，weights的维度为 (conv_out_channels_ /group_) x kernel_dim_，
    // col_buff的维度为 kernel_dim_ x conv_out_spatial_dim_， 
    // output的维度为 (conv_out_channels_ /group_) x conv_out_spatial_dim_.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

// 前向传播卷积后加bias 
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  // output = bias * bias_multiplier_    
  // num_output 与 conv_out_channel是一样的    
  // num_output_ * out_spatial_dim_ = num_output_ * 1 * 1 * out_spatial_dim_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

// 反向传播，计算关于bottom data的导数以便传给下一层
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    // kernel_dim_ = conv_in_channels_ * kernel_h * kernel_w
    // conv_out_spatial_dim_ = H_out * W_out
    // weights 是 conv_out_channels_ * kernel_dim_
    // output 是 conv_out_channels_ * conv_out_spatial_dim_
    // col_buff 是输入feature map的转换矩阵，kernel_dim_ * conv_out_spatial_dim_
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    // 转换为输入feature map
    conv_col2im_cpu(col_buff, input);
  }
}

// 反向传播计算关于权重的导数用于更新权重
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

// 反向传播计算关于bias的导数
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

// 实现前向传播卷积操作
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

// 前向传播卷积后加bias 
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

// 反向传播，计算关于bottom data的导数以便传给下一层
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

// 反向传播计算关于权重的导数用于更新权重
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

// 反向传播计算关于bias的导数
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
