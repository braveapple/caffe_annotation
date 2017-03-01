#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp" // data_transformer文件中实现了常用的数据预处理操作，如尺度变换，减均值，镜像变换等
#include "caffe/internal_thread.hpp" //处理多线程的代码文件
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp" //线程队列的相关文件

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
// Layer的子类,data_layer的基类负责将Blobs数据送入网络
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  // 该虚函数实现了一般data_layer的功能，能够调用DataLayerSetUp来完成具体的data_layer的设置
  // 只能被BasePrefetchingDataLayer类来重载
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  // 数据层可以被其他的solver共享
  virtual inline bool ShareInParallel() const { return true; }
  // 层数据设置，具体要求的data_layer要重载这个函数来具体实现
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  // 数据层没有 bottom blobs，所以变形不是非常重要
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  // 虚函数由子类具体实现具体的 cpu 和 gpu 的后向传播
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_; // 在caffe.proto中定义的参数类
  // DataTransformer类的智能指针，DataTransformer类主要负责对数据进行预处理
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_; //是否有labels
};

// 定义 Batch 类，里面包含数据和标签
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

// 派生自类BaseDataLayer和类InternalThread
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  // 该虚函数实现了一般data_layer的功能，能够调用DataLayerSetUp来完成具体的data_layer的设置
  // 该函数不能被重载
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //具体的data_layer具体的实现这两个函数
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  // 通过这个函数执行线程函数
  virtual void InternalThreadEntry();
  // 加载batch
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  vector<shared_ptr<Batch<Dtype> > > prefetch_; // batch向量
  // 从 prefetch_free_ 队列取 batch，将该 batch 放到 prefetch_full_ 队列
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  // 从 prefetch_full_ 队列取 batch，将该 batch 输入网络计算，
  // 然后在 prefetch_full_ 中清空该 batch，最后将其放到 prefetch_free_ 队列
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Batch<Dtype>* prefetch_current_; // 当前所提取的 batch

  Blob<Dtype> transformed_data_; // 转换过的blob数据,中间变量用来辅助图像变换
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
