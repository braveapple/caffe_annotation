#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  // 从 prototxt 中读取 top_k 参数
  this->top_k_ = this->layer_param_.accuracy_param().top_k();

  this->has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
  // 如果我们需要忽略某些 label
  if (this->has_ignore_label_) {
    // 从 prototxt 文件中读取 ignore_label 参数
    this->ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  // bottom[0]: predicted label (N, C, H, W)
  // bottom[1]: ground truth (N, 1, H, W)
  // top_k 必须要小于类别数
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";

  // 获取类别轴的索引值
  this->label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  // 获取 outer_num_ = N
  this->outer_num_ = bottom[0]->count(0, this->label_axis_);
  // 获取 inner_num_ = H * W
  this->inner_num_ = bottom[0]->count(this->label_axis_ + 1);
  // 判断 label 的个数是否与预测的数目一致
  CHECK_EQ(this->outer_num_ * this->inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    // 每类的准确率是一个向量
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(this->label_axis_);
    // 对 top[1] 进行 Reshape
    top[1]->Reshape(top_shape_per_class);
    // 对 num_buffer 进行 Reshape
    this->nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  // 获取只读的 bottom_data 的指针 (predicted label)
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // 获取只读的 bottom_label 的指针 (ground truth)
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // 获取 dim = C * H * W
  const int dim = bottom[0]->count() / this->outer_num_;
  // 获取 类别数 num_labels = C
  const int num_labels = bottom[0]->shape(this->label_axis_);
  vector<Dtype> maxval(this->top_k_ + 1);
  vector<int> max_id(this->top_k_ + 1);
  if (top.size() > 1) {
    caffe_set(this->nums_buffer_.count(), Dtype(0), this->nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      // 获取 label 的值
      const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
      // 如果我们需要忽略某些 label， 而且该 label 的值正好等于我们忽略的 label
      if (this->has_ignore_label_ && label_value == this->ignore_label_) {
        continue;
      }
      if (top.size() > 1) 
        ++nums_buffer_.mutable_cpu_data()[label_value]; // 增加该类别的个数
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(bottom_data[i * dim + k * inner_num_ + j], k));
      }
      // 对 bottom_data_vector 进行 top_k 次最小堆排序
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + this->top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      
      // check if true label is in top k predictions
      // 判断 ground truth 在 top_k 预测值中
      for (int k = 0; k < this->top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // 计算每类的准确率
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          this->nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / this->nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
