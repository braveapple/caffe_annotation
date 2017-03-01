/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); // 打开 hdf5 文件
  if (file_id < 0) { // 返回值是负数，那么打开失败
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  // 设置数据的最小和最大维度
  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  // 设置每一个 hdf5_blobs_ 数据
  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  // 保证 hdf_blobs_ 至少有一个轴
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0); // hdf_blobs_ 中包含图片的个数
  for (int i = 1; i < top_size; ++i) { // 检查 data 和 label 的hdf_blobs中是否有相同的图片数量
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  // 默认的 data_permutation_ 是按照自然顺序排列的
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  // 是否需要 shuffle
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  // hdf5数据层不需要进行数据预处理
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  // 从 txt 文件路径来提取 hdf5 文件
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line); // 读取 txt 文件的每一行来提取 hdf5 文件路径
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size(); // 获取 hdf5 文件数目
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  // 默认 file_permutation_ 是按照自然顺序排列的
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  // 是否要对 file_permutation_ 进行 shuffle
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  // 加载第一个 hdf5 文件并且初始化行数累加器
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size(); // 获取 batch_size
  const int top_size = this->layer_param_.top_size(); // 获取 top_size
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_[i]->num_axes()); // 初始化 top_shape 的 size
    top_shape[0] = batch_size; // top_shape[0] = 一个 batch 中的图片数目
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape); // Reshape top blob
  }
}

template <typename Dtype>
bool HDF5DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count(); // 获取 solver 的数目
  int rank = Caffe::solver_rank(); // 获取 solver rank
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void HDF5DataLayer<Dtype>::Next() {
  if (++current_row_ == hdf_blobs_[0]->shape(0)) { // 如果该 hdf5 文件已经读完
    if (num_files_ > 1) {
      ++current_file_; // 读取下一个 hdf5 文件
      if (current_file_ == num_files_) { // 如果已经读完最后一个 hdf5 文件
        current_file_ = 0; // 读取第一文件
        // 对 file_permutation 进行 shuffle
        if (this->layer_param_.hdf5_data_param().shuffle()) {
          std::random_shuffle(file_permutation_.begin(),
                              file_permutation_.end());
        }
        DLOG(INFO) << "Looping around to first file.";
      }
      LoadHDF5FileData(
        hdf_filenames_[file_permutation_[current_file_]].c_str());  // 加载 current_file_ 
    }
    current_row_ = 0; // 从第一行开始读取数据
    // 对 data_permutation_ 进行 shuffle
    if (this->layer_param_.hdf5_data_param().shuffle())
      std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
  }
  offset_++; // 自增偏置
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i) { // 将一个 batch 中的所有图片加载到 top blob 中
    while (Skip()) {
      Next();
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0); // 计算每一张图片的 channel * height * width 像素点个数
      // 将一张图片拷贝至 top blob 中
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
    Next();
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
