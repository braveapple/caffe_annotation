#include "caffe/util/hdf5.hpp"

#include <string>
#include <vector>

namespace caffe {

/*
1. Signature:
herr_t H5LTget_dataset_ndims ( hid_t loc_id, const char *dset_name, int *rank )

2. Purpose:
Gets the dimensionality of a dataset.

3. Description:
H5LTget_dataset_ndims gets the dimensionality of a dataset named dset_name exists attached to the object loc_id.

4. Parameters:
hid_t loc_id            IN: Identifier of the object to locate the dataset within.
const char* dset_name   IN: The dataset name.
int* rank               OUT: The dimensionality of the dataset.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  // 验证 dataset 是否存在
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  // 验证 维数（轴数）是否在可接受的范围之内
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  // 验证 hdf5 文件的数据格式是否满足要求（float 或者 double）
  std::vector<hsize_t> dims(ndims); // 存储每一维的维度
  H5T_class_t class_;
  // 获取 hdf5 文件的数据库信息
  // vector.data() 返回指向vector第一块数据的指针
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  switch (class_) {
  case H5T_FLOAT:
    { LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_FLOAT"; }
    break;
  case H5T_INTEGER:
    { LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_INTEGER"; }
    break;
  case H5T_TIME:
    LOG(FATAL) << "Unsupported datatype class: H5T_TIME";
  case H5T_STRING:
    LOG(FATAL) << "Unsupported datatype class: H5T_STRING";
  case H5T_BITFIELD:
    LOG(FATAL) << "Unsupported datatype class: H5T_BITFIELD";
  case H5T_OPAQUE:
    LOG(FATAL) << "Unsupported datatype class: H5T_OPAQUE";
  case H5T_COMPOUND:
    LOG(FATAL) << "Unsupported datatype class: H5T_COMPOUND";
  case H5T_REFERENCE:
    LOG(FATAL) << "Unsupported datatype class: H5T_REFERENCE";
  case H5T_ENUM:
    LOG(FATAL) << "Unsupported datatype class: H5T_ENUM";
  case H5T_VLEN:
    LOG(FATAL) << "Unsupported datatype class: H5T_VLEN";
  case H5T_ARRAY:
    LOG(FATAL) << "Unsupported datatype class: H5T_ARRAY";
  default:
    LOG(FATAL) << "Datatype class unknown";
  }

  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims); // Reshape blob
}

/*
1. Signature:
herr_t H5LTread_dataset_float ( hid_t loc_id, const char *dset_name, float *buffer )

2. Purpose:
Reads a dataset from disk.

3. Description:
H5LTread_dataset reads a dataset named dset_name attached to the object specified by the identifier loc_id. 
The HDF5 datatype is H5T_NATIVE_FLOAT.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to read the dataset within.
const char* dset_name   IN: The name of the dataset to read.
double* buffer          OUT: Buffer with data.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 从 hdf5 文件中加载 float 数据
template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

/*
1. Signature:
herr_t H5LTread_dataset_double ( hid_t loc_id, const char *dset_name, double *buffer )

2. Purpose:
Reads a dataset from disk.

3. Description:
H5LTread_dataset reads a dataset named dset_name attached to the object specified by the identifier loc_id. 
The HDF5 datatype is H5T_NATIVE_DOUBLE.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to read the dataset within.
const char* dset_name   IN: The name of the dataset to read.
double* buffer          OUT: Buffer with data.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 从 hdf5 文件中加载 double 数据
template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

/*
1. Signature:
herr_t H5LTmake_dataset_float ( hid_t loc_id, const char *dset_name, int rank, 
const hsize_t *dims, const float *buffer )

2. Purpose:
Creates and writes a dataset.

3. Description:
H5LTmake_dataset creates and writes a dataset named dset_name attached to the object specified by 
the identifier loc_id. The dataset’s datatype will be native floating point, H5T_NATIVE_FLOAT.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to create the dataset within.
const char* dset_name   IN: The name of the dataset to create.
int rank                IN: Number of dimensions of dataspace.
const hsize_t* dims     IN: An array of the size of each dimension.
const float* buffer     IN: Buffer with data to be written to the dataset.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 将 float 数据写入到 hdf5 文件中
template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes(); // 获取数据 blob 的维数（轴数）
  hsize_t *dims = new hsize_t[num_axes];
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const float* data;
  if (write_diff) { // 判断是否要写 diff 部分还是 data 部分
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  // 向 hdf5 文件中写入数据
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), num_axes, dims, data);
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
  delete[] dims;
}

/*
1. Signature:
herr_t H5LTmake_dataset_double ( hid_t loc_id, const char *dset_name, int rank, 
const hsize_t *dims, const double *buffer )

2. Purpose:
Creates and writes a dataset.

3. Description:
H5LTmake_dataset creates and writes a dataset named dset_name attached to the object specified by 
the identifier loc_id. The dataset’s datatype will be native floating-point double, H5T_NATIVE_DOUBLE.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to create the dataset within.
const char* dset_name   IN: The name of the dataset to create.
int rank                IN: Number of dimensions of dataspace.
const hsize_t* dims     IN: An array of the size of each dimension.
const double* buffer    IN: Buffer with data to be written to the dataset.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 将 double 数据写入到 hdf5 文件
template <>
void hdf5_save_nd_dataset<double>(
    hid_t file_id, const string& dataset_name, const Blob<double>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes(); // 获取数据 blob 的维数（轴数）
  hsize_t *dims = new hsize_t[num_axes];
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const double* data; 
  if (write_diff) { // 判断是否要写 diff 部分还是 data 部分
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  // 将 double 数据写入到 hdf5 文件
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), num_axes, dims, data);
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
  delete[] dims;
}

/*
1. Signature:
herr_t H5LTread_dataset_string ( hid_t loc_id, const char *dset_name, char *buffer )

2. Purpose:
Reads a dataset from disk.

3. Description:
H5LTread_dataset_string reads a dataset named dset_name attached to the object specified by the identifier loc_id. The HDF5 datatype is H5T_C_S1.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to read the dataset within.
const char* dset_name   IN: The name of the dataset to read.
double* buffer          OUT: Buffer with data.

Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 从 hdf5 文件中读取 string
string hdf5_load_string(hid_t loc_id, const string& dataset_name) {
  // Get size of dataset
  size_t size;
  H5T_class_t class_;
  // 获取 dataset 的 size 和 class
  herr_t status = \
    H5LTget_dataset_info(loc_id, dataset_name.c_str(), NULL, &class_, &size);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name;
  char *buf = new char[size];
  // 将 dataset 读入 string 中
  status = H5LTread_dataset_string(loc_id, dataset_name.c_str(), buf);
  CHECK_GE(status, 0)
    << "Failed to load int dataset with name " << dataset_name;
  string val(buf);
  delete[] buf;
  return val;
}

/*
1. Signature:
herr_t H5LTmake_dataset_string ( hid_t loc_id, const char *dset_name, const char *buffer )

2. Purpose:
Creates and writes a dataset with string datatype.

3. Description:
H5LTmake_dataset_string creates and writes a dataset named dset_name attached to the object specified by 
the identifier loc_id. The dataset’s datatype will be C string, H5T_C_S1.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to create the dataset within.
const char* dset_name   IN: The name of the dataset to create.
const char* buffer      IN: Buffer with data to be written to the dataset.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 将 string 存入 hdf5 文件中
void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s) {
  herr_t status = \
    H5LTmake_dataset_string(loc_id, dataset_name.c_str(), s.c_str());
  CHECK_GE(status, 0)
    << "Failed to save string dataset with name " << dataset_name;
}

/*
1. Signature:
herr_t H5LTread_dataset_int ( hid_t loc_id, const char *dset_name, int *buffer )

2. Purpose:
Reads a dataset from disk.

3. Description:
H5LTread_dataset_int reads a dataset named dset_name attached to the object specified by the identifier loc_id. 
The HDF5 datatype is H5T_NATIVE_INT.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to read the dataset within.
const char *dset_name   IN: The name of the dataset to read.
int* buffer             OUT: Buffer with data.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 从 hdf5 文件中读取 string
int hdf5_load_int(hid_t loc_id, const string& dataset_name) {
  int val;
  herr_t status = H5LTread_dataset_int(loc_id, dataset_name.c_str(), &val);
  CHECK_GE(status, 0)
    << "Failed to load int dataset with name " << dataset_name;
  return val;
}

/*
1. Signature:
herr_t H5LTmake_dataset_int ( hid_t loc_id, const char *dset_name, int rank, 
const hsize_t *dims, const int *buffer )

2. Purpose:
Creates and writes a dataset. 

3. Description:
H5LTmake_dataset_int creates and writes a dataset named dset_name attached to the object 
specified by the identifier loc_id. The dataset’s datatype will be native signed integer, H5T_NATIVE_INT.

4. Parameters:
hid_t loc_id            IN: Identifier of the file or group to create the dataset within.
const char *dset_name   IN: The name of the dataset to create.
int rank                IN: Number of dimensions of dataspace.
const hsize_t * dims    IN: An array of the size of each dimension.
const int * buffer      IN: Buffer with data to be written to the dataset.

5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 将 string 存入 hdf5 文件中
void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i) {
  hsize_t one = 1;
  herr_t status = \
    H5LTmake_dataset_int(loc_id, dataset_name.c_str(), 1, &one, &i);
  CHECK_GE(status, 0)
    << "Failed to save int dataset with name " << dataset_name;
}

/*
1. Signature:
herr_t H5Gget_info( hid_t group_id, H5G_info_t *group_info )

2. Purpose:
Retrieves information about a group.

3. Description:
H5Gget_info retrieves information about the group specified by group_id. 
The information is returned in the group_info struct.
group_info is an H5G_info_t struct and is defined (in H5Gpublic.h) as follows:
      H5G_storage_type_t storage_type   Type of storage for links in group 
        H5G_STORAGE_TYPE_COMPACT: Compact storage 
        H5G_STORAGE_TYPE_DENSE: Indexed storage 
        H5G_STORAGE_TYPE_SYMBOL_TABLE: Symbol tables, the original HDF5 structure
      hsize_t nlinks  Number of links in group
      int64_t max_corder  Current maximum creation order value for group
      hbool_t mounted Whether the group has a file mounted on it

4. Parameters:
hid_t group_id  IN: Group identifier
H5G_info_t *group_info      OUT: Struct in which group information is returned
5. Returns:
Returns a non-negative value if successful; otherwise returns a negative value.
*/
// 获取 hdf5 文件的链接（类似于 Linux 的软连接和硬链接）
int hdf5_get_num_links(hid_t loc_id) {
  H5G_info_t info;
  herr_t status = H5Gget_info(loc_id, &info);
  CHECK_GE(status, 0) << "Error while counting HDF5 links.";
  return info.nlinks;
}

/* 
1. Signature:
ssize_t H5Lget_name_by_idx( hid_t loc_id, const char *group_name, H5_index_t index_field, 
H5_iter_order_t order, hsize_t n, char *name, size_t size, hid_t lapl_id )

2. Purpose:
Retrieves name of the nth link in a group, according to the order within a specified field or index.

3. Description:
H5Lget_name_by_idx retrieves the name of the nth link in a group, according to the specified order, order, 
within a specified field or index, index_field. If loc_id specifies the group in which the link resides, 
group_name can be a dot (.).

The size in bytes of name is specified in size. If size is unknown, it can be determined via an initial 
H5Lget_name_by_idx call with name set to NULL; the function's return value will be the size of the name.

4. Parameters:
hid_t loc_id                IN: File or group identifier specifying location of subject group
const char *group_name      IN: Name of subject group
H5_index_t index_field      IN: Index or field which determines the order
H5_iter_order_t order       IN: Order within field or index
hsize_t n                   IN: Link for which to retrieve information
char *name                  OUT: Buffer in which link value is returned
size_t size                 IN: Size in bytes of name
hid_t lapl_id               IN: Link access property list

5. Returns:
Returns the size of the link name if successful; otherwise returns a negative value.
*/
string hdf5_get_name_by_idx(hid_t loc_id, int idx) {
  ssize_t str_size = H5Lget_name_by_idx(
      loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
  CHECK_GE(str_size, 0) << "Error retrieving HDF5 dataset at index " << idx;
  char *c_str = new char[str_size+1];
  ssize_t status = H5Lget_name_by_idx(
      loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, c_str, str_size+1,
      H5P_DEFAULT);
  CHECK_GE(status, 0) << "Error retrieving HDF5 dataset at index " << idx;
  string result(c_str);
  delete[] c_str;
  return result;
}

}  // namespace caffe
