/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

// 该类主要实现注册类的功能，就是将类和对应的字符串类型放入一个map容器里面
template <typename Dtype>
class LayerRegistry {
 public:
  // 函数指针Creator，返回的是Layer<Dtype>类型的指针
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  // CreatorRegistry是类型字符串与对应的Creator的映射
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() { // 获取全局的类型字符串与对应的Creator的映射map容器
    static CreatorRegistry* g_registry_ = new CreatorRegistry(); // 在CPU内存中申请空间存储全局注册表
    return *g_registry_;
  }

  // Adds a creator.
  // 给定类型字符串与对应的Creator，将该键值对加入注册表中
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    // 判断类型字符串为type的Layer是否注册了
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  // 给定层的参数，创建层
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    // 从参数中获得类型字符串
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    // 判断类型字符串为type的Layer是否注册了
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    // 调用对应的Layer的Creator()函数
    return registry[type](param);
  }

  // 获取已注册的Layer的类型字符串列表
  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.

  // 禁止实例化，因为该类都是静态函数，所以是私有的
  LayerRegistry() {}

  // 获取所有已注册的Layer的字符串
  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


template <typename Dtype>
class LayerRegisterer {
 public:
  // 层注册器构造函数
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    // 调用层注册表中的AddCreator(type, creator)函数加入注册表
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

/* 注册自己定义的类，类名为type，  
 * 假设比如type=bias，那么生成如下的代码  
 * 下面的函数直接调用你自己的类的构造函数生成一个类的实例并返回  
 * CreatorbiasLayer(const LayerParameter& param)生成对象的指针
 * 下面的语句是为你自己的类定义了LayerRegisterer<float>类型的静态变量g_creator_f_biasLayer
 *（float类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）  
 * static LayerRegisterer<float> g_creator_f_biasLayer(bias, CreatorbiasLayer)  
 * 下面的语句为你自己的类定义了LayerRegisterer<double>类型的静态变量g_creator_d_biasLayer
 *（double类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）  
 * static LayerRegisterer<double> g_creator_d_biasLayer(bias, CreatorbiasLayer) 
*/

// REGISTER_LAYER_CREATOR 宏可以实现将特定Layer注册到全局注册表。
// 生成 g_creator_f_type(type, creator<float>) 和 g_creator_d_type(type, creator<float>) 这两个函数 
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
