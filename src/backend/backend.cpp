#include "backend.h"
#include "parsing_utils.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#define CPU torch::kCPU
#define CUDA torch::kCUDA
#define MPS torch::kMPS

Backend::Backend() : m_loaded(0), m_device(CPU), m_use_gpu(false) {
  at::init_num_threads();
}

//Gestion des accès 
std::unordered_map<std::string, std::shared_ptr<std::mutex>> Backend::g_file_mutexes;
std::mutex Backend::g_file_mutexes_map_mutex;


void Backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method, int n_batches) {
  c10::InferenceMode guard;

  auto params = get_method_params(method);
  // std::cout << "in_buffer length : " << in_buffer.size() << std::endl;
  // std::cout << "out_buffer length : " << out_buffer.size() << std::endl;

  if (!params.size())
    return;

  auto in_dim = params[0];
  auto in_ratio = params[1];
  auto out_dim = params[2];
  auto out_ratio = params[3];

  if (!m_loaded)
    return;

  // COPY BUFFER INTO A TENSOR
  std::vector<at::Tensor> tensor_in;
  // for (auto buf : in_buffer)
  for (int i(0); i < in_buffer.size(); i++) {
    tensor_in.push_back(torch::from_blob(in_buffer[i], {1, 1, n_vec}));
    // std::cout << i << " : " << tensor_in[i].min().item<float>() << std::endl;
  }

  auto cat_tensor_in = torch::cat(tensor_in, 1);
  cat_tensor_in = cat_tensor_in.reshape({in_dim, n_batches, -1, in_ratio});
  cat_tensor_in = cat_tensor_in.select(-1, -1);
  cat_tensor_in = cat_tensor_in.permute({1, 0, 2});
  // std::cout << cat_tensor_in.size(0) << ";" << cat_tensor_in.size(1) << ";" << cat_tensor_in.size(2) << std::endl;
  // for (int i = 0; i < cat_tensor_in.size(1); i++ )
    // std::cout << cat_tensor_in[0][i][0] << ";";
  // std::cout << std::endl;

  // SEND TENSOR TO DEVICE
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  cat_tensor_in = cat_tensor_in.to(m_device);
  std::vector<torch::jit::IValue> inputs = {cat_tensor_in};

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method(method)(inputs).toTensor();
    tensor_out = tensor_out.repeat_interleave(out_ratio).reshape(
        {n_batches, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return;
  }
  model_lock.unlock();

  int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
      out_n_vec(tensor_out.size(2));

  // for (int b(0); b < out_batches; b++) {
  //   for (int c(0); c < out_channels; c++) {
  //     std::cout << b << ";" << c << ";" << tensor_out[b][c].min().item<float>() << std::endl;
  //   }
  // }

  // CHECKS ON TENSOR SHAPE
  if (out_batches * out_channels != out_buffer.size()) {
    std::cout << "bad out_buffer size, expected " << out_batches * out_channels
              << " buffers, got " << out_buffer.size() << "!\n";
    return;
  }

  if (out_n_vec != n_vec) {
    std::cout << "model output size is not consistent, expected " << n_vec
              << " samples, got " << out_n_vec << "!\n";
    return;
  }

  tensor_out = tensor_out.to(CPU);
  tensor_out = tensor_out.reshape({out_batches * out_channels, -1});
  auto out_ptr = tensor_out.contiguous().data_ptr<float>();

  for (int i(0); i < out_buffer.size(); i++) {
    memcpy(out_buffer[i], out_ptr + i * n_vec, n_vec * sizeof(float));
  }
}

int Backend::load(std::string path) {
  try {
    std::shared_ptr<std::mutex> file_mutex;
    {
        std::lock_guard<std::mutex> map_lock(g_file_mutexes_map_mutex);
        auto it = g_file_mutexes.find(path);
        if (it == g_file_mutexes.end()) {
            file_mutex = std::make_shared<std::mutex>();
            g_file_mutexes[path] = file_mutex;
        } else {
            file_mutex = it->second;
        }
    }
    std::lock_guard<std::mutex> file_lock(*file_mutex);

    auto model = torch::jit::load(path);

    model.eval();
    model.to(m_device);

    {
      std::lock_guard<std::mutex> lock(m_model_mutex);
      m_model = std::move(model);
      m_available_methods = get_available_methods();
      m_path = path;
      m_loaded = 1;
    }
    return 0;

  } catch (const std::exception& e) {
        std::cerr << "Exception dans `Backend::load()` : " << e.what() << std::endl;
        return 1;
  } catch (...) {
        std::cerr << "Exception inconnue dans `Backend::load()` !" << std::endl;
        return 1;
  } 

}

int Backend::reload() {
  auto return_code = load(m_path);
  return return_code;
}

bool Backend::has_method(std::string method_name) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &m : m_model.get_methods()) {
    if (m.name() == method_name)
      return true;
  }
  return false;
}

bool Backend::has_settable_attribute(std::string attribute) {
  for (const auto &a : get_settable_attributes()) {
    if (a == attribute)
      return true;
  }
  return false;
}

std::vector<std::string> Backend::get_available_methods() {
  std::vector<std::string> methods;
  try {
    std::vector<c10::IValue> dumb_input = {};
    auto methods_from_model =
        m_model.get_method("get_methods")(dumb_input).toList();

    for (int i = 0; i < methods_from_model.size(); i++) {
      methods.push_back(methods_from_model.get(i).toStringRef());
    }
  } catch (...) {
    for (const auto &m : m_model.get_methods()) {
      try {
        auto method_params = m_model.attr(m.name() + "_params");
        methods.push_back(m.name());
      } catch (...) {
      }
    }
  }
  return methods;
}

std::vector<std::string> Backend::get_available_attributes() {
  std::vector<std::string> attributes;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &attribute : m_model.named_attributes())
    attributes.push_back(attribute.name);
  return attributes;
}

std::vector<std::string> Backend::get_settable_attributes() {
  std::vector<std::string> attributes;
  try {
    std::vector<c10::IValue> dumb_input = {};
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto methods_from_model =
        m_model.get_method("get_attributes")(dumb_input).toList();
    model_lock.unlock();
    for (int i = 0; i < methods_from_model.size(); i++) {
      attributes.push_back(methods_from_model.get(i).toStringRef());
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    for (const auto &a : m_model.named_attributes()) {
      try {
        auto method_params = m_model.attr(a.name + "_params");
        attributes.push_back(a.name);
      } catch (...) {
      }
    }
    model_lock.unlock();
  }
  return attributes;
}

std::vector<c10::IValue> Backend::get_attribute(std::string attribute_name) {
  std::string attribute_getter_name = "get_" + attribute_name;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto attribute_getter = m_model.get_method(attribute_getter_name);
    model_lock.unlock();
  } catch (...) {
    throw "getter for attribute " + attribute_name + " not found in model";
  }
  std::vector<c10::IValue> getter_inputs = {}, attributes;
  try {
    try {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      attributes = m_model.get_method(attribute_getter_name)(getter_inputs)
                       .toList()
                       .vec();
      model_lock.unlock();
    } catch (...) {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      auto output_tuple =
          m_model.get_method(attribute_getter_name)(getter_inputs).toTuple();
      attributes = (*output_tuple.get()).elements();
      model_lock.unlock();
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    attributes.push_back(
        m_model.get_method(attribute_getter_name)(getter_inputs));
    model_lock.unlock();
  }
  return attributes;
}

std::string Backend::get_attribute_as_string(std::string attribute_name) {
  std::vector<c10::IValue> getter_outputs = get_attribute(attribute_name);
  // finstringd arguments
  torch::Tensor setter_params;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
    model_lock.unlock();
  } catch (...) {
    throw "parameters to set attribute " + attribute_name +
        " not found in model";
  }
  std::string current_attr = "";
  for (int i = 0; i < setter_params.size(0); i++) {
    int current_id = setter_params[i].item().toInt();
    switch (current_id) {
    // bool case
    case 0: {
      current_attr += (getter_outputs[i].toBool()) ? "true" : "false";
      break;
    }
    // int case
    case 1: {
      current_attr += std::to_string(getter_outputs[i].toInt());
      break;
    }
    // float case
    case 2: {
      float result = getter_outputs[i].to<float>();
      current_attr += std::to_string(result);
      break;
    }
    // str case
    case 3: {
      current_attr += getter_outputs[i].toStringRef();
      break;
    }
    default: {
      throw "bad type id : " + std::to_string(current_id) + "at index " +
          std::to_string(i);
      break;
    }
    }
    if (i < setter_params.size(0) - 1)
      current_attr += " ";
  }
  return current_attr;
}

void Backend::set_attribute(std::string attribute_name,
                            std::vector<std::string> attribute_args) {
  // find setter
  std::cout << "set_attribute: entering the function" << std::endl;
  std::string attribute_setter_name = "set_" + attribute_name;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto attribute_setter = m_model.get_method(attribute_setter_name);
    model_lock.unlock();
  } catch (...) {
    throw "setter for attribute " + attribute_name + " not found in model";
  }
  // find arguments
  torch::Tensor setter_params;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
    model_lock.unlock();
  } catch (...) {
    throw "parameters to set attribute " + attribute_name +
        " not found in model";
  }
  std::cout << "set_attribute: found setter and params" << std::endl;
  // process inputs
  std::vector<c10::IValue> setter_inputs = {};
  for (int i = 0; i < setter_params.size(0); i++) {
    int current_id = setter_params[i].item().toInt();
    std::cout << "current_id : " << current_id << std::endl;
    switch (current_id) {
    // bool case
    case 0:
      setter_inputs.push_back(c10::IValue(to_bool(attribute_args[i])));
      break;
    // int case
    case 1:
      setter_inputs.push_back(c10::IValue(to_int(attribute_args[i])));
      break;
    // float case
    case 2:
      std::cout << "to_float : " << attribute_args[i] << std::endl;
      setter_inputs.push_back(c10::IValue(to_float(attribute_args[i])));
      break;
    // str case
    case 3:
      setter_inputs.push_back(c10::IValue(attribute_args[i]));
      break;
    default:
      throw "bad type id : " + std::to_string(current_id) + "at index " +
          std::to_string(i);
      break;
    }
  }
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto setter_out = m_model.get_method(attribute_setter_name)(setter_inputs);
    model_lock.unlock();
    int setter_result = setter_out.toInt();
    if (setter_result != 0) {
      throw "setter returned -1";
    }
  } catch (...) {
    throw "setter for " + attribute_name + " failed";
  }
}

std::vector<int> Backend::get_method_params(std::string method) {
  std::vector<int> params;

  if (std::find(m_available_methods.begin(), m_available_methods.end(),
                method) != m_available_methods.end()) {
    try {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      auto p = m_model.attr(method + "_params").toTensor();
      model_lock.unlock();
      for (int i(0); i < 4; i++)
        params.push_back(p[i].item().to<int>());
    } catch (...) {
    }
  }
  return params;
}

int Backend::get_higher_ratio() {
  int higher_ratio = 1;
  for (const auto &method : m_available_methods) {
    auto params = get_method_params(method);
    if (!params.size())
      continue; // METHOD NOT USABLE, SKIPPING
    int max_ratio = std::max(params[1], params[3]);
    higher_ratio = std::max(higher_ratio, max_ratio);
  }
  return higher_ratio;
}


std::vector<std::string> Backend::get_available_layers() {
    std::vector<std::string> layers;
    if (!m_loaded) { // Vérification explicite
        std::cerr << "Modèle non chargé!" << std::endl;
        return layers;
    }
    std::unique_lock<std::mutex> model_lock(m_model_mutex); // Verrou tjs actif
    try {
        for (const auto &layer : m_model.named_parameters()) {
            if (!layer.name.empty()) { // Filtre les noms vides
                layers.push_back(layer.name);
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Erreur dans get_available_layers: " << e.what() << std::endl;
        model_lock.unlock(); // Libération explicite
        return layers;
    }
    model_lock.unlock(); // Libération redondante pour sécurité
    return layers;
}


std::vector<float> Backend::get_layer_weights(std::string layer_name) {
    try {
        std::unique_lock<std::mutex> model_lock(m_model_mutex);
        torch::Tensor m_tensor;
        
        bool layer_found = false;
        
        for (const auto &layer : m_model.named_parameters()) {
            if (layer.name == layer_name) {
                m_tensor = layer.value.contiguous().to(CPU);
                layer_found = true;
                break;
            }
        }
        if (!layer_found) {
            throw std::runtime_error("Layer '" + layer_name + "' not found.");
        }
        
        auto m_weights = std::vector(m_tensor.data_ptr<float>(), m_tensor.data_ptr<float>()+m_tensor.numel());
        return m_weights;
    } catch (const std::exception &e) {
        std::cerr << "Error in get_layer_weights: " << e.what() << std::endl;
        return {};
    }
}

void Backend::set_layer_weights(std::string layer_name,
                                std::vector<float> weights) {
    std::cout << "set_layer_weights: entering the function" << std::endl;
      try {
          std::unique_lock<std::mutex> model_lock(m_model_mutex);
          bool layer_found = false;
          
          for (const auto &layer : m_model.named_parameters()) {
              if (layer.name == layer_name) {
                  layer_found = true;
                  
                  if (weights.size() != layer.value.numel()) {
                      throw std::runtime_error("Size mismatch: Expected " + std::to_string(layer.value.numel()) + ", but got " + std::to_string(weights.size()));
                  }
                  
                  torch::NoGradGuard no_grad;
                  torch::Tensor new_weights = torch::from_blob(
                          const_cast<void*>(static_cast<const void*>(weights.data())),
                          layer.value.sizes(),
                          torch::TensorOptions().dtype(torch::kFloat32)
                  );
                  layer.value.copy_(new_weights.clone());
                  std::cout << "set_layer_weights: weights copied" << std::endl;
                  break;
              }
          }
          if (!layer_found) {
              std::cerr << "Layer '" << layer_name << "' not found!" << std::endl;
          }
          
      } catch (const std::exception &e) {
          std::cerr << "Erreur dans set_layer_weights: " << e.what() <<std::endl;
      }
}

bool Backend::is_loaded() { return m_loaded; }

void Backend::use_gpu(bool value) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  if (value) {
    if (torch::hasCUDA()) {
      std::cout << "sending model to cuda" << std::endl;
      m_device = CUDA;
    } else if (torch::hasMPS()) {
      std::cout << "sending model to mps" << std::endl;
      m_device = MPS;
    } else {
      std::cout << "sending model to cpu" << std::endl;
      m_device = CPU;
    }
  } else {
    m_device = CPU;
  }
  m_model.to(m_device);
}

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

Backend::~Backend() {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);

    if (m_loaded) {
        try {
            switch (m_device) {
                case torch::kCUDA:
#ifdef USE_CUDA
                    if (torch::hasCUDA()) {
                        torch::cuda::synchronize();
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
#endif
                    break;
                case torch::kMPS:
                    if (torch::hasMPS()) {
                    }
                    break;
                case torch::kCPU:
                default:
                    at::set_num_threads(1);
                    break;
            }
            m_model = torch::jit::Module();    
            m_loaded = 0;

        } catch (const std::exception& e) {
            std::cerr << "Exception dans Backend::~Backend(): " << e.what() << std::endl;
        }
    }
    model_lock.unlock();
}
