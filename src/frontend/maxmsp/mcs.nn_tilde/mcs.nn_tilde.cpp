#include "../../../backend/backend.h"
#include "../shared/circular_buffer.h"
#include "c74_min.h"
#include <chrono>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <future>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

namespace {
    static std::mutex g_model_load_mutex;
}

using namespace c74::min;

unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}


long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count);
long simplemc_inputchanged(c74::max::t_object *x, long index, long count);

class mc_bnn_tilde : public object<mc_bnn_tilde>, public mc_operator<> {
private:
        c74::max::t_clock* m_clock_layers = nullptr;
        c74::max::t_clock* m_clock_weights = nullptr;
        c74::max::t_clock* m_clock_error = nullptr;
        atoms m_cached_layers_result;
        std::vector<float> m_cached_weights_result;
        std::unordered_map<std::string, size_t> m_layer_sizes;

        // Backend & gestion du modèle
        c74::min::path m_path;
        bool m_is_backend_init = false;

        // Threads et accès concurrents
        std::mutex m_cache_mutex;
        std::mutex m_model_mutex;
        std::mutex model_access_mutex;
        std::atomic<bool> m_is_destroying {false};
        std::atomic<bool> m_loading{false};
        std::atomic<bool> processing_active {true};
        std::atomic<bool> enable {false};
        std::thread m_load_thread;
        std::thread m_weights_thread;
        std::thread m_layers_thread;
        std::unique_ptr<std::thread> m_compute_thread;

        // Fonctions internes
        void send_weights();
        void send_layers();
        static void model_perform_loop(mc_bnn_tilde* instance);
        static void send_layers_static(void* x);
        static void send_weights_static(void* x);
        static void get_weights_thread(mc_bnn_tilde* self, std::string layer_name);
        static void send_error_static(void* x);

        // AUDIO PERFORM
        bool m_use_thread, m_should_stop_perform_thread;
        std::binary_semaphore m_data_available_lock, m_result_available_lock;
       
        // BUFFER RELATED MEMBERS
        std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
        std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;

        // Paramètres internes du modèle
        std::vector<std::string> settable_attributes;
        bool has_settable_attribute(std::string attribute);
        
    
public:
        MIN_DESCRIPTION{
          "Multi-channel interface for deep learning models (batch version)"};
        MIN_TAGS{"audio, deep learning, ai"};
        MIN_AUTHOR{"Antoine Caillon, Axel Chemla--Romeu-Santos"};

        mc_bnn_tilde(const atoms &args = {});
        ~mc_bnn_tilde();

        // INLETS OUTLETS
        std::vector<std::unique_ptr<inlet<>>> m_inlets;
        std::vector<std::unique_ptr<outlet<>>> m_outlets;
        std::unique_ptr<outlet<>> m_attribute_outlet;

        // CHANNELS
        std::vector<int> input_chans;
        int get_batches();
        bool check_inputs();
        
        // MODELS
        std::string m_method;
        int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio, m_batches;
        std::unique_ptr<Backend> m_model;
        int m_buffer_size;
        std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

        // FUNCTIONS
        void load_model(const std::string& model_path, const std::string& method_name);
        void initialize_after_load();
        void operator()(audio_bundle input, audio_bundle output);
        void perform(audio_bundle input, audio_bundle output);
        std::vector<float> resample_vector(const std::vector<float>& input, size_t target_size, int mode);
        int downsample_mode = 0;

    
        // ONLY FOR DOCUMENTATION
        argument<symbol> path_arg{this, "model path",
                                "Absolute path to the pretrained model."};
        argument<symbol> method_arg{this, "method",
                                  "Name of the method to call during synthesis."};
        argument<int> batches_arg{this, "batches", "Number of batches"};

        argument<int> buffer_arg{
          this, "buffer size",
          "Size of the internal buffer (can't be lower than the method's ratio)."};

        // ENABLE / DISABLE GPU
        attribute<bool> gpu{this, "gpu", false,
                            description{"Enable / disable gpu usage when available"},
                            setter{[this](const c74::min::atoms &args,
                                          const int inlet) -> c74::min::atoms {
                              if (m_is_backend_init)
                                m_model->use_gpu(bool(args[0]));
                              return args;
                            }}};

        // BOOT STAMP
        message<> maxclass_setup{
          this, "maxclass_setup",
          [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
            cout << "nn~ " << VERSION << " - torch " << TORCH_VERSION;
            c74::max::t_class *c = args[0];
            c74::max::class_addmethod(
                c, (c74::max::method)simplemc_multichanneloutputs,
                "multichanneloutputs", c74::max::A_CANT, 0);
            c74::max::class_addmethod(c, (c74::max::method)simplemc_inputchanged,
                                      "inputchanged", c74::max::A_CANT, 0);
            return {};
        }};

        message<> run{ this, "run", "Enable or disable tensor calculation",
            MIN_FUNCTION{
                if (args.size() <1) {
                    cerr << "enable: requires an argument (0 or 1)" << endl;
                    return {};
                }
                enable = bool(args[0]);
                return {};
            }
        };

        message<> mode{ this, "mode", "Mode de downsampling pour les paramètre du modèle: 0=linéaire, 1=log, 2=cosine",
            MIN_FUNCTION{
                if (args.size() <1) {
                    cerr << "downsample_mode: requires an argument (0 or 1)" << endl;
                    return {};
                }
                downsample_mode = int(args[0]);
                return {};
            }
        };

        message<> load { this, "load", "load a .ts model file",
              MIN_FUNCTION {
                if (args.size() < 1) {
                  cerr << "No file specified" << endl;
                  return{};
                }
                if (args.size() < 2) {
                  cerr << "No method specified" << endl;
                  return{};
                }
                if (args.size() > 2) {
                  cerr << "Message load should contain a model name and a method" << endl;
                  return{};
                }

                std::string model_path = std::string(args[0]);
                if (model_path.substr(model_path.length() - 3) != ".ts") {
                  model_path = model_path + ".ts";
                }

                m_path = path(model_path);
                m_method = std::string(args[1]);

                if (m_loading) {
                    cerr << "Load déjà en cours, requête ignorée" << endl;
                    return {};
                }

                if (m_load_thread.joinable()) {
                    m_load_thread.join();
                }

                m_loading = true;  

                m_load_thread = std::thread([this, path = std::string(m_path), method = m_method]() {
                    try {
                        this->load_model(path, method);
                    } catch (const std::exception& e) {
                        cerr << "Exception in load_model()" << endl;
                    }
                });

                return{};
            }
        };

        message<> get{this, "get",
            MIN_FUNCTION {
                if (args.size() < 1) {
                    cerr << "get: must be given an attribute name" << endl;
                    return {};
                }
                if (args.size() > 1) {
                    cerr << "get: must be given one attribute name" << endl;
                    return {};
                }
                symbol attribute_name = args[0];
                string attribute_value = m_model->get_attribute_as_string(attribute_name);
                m_attribute_outlet->send("attribute", attribute_value);
                return {};
            }
        };
         
         
        //GET LAYERS
        message<> layers{this, "layers", MIN_FUNCTION{
            if (!m_model || !m_is_backend_init || !m_model->is_loaded()) {
                cerr << "Modèle non initialisé !" << endl;
                return {};
            }
            
            if (m_layers_thread.joinable()) {
                m_layers_thread.join();
            }
            m_layers_thread = std::thread([this]() {
                if (m_is_destroying) return;
                try {
                    atoms output_atoms;
                    std::vector<std::string> layers = m_model->get_available_layers();
                    for (const auto& layer : layers) {
                        if (!layer.empty()) {
                            output_atoms.push_back(symbol(layer.c_str()));
                        }
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(m_cache_mutex);
                        m_cached_layers_result = output_atoms;
                    }
                    if (m_is_destroying) return;
                    c74::max::clock_delay(m_clock_layers, 0);
                    
                } catch (const std::exception& e) {
                    std::cerr << "Erreur dans 'layers': " << e.what() << std::endl;
                }
            });
            m_layers_thread.detach();
            return {};
        }};

        //GET LAYER WEIGHTS
        message<> get_weights{this, "get_weights", "Retrieve the weights of a layer",
            MIN_FUNCTION {
                if (!m_model || !m_is_backend_init || !m_model->is_loaded()) {
                    cerr << "Modèle non initialisé !" << endl;
                    return {};
                }
                if (args.empty()) {
                    cerr << "get_weights: must be given a layer name" << endl;
                    return {};
                }
                if (args.size() < 1 || args[0].a_type != c74::max::A_SYM) {
                    cerr << "get_weights() : Argument invalide !" << endl;
                    return {};
                }
                if (args.size() > 1) {
                    cerr << "get_weights() : should be given one argument !" << endl;
                    return {};
                }
                std::string layer_name_copy = args[0];
                if (m_weights_thread.joinable()) {
                    m_weights_thread.join();
                }
                
                m_weights_thread = std::thread(get_weights_thread, this, layer_name_copy);
                m_weights_thread.detach();
                return {};
            }};

        //SET LAYER WEIGHTS
        message<> set_weights{this, "set_weights", "Set weights of a layer",
            MIN_FUNCTION {
                if (args.size() < 1 || static_cast<int>(args[0].type()) != c74::max::A_SYM) {
                    cerr << "set_weights: first argument must be a layer name (symbol)" << endl;
                    return {};
                }
                std::string layer_name = args[0];
                std::vector<float> layer_weights;
                std::vector<float> upsampled;

                for (size_t i = 1; i < args.size(); ++i) {
                    if (static_cast<int>(args[i].type()) != c74::max::A_FLOAT) {
                        cerr << "set_weights: arguments must be floats" << endl;
                        return{};
                    }
                    layer_weights.push_back(args[i]);
                }
                
                if (layer_weights.empty()) {
                    cerr << "set_weights: no weights provided" << endl;
                    return {};
                }
                if (m_layer_sizes.find(layer_name) != m_layer_sizes.end()) {
                    size_t expected_size = m_layer_sizes[layer_name];
                    if (layer_weights.size() != expected_size) {
                        upsampled = resample_vector(layer_weights, expected_size, downsample_mode);
                    } else { 
                        upsampled = layer_weights;
                    };
                } else {
                    cerr << "Unknown layer: " << layer_name << endl;
                    return {};
                }
                std::thread([this, layer_name, upsampled]() {
                    if (m_is_destroying) return;
                    try {
                        std::lock_guard<std::mutex> lock(model_access_mutex);
                        m_model->set_layer_weights(layer_name, upsampled);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Exception in update_layer_weights_async: " << e.what() << std::endl;
                    }
                }).detach();
                
                m_attribute_outlet->send("set");
                return{};
            }
        };

        message<> reload{this, "reload", "reload backend model",
            MIN_FUNCTION{
                m_model->reload();
                cout << "model reloaded" << endl;
                m_attribute_outlet->send("reloaded");
                return {};
            }
        };
 
};

int mc_bnn_tilde::get_batches() { return m_batches; }

void model_perform(mc_bnn_tilde *mc_nn_instance) {
  std::vector<float *> in_model, out_model;
  auto num_batches = mc_nn_instance->get_batches();
  for (int c(0); c < mc_nn_instance->m_in_dim * num_batches; c++)
    in_model.push_back(mc_nn_instance->m_in_model[c].get());
  for (int c(0); c < mc_nn_instance->m_out_dim * num_batches; c++)
    out_model.push_back(mc_nn_instance->m_out_model[c].get());

  mc_nn_instance->m_model->perform(
      in_model, out_model, mc_nn_instance->m_buffer_size,
      mc_nn_instance->m_method, mc_nn_instance->get_batches());
}

void mc_bnn_tilde::model_perform_loop(mc_bnn_tilde *mc_nn_instance) {
  std::vector<float *> in_model, out_model;

  for (auto &ptr : mc_nn_instance->m_in_model)
    in_model.push_back(ptr.get());

  for (auto &ptr : mc_nn_instance->m_out_model)
    out_model.push_back(ptr.get());

  while (!mc_nn_instance->m_should_stop_perform_thread) {
    if (mc_nn_instance->m_data_available_lock.try_acquire_for(
            std::chrono::milliseconds(5))) {
        if (mc_nn_instance->m_should_stop_perform_thread)
            break;
        {
            std::lock_guard<std::mutex> lock(mc_nn_instance->m_model_mutex);
            if (!mc_nn_instance->m_model)
                break;
        }
        if (mc_nn_instance->m_should_stop_perform_thread)
            break;
        mc_nn_instance->m_model->perform(
            in_model, out_model, mc_nn_instance->m_buffer_size,
            mc_nn_instance->m_method, mc_nn_instance->get_batches());
        mc_nn_instance->m_result_available_lock.release();
    }
  }
}

//CONSTRUCTOR

mc_bnn_tilde::mc_bnn_tilde(const atoms &args)
        : m_method("forward"),
        m_in_dim(1), m_in_ratio(1),
        m_out_dim(1), m_out_ratio(1),
        m_buffer_size(2048),
        m_use_thread(true),  
        m_should_stop_perform_thread(false),
        m_compute_thread(nullptr),
        m_data_available_lock(0),
        m_result_available_lock(1),
        m_batches(1) {

    m_clock_layers = c74::max::clock_new(this, (c74::max::method) &mc_bnn_tilde::send_layers_static);
    m_clock_weights = c74::max::clock_new(this, (c74::max::method) &mc_bnn_tilde::send_weights_static);
    m_clock_error = c74::max::clock_new(this, (c74::max::method) &mc_bnn_tilde::send_error_static);

    m_model = std::make_unique<Backend>();
    m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(1);
    m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(1);
    m_in_buffer[0].initialize(m_buffer_size);
    m_out_buffer[0].initialize(m_buffer_size);
    m_inlets.push_back(std::make_unique<inlet<>>(this, "model input", "multichannelsignal"));
    m_outlets.push_back(std::make_unique<outlet<>>(this, "model output", "multichannelsignal"));
    m_attribute_outlet = std::make_unique<outlet<>>(this, "Messages from model", "message");
    input_chans.push_back(1);

    if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
        auto model_path = std::string(args[0]);
        if (model_path.substr(model_path.length() - 3) != ".ts")
            model_path = model_path + ".ts";
            m_path = path(model_path);    
        if (args.size() > 1) { // TWO ARGUMENTS ARE GIVEN
          m_method = std::string(args[1]);
        }
        if (args.size() > 2) { // THREE ARGUMENTS ARE GIVEN
          m_batches = int(args[2]);
        }
        if (args.size() > 3) { // FOUR ARGUMENTS ARE GIVEN
          m_buffer_size = int(args[3]);
        }
        load_model(m_path, m_method);

    }
}


//DESTRUCTOR
mc_bnn_tilde::~mc_bnn_tilde() {
    enable = false;
    m_is_destroying = true;
    m_should_stop_perform_thread = true;
    processing_active = false;

    if (m_clock_layers) {
        c74::max::clock_unset(m_clock_layers);
        c74::max::object_free(m_clock_layers);
        m_clock_layers = nullptr;
    }
    if (m_clock_weights) {
        c74::max::clock_unset(m_clock_weights);
        c74::max::object_free(m_clock_weights);
        m_clock_weights = nullptr;
    }
    if (m_clock_error) {
        c74::max::clock_unset(m_clock_error);
        c74::max::object_free(m_clock_error);
        m_clock_error = nullptr;
    }


   {
           std::lock_guard<std::mutex> lock(m_model_mutex);
           m_data_available_lock.try_acquire();
           m_data_available_lock.release();
           m_result_available_lock.try_acquire();
           m_result_available_lock.release();

           if (m_compute_thread && m_compute_thread->joinable()) {
               m_compute_thread->join();
               m_compute_thread.reset();
           }
           if (m_layers_thread.joinable()) {
               m_layers_thread.join();
           }
           m_layers_thread = std::thread();
    
           if (m_load_thread.joinable()) {
               m_load_thread.join();
           }
           m_load_thread = std::thread();

           if (m_weights_thread.joinable()) {
               m_weights_thread.join();
           }
           m_weights_thread = std::thread();

           m_in_buffer.reset();
           m_out_buffer.reset();
           m_in_model.clear();
           m_out_model.clear();

           m_model.reset();
    }
}

//LOAD MODEL
void mc_bnn_tilde::load_model(const std::string& model_path, const std::string& method_name) {

    // Désactiver le traitement audio pendant le chargement
    processing_active = false;
    enable = false;
    m_is_backend_init = false;

    std::lock_guard<std::mutex> global_lock(g_model_load_mutex);

    // Verrouiller les mutex dans l'ordre pour éviter tout deadlock
    std::unique_lock<std::mutex> modelLock(m_model_mutex, std::defer_lock);
    std::unique_lock<std::mutex> accessLock(model_access_mutex, std::defer_lock);
    std::lock(modelLock, accessLock);

    // Mettre à jour le chemin et la méthode du modèle
    this->m_path = model_path;
    this->m_method = method_name;

    // Stop threads compute & secondaires
    m_should_stop_perform_thread = true;
    m_data_available_lock.release();
    m_result_available_lock.release();
    if (m_compute_thread && m_compute_thread->joinable()) {
        auto ft = std::async(std::launch::async, [&] {
            return m_compute_thread->join();
        });
        auto status = ft.wait_for(std::chrono::milliseconds(300));
        if (status != std::future_status::ready) {
            cerr << "Warning: compute thread did not terminate properly" << endl;
        }
        m_compute_thread.reset();
    }
    if (m_layers_thread.joinable()) {
        m_layers_thread.join();
    }
    m_layers_thread = std::thread();

    if (m_weights_thread.joinable()) {
        m_weights_thread.join();
    }
    m_weights_thread = std::thread();

    m_in_buffer = nullptr;
    m_out_buffer = nullptr;
    m_in_model.clear();
    m_out_model.clear();
    m_model.reset();
    m_model = nullptr;

    m_model = std::make_unique<Backend>();
    m_is_backend_init = true;
    
    try {
            if (m_model->load(this->m_path) != 0) {
                error("model loading failed");
                m_loading = false;
                processing_active = true;
                return;
            }
    } catch (const std::exception& e) {
        cerr << "Exception dans m_model->load() : " << e.what() << endl;
        error("mcs.nn~: Exception in load_model()");
        m_loading = false;
        processing_active = true;
        return;
    } catch (...) {
        cerr << "Exception inconnue dans m_model->load()" << endl;
        error("mcs.nn~: unknown exception in load_model()");
        m_loading = false;
        processing_active = true;
        return;
    }

    try {
        initialize_after_load();

    } catch (const std::exception& e) {
        cerr << "Erreur dans initialize_after_load : " << e.what() << endl;
        c74::max::clock_delay(m_clock_error, 0);
        m_loading = false;
        processing_active = true;
        return;
    }

    atoms in{ "m_in_dim" };
    in.push_back(m_in_dim);
    m_attribute_outlet->send(in);

    m_attribute_outlet->send("loaded");
    m_loading = false;
    processing_active = true;

    modelLock.unlock();
    accessLock.unlock();

    m_should_stop_perform_thread = false;

    try {
        while(m_data_available_lock.try_acquire_for(std::chrono::milliseconds(1))) {}
    } catch(...) {}

    try {
        while(m_result_available_lock.try_acquire_for(std::chrono::milliseconds(1))) {}
    } catch(...) {}

    m_result_available_lock.release();

    if (m_use_thread) {
        m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
    }
}


//INITIALIZE
void mc_bnn_tilde::initialize_after_load() {
    
    // FIND MINIMUM BUFFER SIZE GIVEN MODEL RATIO
    m_higher_ratio = 1;
    auto model_methods = m_model->get_available_methods();
    for (int i(0); i < model_methods.size(); i++) {
      auto params = m_model->get_method_params(model_methods[i]);
      if (!params.size())
        continue; // METHOD NOT USABLE, SKIPPING
      int max_ratio = std::max(params[1], params[3]);
      m_higher_ratio = std::max(m_higher_ratio, max_ratio);
    }
    
    // GET MODEL'S METHOD PARAMETERS
    auto params = m_model->get_method_params(m_method);
    if (params.size() < 4) {
        cout << "Erreur : les paramètres du modèle sont incomplets !" << endl;
    }
    if (!params.size()) {
        error("method " + m_method + " not found !");
        throw std::runtime_error("error prior not found");
    }
    
    // GET MODEL'S SETTABLE ATTRIBUTES
    try {
        settable_attributes = m_model->get_settable_attributes();
    } catch (...) {
    }

    //CREATE A LIST WITH LAYERS SIZE
    auto layers = m_model->get_available_layers();
    for (const auto& layer : layers) {
        auto weights = m_model->get_layer_weights(layer);
        m_layer_sizes[layer] = weights.size();
    }
    
    
    m_in_dim = params[0];
    m_in_ratio = params[1];
    m_out_dim = params[2];
    m_out_ratio = params[3];
    
    input_chans.clear();
    for (int i(0); i < m_batches; i++)
        input_chans.push_back(m_in_dim);
    
    if (!m_buffer_size) {
        // NO THREAD MODE
        m_use_thread = false;
        m_buffer_size = m_higher_ratio;
    } else if (m_buffer_size < m_higher_ratio) {
        m_buffer_size = m_higher_ratio;
        cerr << "buffer size too small, switching to " << m_buffer_size << endl;
    } else {
        m_buffer_size = power_ceil(m_buffer_size);
    }
    
    
    // CREATE BUFFERS
    auto new_in_buffer = std::make_unique<circular_buffer<double, float>[]>(
                                                                     m_in_dim * get_batches());
    
    auto new_out_buffer = std::make_unique<circular_buffer<float, double>[]>(
                                                                      m_out_dim * get_batches());
    m_in_buffer.swap(new_in_buffer);
    m_out_buffer.swap(new_out_buffer);

    for (int i(0); i < std::max(1, m_in_dim) * get_batches(); i++) {
        m_in_buffer[i].initialize(m_buffer_size);
        m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
    }
    
    for (int i(0); i < std::max(1, m_out_dim) * get_batches(); i++) {
        m_out_buffer[i].initialize(m_buffer_size);
        m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
    }

};

//DOWN-UPSAMPLING

std::vector<float> mc_bnn_tilde::resample_vector(const std::vector<float>& input, size_t target_size, int mode) {
    size_t original_size = input.size();
    if (original_size == target_size) return input;

    std::vector<float> output;
    output.reserve(target_size);

    for (size_t i = 0; i < target_size; ++i) {
        float u = static_cast<float>(i) / (target_size - 1);
        float curved_idx = 0.0f;

        switch (mode) {
            case 1: // Logarithmique
                curved_idx = std::log(1 + u * 9) / std::log(10);
                break;
            case 2: // Cosine
                curved_idx = (1 - std::cos(u * M_PI)) / 2;
                break;
            default: // Linéaire
                curved_idx = u;
                break;
        }

        float idx = curved_idx * (original_size - 1);
        int left = static_cast<int>(idx);
        int right = std::min(left + 1, static_cast<int>(original_size - 1));
        float frac = idx - left;

        // interpolation linéaire
        float interpolated = (1.0f - frac) * input[left] + frac * input[right];
        output.push_back(interpolated);
    }

    return output;
}

//LOW PRIORITY MESSSAGES TO ATTRIBUTE OUTLET
void mc_bnn_tilde::send_layers_static(void* x) {
    auto* self = static_cast<mc_bnn_tilde*>(x);
    if (self) {
        self->send_layers();
    }
}
void mc_bnn_tilde::send_layers() {
    atoms msg{ "layers" };
    msg.insert(msg.end(), m_cached_layers_result.begin(), m_cached_layers_result.end());
    m_attribute_outlet->send(msg);
    m_attribute_outlet->send("layers", "done");
}


void mc_bnn_tilde::send_weights_static(void* x) {
    auto* self = static_cast<mc_bnn_tilde*>(x);
    if (self) {
        self->send_weights();
    }
}

void mc_bnn_tilde::send_weights() {
    const int max_atoms = 32767;
    std::vector<float> downsampled;

    if (m_cached_weights_result.size() > max_atoms) {
        downsampled = resample_vector(m_cached_weights_result, max_atoms, downsample_mode);
    } else { 
        downsampled = m_cached_weights_result; 
    };

    size_t max_size = 8192;
    size_t total_size = downsampled.size();
    for (size_t i = 0; i < total_size; i += max_size) {
        size_t chunk_size = std::min(max_size, total_size - i);
        c74::min::atoms chunk;
        chunk.reserve(chunk_size);
        for (size_t j = 0; j < chunk_size; ++j) {
            chunk.push_back(downsampled[i + j]);
        }
        m_attribute_outlet->send(chunk);
    }
    m_attribute_outlet->send("bang");
}

void mc_bnn_tilde::send_error_static(void* x) {
    if (auto* self = static_cast<mc_bnn_tilde*>(x)) {
        atoms msg = { symbol("error"), symbol("prior"), symbol("not"), symbol("found"), 0 };
        self->m_attribute_outlet->send(msg);
    }
}


//GESTION THREAD POUR L'ACCES LOCAL À LAYER
void mc_bnn_tilde::get_weights_thread(mc_bnn_tilde* self, std::string layer_name) {
    if (self->m_is_destroying) return;
    try {
        std::vector<float> weights = self->m_model->get_layer_weights(layer_name);
        {
            std::lock_guard<std::mutex> lock(self->m_cache_mutex);
            self->m_cached_weights_result = weights;
            
        }
        if (self->m_is_destroying) return;
        c74::max::clock_delay(self->m_clock_weights, 0);

    } catch (const std::exception& e) {
        std::cerr << "Erreur dans get_weights: " << e.what() << std::endl;
    }
}

bool mc_bnn_tilde::has_settable_attribute(std::string attribute) {
  for (std::string candidate : settable_attributes) {
    if (candidate == attribute)
      return true;
  }
  return false;
}

void fill_with_zero(audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

bool mc_bnn_tilde::check_inputs() {
  bool check = true;
  if (!m_model || !m_model->is_loaded()) return true;
  for (int i = 0; i < input_chans.size(); i++) {
    if (input_chans[i] != m_in_dim)
      check = false;
  }
  return check;
}

void mc_bnn_tilde::operator()(audio_bundle input, audio_bundle output) {
  auto dsp_vec_size = output.frame_count();

  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_model 
        || !m_model->is_loaded() 
        || !enable 
        || !check_inputs()
        || !processing_active) {
    fill_with_zero(output);
    return;
  }

  std::lock_guard<std::mutex> lock(m_model_mutex);

  // CHECK IF DSP_VEC_SIZE IS LARGER THAN BUFFER SIZE
  if (dsp_vec_size > m_buffer_size) {
    cerr << "vector size (" << dsp_vec_size << ") ";
    cerr << "larger than buffer size (" << m_buffer_size << "). ";
    cerr << "disabling model.";
    cerr << endl;
    enable = false;
    fill_with_zero(output);
    return;
  }
  perform(input, output);
}

void mc_bnn_tilde::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = input.frame_count();

  if (!m_model || !m_in_buffer || !m_out_buffer || m_inlets.empty() || !processing_active) {
        fill_with_zero(output);
        return;
  }

  // COPY INPUT TO CIRCULAR BUFFER
  for (int b(0); b < m_inlets.size(); b++) {
    for (int d(0); d < m_in_dim; d++) {
      auto in = input.samples(b * m_in_dim + d);
      m_in_buffer[d * get_batches() + b].put(in, vec_size);
    }
  }

  if (m_in_buffer[0].full()) { // BUFFER IS FULL
    if (!m_use_thread) {
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim * get_batches(); c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // CALL MODEL PERFORM IN CURRENT THREAD
      model_perform(this);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim; c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

    } else if (m_result_available_lock.try_acquire()) {
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim * get_batches(); c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim * get_batches(); c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_data_available_lock.release();
    }
  }

  // COPY CIRCULAR BUFFER TO OUTPUT
  for (int b(0); b < m_outlets.size(); b++) {
    for (int d(0); d < m_out_dim; d++) {
      auto out = output.samples(b * m_out_dim + d);
      m_out_buffer[b * m_out_dim + d].get(out, vec_size);
    }
  }
}

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count) {
  minwrap<mc_bnn_tilde> *ob = (minwrap<mc_bnn_tilde> *)(x);
  std::cerr << "Nombre de canaux de sortie demandé : " 
       << ob->m_min_object.m_out_dim 
       << std::endl;
  return ob->m_min_object.m_out_dim;
}

long simplemc_inputchanged(c74::max::t_object *x, long index, long count) {
  minwrap<mc_bnn_tilde> *ob = (minwrap<mc_bnn_tilde> *)(x);
  auto chan_number = ob->m_min_object.m_in_dim;
  ob->m_min_object.input_chans[index] = count;
  return false;
}


MIN_EXTERNAL(mc_bnn_tilde);
