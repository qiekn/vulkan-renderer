export module engine.resource;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: Resource base

// Abstract base for anything that loads data from disk into GPU memory.
// Subclasses override DoLoad/DoUnload; the base tracks Id + loaded state.
export class Resource {
public:
  explicit Resource(std::string id) : id_(std::move(id)) {}
  virtual ~Resource() = default;

  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;
  Resource(Resource&&) = delete;
  Resource& operator=(Resource&&) = delete;

  const std::string& GetId() const { return id_; }
  bool IsLoaded() const { return loaded_; }

  bool Load() {
    if (loaded_) {
      return true;
    }
    loaded_ = DoLoad();
    return loaded_;
  }

  void Unload() {
    if (!loaded_) {
      return;
    }
    DoUnload();
    loaded_ = false;
  }

protected:
  virtual bool DoLoad() = 0;
  virtual void DoUnload() = 0;

private:
  std::string id_;
  bool loaded_ = false;
};

// ---------------------------------------------------------------------------: Forward decl

export class ResourceManager;

// ---------------------------------------------------------------------------: Handle

// Non-owning, validatable reference to a cached resource. Cheap to copy.
// Access the resource via Get() / operator-> / operator*. Call
// ResourceManager::Release(id) when done; the handle itself does not ref-count
// automatically, to keep lifetime decisions explicit.
export template <typename T>
class ResourceHandle {
public:
  ResourceHandle() = default;
  ResourceHandle(std::string id, ResourceManager* manager)
      : id_(std::move(id)), manager_(manager) {}

  T* Get() const;
  bool IsValid() const { return manager_ != nullptr && Get() != nullptr; }
  const std::string& GetId() const { return id_; }

  T* operator->() const { return Get(); }
  T& operator*() const { return *Get(); }
  explicit operator bool() const { return IsValid(); }

private:
  std::string id_;
  ResourceManager* manager_ = nullptr;
};

// ---------------------------------------------------------------------------: Manager

// Two-level cache: std::type_index -> id -> (resource + ref count).
// Load<T>(id, args...) constructs T(id, args...) on cache miss and forwards
// args through to the constructor, allowing per-type extra parameters
// (e.g. file path, shader stage, etc.).
export class ResourceManager {
public:
  ResourceManager() = default;
  ~ResourceManager() { UnloadAll(); }

  ResourceManager(const ResourceManager&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;
  ResourceManager(ResourceManager&&) = delete;
  ResourceManager& operator=(ResourceManager&&) = delete;

  template <typename T, typename... Args>
  ResourceHandle<T> Load(const std::string& id, Args&&... args) {
    static_assert(std::is_base_of_v<Resource, T>, "T must derive from Resource");

    auto& bucket = entries_[std::type_index(typeid(T))];
    auto it = bucket.find(id);
    if (it != bucket.end()) {
      ++it->second.ref_count;
      return ResourceHandle<T>(id, this);
    }

    auto resource = std::make_shared<T>(id, std::forward<Args>(args)...);
    if (!resource->Load()) {
      return {};
    }

    bucket.emplace(id, Entry{std::move(resource), 1});
    return ResourceHandle<T>(id, this);
  }

  template <typename T>
  T* GetResource(const std::string& id) {
    auto bucket_it = entries_.find(std::type_index(typeid(T)));
    if (bucket_it == entries_.end()) {
      return nullptr;
    }
    auto it = bucket_it->second.find(id);
    if (it == bucket_it->second.end()) {
      return nullptr;
    }
    return static_cast<T*>(it->second.resource.get());
  }

  template <typename T>
  bool HasResource(const std::string& id) const {
    auto bucket_it = entries_.find(std::type_index(typeid(T)));
    if (bucket_it == entries_.end()) {
      return false;
    }
    return bucket_it->second.contains(id);
  }

  template <typename T>
  void Release(const std::string& id) {
    auto bucket_it = entries_.find(std::type_index(typeid(T)));
    if (bucket_it == entries_.end()) {
      return;
    }
    auto it = bucket_it->second.find(id);
    if (it == bucket_it->second.end()) {
      return;
    }
    if (--it->second.ref_count <= 0) {
      it->second.resource->Unload();
      bucket_it->second.erase(it);
    }
  }

  void UnloadAll() {
    for (auto& [type, bucket] : entries_) {
      for (auto& [id, entry] : bucket) {
        entry.resource->Unload();
      }
    }
    entries_.clear();
  }

private:
  struct Entry {
    std::shared_ptr<Resource> resource;
    int ref_count = 0;
  };

  std::unordered_map<std::type_index, std::unordered_map<std::string, Entry>> entries_;
};

// Deferred definition now that ResourceManager is complete.
template <typename T>
T* ResourceHandle<T>::Get() const {
  if (manager_ == nullptr) {
    return nullptr;
  }
  return manager_->GetResource<T>(id_);
}

// ---------------------------------------------------------------------------: ShaderResource

// Wraps a SPIR-V blob as a vk::raii::ShaderModule. The module is destroyed
// when the resource is unloaded; keep the handle alive at least as long as
// any pipeline that was built from it.
export class ShaderResource : public Resource {
public:
  ShaderResource(std::string id, Device& device, std::filesystem::path spirv_path)
      : Resource(std::move(id)), device_(device), path_(std::move(spirv_path)) {}

  const vk::raii::ShaderModule& GetModule() const { return module_; }

protected:
  bool DoLoad() override {
    std::ifstream file(path_, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      return false;
    }
    std::vector<char> code(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));

    vk::ShaderModuleCreateInfo info{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
    };
    module_ = vk::raii::ShaderModule(device_.GetLogicalDevice(), info);
    return true;
  }

  void DoUnload() override { module_ = nullptr; }

private:
  Device& device_;
  std::filesystem::path path_;
  vk::raii::ShaderModule module_ = nullptr;
};

}  // namespace engine
