module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.scene;

import std;

namespace engine {

// ---------------------------------------------------------------------------: Forward declarations

export class Entity;

// ---------------------------------------------------------------------------: Component

export class Component {
 public:
  virtual ~Component() = default;

  virtual void OnAttach() {}
  virtual void OnDetach() {}
  virtual void Update(float /*delta_time*/) {}

  Entity* GetOwner() const { return owner_; }

 private:
  friend class Entity;
  void SetOwner(Entity* owner) { owner_ = owner; }

  Entity* owner_ = nullptr;
};

// ---------------------------------------------------------------------------: Entity

export class Entity {
 public:
  explicit Entity(std::string name) : name_(std::move(name)) {}

  Entity(const Entity&) = delete;
  Entity& operator=(const Entity&) = delete;
  Entity(Entity&&) = delete;
  Entity& operator=(Entity&&) = delete;

  const std::string& GetName() const { return name_; }
  bool IsActive() const { return active_; }
  void SetActive(bool active) { active_ = active; }

  template <typename T, typename... Args>
  T* AddComponent(Args&&... args) {
    static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");
    auto component = std::make_unique<T>(std::forward<Args>(args)...);
    component->SetOwner(this);
    T* raw = component.get();
    components_[typeid(T)] = std::move(component);
    raw->OnAttach();
    return raw;
  }

  template <typename T>
  T* GetComponent() const {
    auto it = components_.find(typeid(T));
    if (it == components_.end()) {
      return nullptr;
    }
    return static_cast<T*>(it->second.get());
  }

  template <typename T>
  bool RemoveComponent() {
    auto it = components_.find(typeid(T));
    if (it == components_.end()) {
      return false;
    }
    it->second->OnDetach();
    components_.erase(it);
    return true;
  }

  void Update(float delta_time) {
    if (!active_) {
      return;
    }
    for (auto& [id, component] : components_) {
      component->Update(delta_time);
    }
  }

 private:
  std::string name_;
  bool active_ = true;
  std::unordered_map<std::type_index, std::unique_ptr<Component>> components_;
};

// ---------------------------------------------------------------------------: Scene

export class Scene {
 public:
  Scene() = default;

  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  Scene(Scene&&) = delete;
  Scene& operator=(Scene&&) = delete;

  Entity* CreateEntity(std::string name) {
    auto entity = std::make_unique<Entity>(std::move(name));
    Entity* raw = entity.get();
    entities_.push_back(std::move(entity));
    return raw;
  }

  void DestroyEntity(Entity* entity) {
    auto it = std::ranges::find_if(entities_,
                                   [entity](const auto& e) { return e.get() == entity; });
    if (it != entities_.end()) {
      entities_.erase(it);
    }
  }

  void Update(float delta_time) {
    for (auto& entity : entities_) {
      entity->Update(delta_time);
    }
  }

  const std::vector<std::unique_ptr<Entity>>& GetEntities() const { return entities_; }

 private:
  std::vector<std::unique_ptr<Entity>> entities_;
};

// ---------------------------------------------------------------------------: TransformComponent

// Affine transform. Matrix is lazily recomputed on query when any setter was
// called since the last cache.
export class TransformComponent : public Component {
 public:
  const glm::vec3& GetPosition() const { return position_; }
  const glm::quat& GetRotation() const { return rotation_; }
  const glm::vec3& GetScale() const { return scale_; }

  void SetPosition(const glm::vec3& position) {
    position_ = position;
    dirty_ = true;
  }

  void SetRotation(const glm::quat& rotation) {
    rotation_ = rotation;
    dirty_ = true;
  }

  void SetScale(const glm::vec3& scale) {
    scale_ = scale;
    dirty_ = true;
  }

  glm::mat4 GetMatrix() const {
    if (dirty_) {
      matrix_ = glm::translate(glm::mat4(1.0f), position_)
              * glm::mat4_cast(rotation_)
              * glm::scale(glm::mat4(1.0f), scale_);
      dirty_ = false;
    }
    return matrix_;
  }

 private:
  glm::vec3 position_ = glm::vec3(0.0f);
  glm::quat rotation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  glm::vec3 scale_ = glm::vec3(1.0f);

  mutable glm::mat4 matrix_ = glm::mat4(1.0f);
  mutable bool dirty_ = true;
};

// ---------------------------------------------------------------------------: CameraComponent

// Reads owning entity's TransformComponent for the view matrix; projection is
// set explicitly by the caller (SetPerspective / SetAspect).
export class CameraComponent : public Component {
 public:
  void SetPerspective(float fov_radians, float aspect, float near_plane, float far_plane) {
    fov_ = fov_radians;
    aspect_ = aspect;
    near_ = near_plane;
    far_ = far_plane;
    proj_dirty_ = true;
  }

  void SetAspect(float aspect) {
    aspect_ = aspect;
    proj_dirty_ = true;
  }

  void SetFov(float fov_radians) {
    fov_ = fov_radians;
    proj_dirty_ = true;
  }

  float GetAspect() const { return aspect_; }
  float GetFov() const { return fov_; }

  glm::mat4 GetViewMatrix() const {
    auto* transform = GetOwner()->GetComponent<TransformComponent>();
    if (transform == nullptr) {
      return glm::mat4(1.0f);
    }
    glm::vec3 position = transform->GetPosition();
    glm::vec3 forward = transform->GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = transform->GetRotation() * glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::lookAt(position, position + forward, up);
  }

  glm::mat4 GetProjectionMatrix() const {
    if (proj_dirty_) {
      proj_ = glm::perspective(fov_, aspect_, near_, far_);
      // GLM produces an OpenGL-style Y-up projection; Vulkan's NDC has Y flipped.
      proj_[1][1] *= -1.0f;
      proj_dirty_ = false;
    }
    return proj_;
  }

 private:
  float fov_ = glm::radians(60.0f);
  float aspect_ = 16.0f / 9.0f;
  float near_ = 0.1f;
  float far_ = 1000.0f;

  mutable glm::mat4 proj_ = glm::mat4(1.0f);
  mutable bool proj_dirty_ = true;
};

}  // namespace engine
