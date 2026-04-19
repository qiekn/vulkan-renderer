export module engine.render_pass;

import vulkan;
import std;
import engine.scene;

namespace engine {

// ---------------------------------------------------------------------------: Context

// Per-frame state handed to every render pass. target_* points at the current
// swapchain image; additional views (depth, G-Buffer, ...) will extend this
// struct when we grow the pipeline beyond a single forward pass.
export struct RenderContext {
  vk::raii::CommandBuffer& cmd;
  vk::ImageView target_view;
  vk::Extent2D target_extent;
  std::uint32_t frame_index = 0;
};

// ---------------------------------------------------------------------------: RenderPass

// Abstract node in the render pipeline. BeginPass/EndPass typically wrap
// beginRendering/endRendering; Render issues draws in between. Dependencies
// are pass names — the manager runs predecessors before dependents.
export class RenderPass {
 public:
  explicit RenderPass(std::string name) : name_(std::move(name)) {}
  virtual ~RenderPass() = default;

  RenderPass(const RenderPass&) = delete;
  RenderPass& operator=(const RenderPass&) = delete;
  RenderPass(RenderPass&&) = delete;
  RenderPass& operator=(RenderPass&&) = delete;

  const std::string& GetName() const { return name_; }
  bool IsEnabled() const { return enabled_; }
  void SetEnabled(bool enabled) { enabled_ = enabled; }

  void AddDependency(std::string name) { dependencies_.push_back(std::move(name)); }
  const std::vector<std::string>& GetDependencies() const { return dependencies_; }

  void Execute(RenderContext& ctx) {
    if (!enabled_) {
      return;
    }
    BeginPass(ctx);
    Render(ctx);
    EndPass(ctx);
  }

 protected:
  virtual void BeginPass(RenderContext& ctx) = 0;
  virtual void Render(RenderContext& ctx) = 0;
  virtual void EndPass(RenderContext& ctx) = 0;

 private:
  std::string name_;
  std::vector<std::string> dependencies_;
  bool enabled_ = true;
};

// ---------------------------------------------------------------------------: RenderPassManager

// Owns passes and runs them in dependency order. Topological sort is lazy —
// recomputed only when the set of passes changes.
export class RenderPassManager {
 public:
  RenderPassManager() = default;

  RenderPassManager(const RenderPassManager&) = delete;
  RenderPassManager& operator=(const RenderPassManager&) = delete;
  RenderPassManager(RenderPassManager&&) = delete;
  RenderPassManager& operator=(RenderPassManager&&) = delete;

  template <typename T, typename... Args>
  T* AddPass(Args&&... args) {
    static_assert(std::is_base_of_v<RenderPass, T>, "T must derive from RenderPass");
    auto pass = std::make_unique<T>(std::forward<Args>(args)...);
    T* raw = pass.get();
    passes_[pass->GetName()] = std::move(pass);
    dirty_ = true;
    return raw;
  }

  RenderPass* GetPass(const std::string& name) {
    auto it = passes_.find(name);
    return it != passes_.end() ? it->second.get() : nullptr;
  }

  void RemovePass(const std::string& name) {
    if (passes_.erase(name) > 0) {
      dirty_ = true;
    }
  }

  void Execute(RenderContext& ctx) {
    if (dirty_) {
      SortPasses();
      dirty_ = false;
    }
    for (RenderPass* pass : sorted_) {
      pass->Execute(ctx);
    }
  }

 private:
  void SortPasses() {
    sorted_.clear();
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> visiting;
    for (const auto& [name, _] : passes_) {
      if (!visited.contains(name)) {
        Visit(name, visited, visiting);
      }
    }
  }

  void Visit(const std::string& name,
             std::unordered_set<std::string>& visited,
             std::unordered_set<std::string>& visiting) {
    if (visiting.contains(name)) {
      throw std::runtime_error("Cycle detected in render passes at: " + name);
    }
    if (visited.contains(name)) {
      return;
    }
    auto it = passes_.find(name);
    if (it == passes_.end()) {
      // Dependency names a pass that is not registered — skip silently; the
      // caller can still add it later and trigger a re-sort.
      visited.insert(name);
      return;
    }
    visiting.insert(name);
    for (const auto& dep : it->second->GetDependencies()) {
      Visit(dep, visited, visiting);
    }
    visiting.erase(name);
    visited.insert(name);
    sorted_.push_back(it->second.get());
  }

  std::unordered_map<std::string, std::unique_ptr<RenderPass>> passes_;
  std::vector<RenderPass*> sorted_;
  bool dirty_ = true;
};

// ---------------------------------------------------------------------------: CullingSystem

// Placeholder visibility filter. Once entities carry a MeshComponent with a
// bounding volume we'll intersect it against the camera frustum; for now we
// just pass through every active entity so the wiring is in place.
export class CullingSystem {
 public:
  void Cull(const Scene& scene) {
    visible_.clear();
    for (const auto& entity : scene.GetEntities()) {
      if (entity->IsActive()) {
        visible_.push_back(entity.get());
      }
    }
  }

  const std::vector<Entity*>& GetVisible() const { return visible_; }

 private:
  std::vector<Entity*> visible_;
};

}  // namespace engine
