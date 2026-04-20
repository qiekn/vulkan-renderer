module;

#include "miniaudio.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.audio;

import std;
import engine.scene;

namespace engine {

// ---------------------------------------------------------------------------: AudioClip

// Metadata-only handle for a sound file on disk. miniaudio decodes on demand
// when an AudioSource is initialized from the clip's path, so we deliberately
// do not preload PCM here — per ep6 scope.
export class AudioClip {
 public:
  explicit AudioClip(std::string path) : path_(std::move(path)) {}

  const std::string& GetPath() const { return path_; }

 private:
  std::string path_;
};

// ---------------------------------------------------------------------------: AudioSource

// Single playback instance backed by a ma_sound. Position/volume/looping setters
// forward to miniaudio. Do not construct directly — obtain one from
// AudioSystem::CreateSource so lifetime is tied to the audio engine.
export class AudioSource {
 public:
  AudioSource(ma_engine* engine, const std::string& path) : engine_(engine) {
    ma_result r =
        ma_sound_init_from_file(engine_, path.c_str(), /*flags=*/0,
                                /*group=*/nullptr, /*doneFence=*/nullptr, &sound_);
    if (r != MA_SUCCESS) {
      throw std::runtime_error("ma_sound_init_from_file failed: " + path);
    }
    initialized_ = true;
  }

  ~AudioSource() {
    if (initialized_) {
      ma_sound_uninit(&sound_);
    }
  }

  AudioSource(const AudioSource&) = delete;
  AudioSource& operator=(const AudioSource&) = delete;
  AudioSource(AudioSource&&) = delete;
  AudioSource& operator=(AudioSource&&) = delete;

  void Play() { ma_sound_start(&sound_); }
  void Stop() { ma_sound_stop(&sound_); }
  bool IsPlaying() const { return ma_sound_is_playing(&sound_) != 0; }

  void SetPosition(const glm::vec3& p) { ma_sound_set_position(&sound_, p.x, p.y, p.z); }
  void SetVolume(float volume) { ma_sound_set_volume(&sound_, volume); }
  void SetLooping(bool looping) { ma_sound_set_looping(&sound_, looping ? MA_TRUE : MA_FALSE); }

 private:
  ma_engine* engine_ = nullptr;
  ma_sound sound_{};
  bool initialized_ = false;
};

// ---------------------------------------------------------------------------: AudioListener

// Thin facade over ma_engine's listener 0. Spatial audio uses this to decide
// how each AudioSource should pan/attenuate. AudioSystem owns the instance.
export class AudioListener {
 public:
  explicit AudioListener(ma_engine* engine) : engine_(engine) {}

  void SetPosition(const glm::vec3& p) {
    ma_engine_listener_set_position(engine_, 0, p.x, p.y, p.z);
  }

  void SetOrientation(const glm::vec3& forward, const glm::vec3& up) {
    ma_engine_listener_set_direction(engine_, 0, forward.x, forward.y, forward.z);
    ma_engine_listener_set_world_up(engine_, 0, up.x, up.y, up.z);
  }

 private:
  ma_engine* engine_ = nullptr;
};

// ---------------------------------------------------------------------------: AudioSystem

// Owns ma_engine plus the clip registry and live AudioSource objects.
// Destruction order matters: all sounds must be uninit'd before ma_engine —
// the explicit destructor enforces that sequence.
export class AudioSystem {
 public:
  AudioSystem() {
    ma_result r = ma_engine_init(/*config=*/nullptr, &engine_);
    if (r != MA_SUCCESS) {
      throw std::runtime_error("ma_engine_init failed");
    }
    engine_initialized_ = true;
    listener_ = std::make_unique<AudioListener>(&engine_);
  }

  ~AudioSystem() {
    sources_.clear();
    clips_.clear();
    listener_.reset();
    if (engine_initialized_) {
      ma_engine_uninit(&engine_);
    }
  }

  AudioSystem(const AudioSystem&) = delete;
  AudioSystem& operator=(const AudioSystem&) = delete;
  AudioSystem(AudioSystem&&) = delete;
  AudioSystem& operator=(AudioSystem&&) = delete;

  // Register a clip by id. Missing files or unsupported formats return false
  // and leave the registry untouched, so the engine keeps running.
  bool LoadClip(const std::string& id, const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
      std::cerr << "[audio] clip not found: " << path.string() << '\n';
      return false;
    }
    clips_[id] = std::make_unique<AudioClip>(path.string());
    return true;
  }

  AudioClip* GetClip(const std::string& id) const {
    auto it = clips_.find(id);
    return it == clips_.end() ? nullptr : it->second.get();
  }

  // Returns a non-owning pointer; AudioSystem keeps ownership. Returns nullptr
  // if the clip id is unknown or ma_sound_init_from_file fails.
  AudioSource* CreateSource(const std::string& clip_id) {
    auto* clip = GetClip(clip_id);
    if (clip == nullptr) {
      return nullptr;
    }
    try {
      auto source = std::make_unique<AudioSource>(&engine_, clip->GetPath());
      AudioSource* raw = source.get();
      sources_.push_back(std::move(source));
      return raw;
    } catch (const std::exception& e) {
      std::cerr << "[audio] CreateSource failed: " << e.what() << '\n';
      return nullptr;
    }
  }

  // Fire-and-forget inline playback (used for one-shot UI triggers). No handle
  // returned — miniaudio tracks lifetime internally.
  void PlayOneShot(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
      return;
    }
    ma_engine_play_sound(&engine_, path.string().c_str(), /*group=*/nullptr);
  }

  void SetMasterVolume(float volume) { ma_engine_set_volume(&engine_, volume); }

  AudioListener& GetListener() { return *listener_; }

  void Update(float /*delta_time*/) {
    // miniaudio runs its own audio thread; nothing to pump per frame today.
    // Kept so Components can piggyback on the same Update lifecycle later.
  }

 private:
  ma_engine engine_{};
  bool engine_initialized_ = false;
  std::unique_ptr<AudioListener> listener_;
  std::unordered_map<std::string, std::unique_ptr<AudioClip>> clips_;
  std::vector<std::unique_ptr<AudioSource>> sources_;
};

// ---------------------------------------------------------------------------: AudioListenerComponent

// Attaches to the same entity as a CameraComponent (or any TransformComponent
// holder) and pushes the transform's pose into the AudioListener each frame.
export class AudioListenerComponent : public Component {
 public:
  explicit AudioListenerComponent(AudioListener* listener) : listener_(listener) {}

  void Update(float /*delta_time*/) override {
    auto* transform = GetOwner()->GetComponent<TransformComponent>();
    if (transform == nullptr || listener_ == nullptr) {
      return;
    }
    listener_->SetPosition(transform->GetPosition());
    glm::vec3 forward = transform->GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = transform->GetRotation() * glm::vec3(0.0f, 1.0f, 0.0f);
    listener_->SetOrientation(forward, up);
  }

 private:
  AudioListener* listener_ = nullptr;
};

// ---------------------------------------------------------------------------: AudioSourceComponent

// Syncs an entity's transform position onto an AudioSource owned by the
// AudioSystem. The pointer is non-owning — AudioSystem outlives the scene.
export class AudioSourceComponent : public Component {
 public:
  explicit AudioSourceComponent(AudioSource* source) : source_(source) {}

  void Update(float /*delta_time*/) override {
    auto* transform = GetOwner()->GetComponent<TransformComponent>();
    if (transform == nullptr || source_ == nullptr) {
      return;
    }
    source_->SetPosition(transform->GetPosition());
  }

  AudioSource* GetSource() const { return source_; }

 private:
  AudioSource* source_ = nullptr;
};

}  // namespace engine
