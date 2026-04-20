module;

#include "miniaudio.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.audio;

import std;
import vulkan;
import engine.device;
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

// ---------------------------------------------------------------------------: HrtfSource

// Plays back a pre-baked interleaved stereo PCM buffer through ma_engine as if
// it were any other sound, except spatialization is disabled: the HRTF has
// already been convolved into the samples on the GPU, so we don't want
// miniaudio to pan or attenuate again. The PCM buffer is owned by this object
// and must outlive both ma_audio_buffer and ma_sound — destruction order in the
// dtor is sound → audio_buffer → pcm_ (vector) so the audio thread never reads
// freed memory.
export class HrtfSource {
 public:
  HrtfSource(ma_engine* engine, std::vector<float> stereo_pcm, std::uint32_t sample_rate)
      : engine_(engine), pcm_(std::move(stereo_pcm)) {
    const std::size_t frame_count = pcm_.size() / 2;
    ma_audio_buffer_config cfg = ma_audio_buffer_config_init(
        ma_format_f32, /*channels=*/2, frame_count, pcm_.data(), /*pAllocCb=*/nullptr);
    cfg.sampleRate = sample_rate;
    if (ma_audio_buffer_init(&cfg, &buffer_) != MA_SUCCESS) {
      throw std::runtime_error("ma_audio_buffer_init failed");
    }
    buffer_initialized_ = true;
    if (ma_sound_init_from_data_source(
            engine_, reinterpret_cast<ma_data_source*>(&buffer_),
            MA_SOUND_FLAG_NO_SPATIALIZATION, /*group=*/nullptr, &sound_) != MA_SUCCESS) {
      ma_audio_buffer_uninit(&buffer_);
      buffer_initialized_ = false;
      throw std::runtime_error("ma_sound_init_from_data_source failed");
    }
    sound_initialized_ = true;
  }

  ~HrtfSource() {
    if (sound_initialized_) {
      ma_sound_uninit(&sound_);
    }
    if (buffer_initialized_) {
      ma_audio_buffer_uninit(&buffer_);
    }
  }

  HrtfSource(const HrtfSource&) = delete;
  HrtfSource& operator=(const HrtfSource&) = delete;
  HrtfSource(HrtfSource&&) = delete;
  HrtfSource& operator=(HrtfSource&&) = delete;

  void Play() { ma_sound_start(&sound_); }
  void Stop() { ma_sound_stop(&sound_); }
  bool IsPlaying() const { return ma_sound_is_playing(&sound_) != 0; }

 private:
  ma_engine* engine_ = nullptr;
  std::vector<float> pcm_;
  ma_audio_buffer buffer_{};
  ma_sound sound_{};
  bool buffer_initialized_ = false;
  bool sound_initialized_ = false;
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
    hrtf_sources_.clear();
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

  // Exposed so HRTF code (outside the class) can build ma_sound objects that
  // live inside the same ma_engine. Non-owning — AudioSystem retains ownership.
  ma_engine* GetEngine() { return &engine_; }

  // Adopt an HRTF-backed sound so its lifetime is tracked alongside other
  // sources. Returns a non-owning pointer for callers who want to keep it
  // playing or stop it later.
  HrtfSource* AdoptHrtfSource(std::unique_ptr<HrtfSource> source) {
    HrtfSource* raw = source.get();
    hrtf_sources_.push_back(std::move(source));
    return raw;
  }

  // Drop all baked HRTF sounds. Used by the demo UI so rapid re-bakes don't
  // pile up simultaneously-playing copies.
  void ClearHrtfSources() { hrtf_sources_.clear(); }

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
  std::vector<std::unique_ptr<HrtfSource>> hrtf_sources_;
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

// ---------------------------------------------------------------------------: HrtfClip

// CPU-resident mono PCM decoded from a clip on disk. Unlike AudioClip (which
// keeps only the path and hands it to ma_sound_init_from_file), HRTF needs
// direct access to the samples to upload them into the compute shader's input
// buffer. LoadFromFile uses ma_decoder so we get the file's native sample rate
// back.
export class HrtfClip {
 public:
  static std::unique_ptr<HrtfClip> LoadFromFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
      std::cerr << "[hrtf] clip not found: " << path.string() << '\n';
      return nullptr;
    }

    ma_decoder_config cfg =
        ma_decoder_config_init(ma_format_f32, /*channels=*/1, /*sampleRate=*/0);
    ma_decoder decoder;
    if (ma_decoder_init_file(path.string().c_str(), &cfg, &decoder) != MA_SUCCESS) {
      std::cerr << "[hrtf] ma_decoder_init_file failed: " << path.string() << '\n';
      return nullptr;
    }

    ma_uint64 total_frames = 0;
    ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);

    std::vector<float> pcm(static_cast<std::size_t>(total_frames));
    ma_uint64 frames_read = 0;
    ma_decoder_read_pcm_frames(&decoder, pcm.data(), total_frames, &frames_read);
    pcm.resize(static_cast<std::size_t>(frames_read));

    ma_format out_fmt = ma_format_unknown;
    ma_uint32 out_channels = 0;
    ma_uint32 out_rate = 0;
    ma_decoder_get_data_format(&decoder, &out_fmt, &out_channels, &out_rate, nullptr, 0);

    ma_decoder_uninit(&decoder);

    auto clip = std::unique_ptr<HrtfClip>(new HrtfClip);
    clip->pcm_ = std::move(pcm);
    clip->sample_rate_ = out_rate;
    return clip;
  }

  std::span<const float> GetMono() const { return pcm_; }
  std::uint32_t GetSampleRate() const { return sample_rate_; }

 private:
  HrtfClip() = default;

  std::vector<float> pcm_;
  std::uint32_t sample_rate_ = 0;
};

// ---------------------------------------------------------------------------: BuildHrtfIr

// Synthesize a 512-float HRTF impulse response pair (left[0..255] | right
// [0..255]) from azimuth + elevation. This is a toy model — ITD from an
// inter-ear delay proportional to sin(azimuth), ILD from attenuating the
// contralateral ear, and a mild elevation attenuation. It's enough to hear the
// source move across the head without pulling in a real HRTF dataset.
//
// Azimuth: 0 = front, positive = right of listener, negative = left.
// Elevation: 0 = horizontal, positive = up.
export std::array<float, 512> BuildHrtfIr(float azimuth_rad, float elevation_rad) {
  std::array<float, 512> ir{};
  constexpr int kTaps = 256;

  // Max inter-aural time delay ≈ 0.7 ms; at 48 kHz that's ~34 samples. Rounded
  // to an integer tap offset — sub-sample ITD would need fractional delay.
  constexpr float kMaxItdSamples = 34.0f;
  const int itd_samples = static_cast<int>(std::round(kMaxItdSamples * std::sin(azimuth_rad)));

  const float lateral = std::abs(std::sin(azimuth_rad));
  const float contralateral_gain = std::max(0.0f, 1.0f - 0.7f * lateral);

  const float elev_factor = std::clamp(std::abs(elevation_rad) / 1.5708f, 0.0f, 1.0f);
  const float both_ears_scale = 1.0f - 0.3f * elev_factor;

  int left_delay = std::clamp(itd_samples >= 0 ? itd_samples : 0, 0, kTaps - 1);
  int right_delay = std::clamp(itd_samples <= 0 ? -itd_samples : 0, 0, kTaps - 1);

  const float left_gain = (azimuth_rad <= 0.0f ? 1.0f : contralateral_gain) * both_ears_scale;
  const float right_gain = (azimuth_rad >= 0.0f ? 1.0f : contralateral_gain) * both_ears_scale;

  // 3-tap triangular window [0.25, 0.5, 0.25] centered on the delay tap. Sums
  // to 1.0 so the peak gain lands close to the requested amplitude — no need
  // for a convolution normalization factor.
  auto write_ir = [&](int offset, int delay, float gain) {
    constexpr float kWindow[3] = {0.25f, 0.5f, 0.25f};
    for (int t = -1; t <= 1; ++t) {
      int idx = delay + t;
      if (idx < 0 || idx >= kTaps) {
        continue;
      }
      ir[offset + idx] = gain * kWindow[t + 1];
    }
  };

  write_ir(0, left_delay, left_gain);
  write_ir(kTaps, right_delay, right_gain);
  return ir;
}

// ---------------------------------------------------------------------------: HrtfProcessor

// First compute pipeline in the project. Owns a descriptor set with three
// storage buffers (mono input, stereo output, IR), a command buffer, and a
// fence. Each Bake() call uploads the mono PCM + IR into host-visible buffers,
// submits the compute dispatch, waits on the fence, and reads the interleaved
// stereo result back. Buffers grow on demand — small clips don't pay for large
// allocations up front.
//
// The graphics queue (same family chosen in Device::FindGraphicsQueueFamily)
// is reused for submission because the device requires COMPUTE on that family
// anyway. A dedicated async-compute queue is a follow-up for real-time
// spatialization, not this bake-once demo.
export class HrtfProcessor {
 public:
  explicit HrtfProcessor(Device& device) : device_(device) {
    const auto& dev = device_.GetLogicalDevice();

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    vk::DescriptorSetLayoutCreateInfo set_layout_info{
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    set_layout_ = vk::raii::DescriptorSetLayout(dev, set_layout_info);

    vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(std::uint32_t),
    };
    vk::PipelineLayoutCreateInfo layout_info{
        .setLayoutCount = 1,
        .pSetLayouts = &*set_layout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    pipe_layout_ = vk::raii::PipelineLayout(dev, layout_info);

    std::ifstream file(kShaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error(std::string("HrtfProcessor: failed to open ") + kShaderPath);
    }
    std::vector<char> code(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));
    vk::ShaderModuleCreateInfo shader_info{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
    };
    shader_ = vk::raii::ShaderModule(dev, shader_info);

    vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *shader_,
        .pName = "computeMain",
    };
    vk::ComputePipelineCreateInfo pipe_info{
        .stage = stage,
        .layout = *pipe_layout_,
    };
    pipeline_ = vk::raii::Pipeline(dev, nullptr, pipe_info);

    vk::DescriptorPoolSize pool_size{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 3,
    };
    vk::DescriptorPoolCreateInfo pool_info{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    pool_ = vk::raii::DescriptorPool(dev, pool_info);

    vk::DescriptorSetAllocateInfo set_alloc_info{
        .descriptorPool = *pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &*set_layout_,
    };
    auto sets = vk::raii::DescriptorSets(dev, set_alloc_info);
    set_ = std::move(sets[0]);

    vk::CommandPoolCreateInfo cmd_pool_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = device_.GetGraphicsQueueFamily(),
    };
    cmd_pool_ = vk::raii::CommandPool(dev, cmd_pool_info);

    vk::CommandBufferAllocateInfo cmd_alloc_info{
        .commandPool = *cmd_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto cmds = vk::raii::CommandBuffers(dev, cmd_alloc_info);
    cmd_ = std::move(cmds[0]);

    fence_ = vk::raii::Fence(dev, vk::FenceCreateInfo{});

    CreateHostBuffer(ir_, kIrFloats * sizeof(float));
    // Input/output buffers are allocated lazily in EnsureCapacity; the
    // descriptor set isn't written until then either.
  }

  HrtfProcessor(const HrtfProcessor&) = delete;
  HrtfProcessor& operator=(const HrtfProcessor&) = delete;
  HrtfProcessor(HrtfProcessor&&) = delete;
  HrtfProcessor& operator=(HrtfProcessor&&) = delete;

  // Convolve `mono` with `ir` on the GPU. Returns interleaved stereo PCM
  // (mono.size() * 2 floats). Blocks until the dispatch completes.
  std::vector<float> Bake(std::span<const float> mono, const std::array<float, 512>& ir) {
    if (mono.empty()) {
      return {};
    }
    EnsureCapacity(mono.size());

    std::memcpy(input_.mapped, mono.data(), mono.size_bytes());
    std::memcpy(ir_.mapped, ir.data(), ir.size() * sizeof(float));

    const auto& dev = device_.GetLogicalDevice();

    cmd_.reset();
    vk::CommandBufferBeginInfo begin_info{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    cmd_.begin(begin_info);
    cmd_.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
    cmd_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipe_layout_, 0,
                            *set_, nullptr);
    const std::uint32_t frame_count = static_cast<std::uint32_t>(mono.size());
    cmd_.pushConstants<std::uint32_t>(*pipe_layout_, vk::ShaderStageFlagBits::eCompute, 0,
                                      frame_count);
    const std::uint32_t group_count = (frame_count + 63) / 64;
    cmd_.dispatch(group_count, 1, 1);

    // Explicit shader-write → host-read barrier. Host-coherent memory means
    // the fence signal is sufficient in practice, but the barrier keeps the
    // validation layer quiet and documents the dependency for future readers.
    vk::MemoryBarrier mem_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eHostRead,
    };
    cmd_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eHost,
                         {}, mem_barrier, nullptr, nullptr);

    cmd_.end();

    dev.resetFences(*fence_);
    vk::SubmitInfo submit_info{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd_,
    };
    device_.GetGraphicsQueue().submit(submit_info, *fence_);

    auto wait_result = dev.waitForFences(*fence_, vk::True,
                                         std::numeric_limits<std::uint64_t>::max());
    if (wait_result != vk::Result::eSuccess) {
      throw std::runtime_error("HrtfProcessor: fence wait failed");
    }

    std::vector<float> out(mono.size() * 2);
    std::memcpy(out.data(), output_.mapped, out.size() * sizeof(float));
    return out;
  }

 private:
  static constexpr const char* kShaderPath = "assets/shaders/hrtf.spv";
  static constexpr std::size_t kIrFloats = 512;

  struct HostBuffer {
    vk::raii::Buffer buf = nullptr;
    vk::raii::DeviceMemory mem = nullptr;
    void* mapped = nullptr;
    vk::DeviceSize size = 0;
  };

  void EnsureCapacity(std::size_t mono_frame_count) {
    const vk::DeviceSize input_bytes = mono_frame_count * sizeof(float);
    const vk::DeviceSize output_bytes = mono_frame_count * 2 * sizeof(float);

    bool realloc_any = false;
    if (input_.size < input_bytes) {
      input_ = HostBuffer{};
      CreateHostBuffer(input_, input_bytes);
      realloc_any = true;
    }
    if (output_.size < output_bytes) {
      output_ = HostBuffer{};
      CreateHostBuffer(output_, output_bytes);
      realloc_any = true;
    }
    if (realloc_any) {
      WriteDescriptorSet();
    }
  }

  void CreateHostBuffer(HostBuffer& dst, vk::DeviceSize size) {
    const auto& dev = device_.GetLogicalDevice();
    vk::BufferCreateInfo buf_info{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    dst.buf = vk::raii::Buffer(dev, buf_info);
    vk::MemoryRequirements reqs = dst.buf.getMemoryRequirements();
    vk::MemoryAllocateInfo alloc_info{
        .allocationSize = reqs.size,
        .memoryTypeIndex = FindMemoryType(
            reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    };
    dst.mem = vk::raii::DeviceMemory(dev, alloc_info);
    dst.buf.bindMemory(*dst.mem, 0);
    dst.mapped = dst.mem.mapMemory(0, size);
    dst.size = size;
  }

  void WriteDescriptorSet() {
    std::array<vk::DescriptorBufferInfo, 3> buffer_infos{{
        {.buffer = *input_.buf, .offset = 0, .range = input_.size},
        {.buffer = *output_.buf, .offset = 0, .range = output_.size},
        {.buffer = *ir_.buf, .offset = 0, .range = ir_.size},
    }};
    std::array<vk::WriteDescriptorSet, 3> writes{{
        {.dstSet = *set_,
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &buffer_infos[0]},
        {.dstSet = *set_,
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &buffer_infos[1]},
        {.dstSet = *set_,
         .dstBinding = 2,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &buffer_infos[2]},
    }};
    device_.GetLogicalDevice().updateDescriptorSets(writes, nullptr);
  }

  std::uint32_t FindMemoryType(std::uint32_t type_filter,
                               vk::MemoryPropertyFlags properties) const {
    auto props = device_.GetPhysicalDevice().getMemoryProperties();
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      bool type_ok = (type_filter & (1u << i)) != 0;
      bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
      if (type_ok && props_ok) {
        return i;
      }
    }
    throw std::runtime_error("HrtfProcessor: no suitable memory type");
  }

  Device& device_;

  vk::raii::DescriptorSetLayout set_layout_ = nullptr;
  vk::raii::PipelineLayout pipe_layout_ = nullptr;
  vk::raii::ShaderModule shader_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
  vk::raii::DescriptorPool pool_ = nullptr;
  vk::raii::DescriptorSet set_ = nullptr;
  vk::raii::CommandPool cmd_pool_ = nullptr;
  vk::raii::CommandBuffer cmd_ = nullptr;
  vk::raii::Fence fence_ = nullptr;

  HostBuffer input_;
  HostBuffer output_;
  HostBuffer ir_;
};

}  // namespace engine
