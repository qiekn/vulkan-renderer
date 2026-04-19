module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.camera_controller;

import std;
import engine.event;
import engine.scene;

namespace engine {

// ---------------------------------------------------------------------------: CameraController

// Translates input events into transform + camera parameter changes.
// Keyboard state is tracked as a set of currently-pressed keys so held keys
// drive continuous movement in Update(). Mouse move events accumulate into
// yaw/pitch Euler angles, which we write back as a quaternion on the attached
// TransformComponent. Scroll adjusts the camera's FOV for a zoom effect.
export class CameraController : public EventListener {
public:
  CameraController(TransformComponent& transform, CameraComponent& camera)
      : transform_(transform), camera_(camera) {
    SyncAnglesFromTransform();
  }

  void SetMovementSpeed(float speed) { movement_speed_ = speed; }
  void SetMouseSensitivity(float sensitivity) { mouse_sensitivity_ = sensitivity; }
  void SetMouseCaptured(bool captured) {
    mouse_captured_ = captured;
    first_mouse_ = true;
  }

  void Update(float delta_time) {
    glm::quat rotation = transform_.GetRotation();
    glm::vec3 forward = rotation * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 right = rotation * glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 world_up(0.0f, 1.0f, 0.0f);

    glm::vec3 position = transform_.GetPosition();
    float velocity = movement_speed_ * delta_time;

    if (pressed_keys_.contains(GLFW_KEY_W)) position += forward * velocity;
    if (pressed_keys_.contains(GLFW_KEY_S)) position -= forward * velocity;
    if (pressed_keys_.contains(GLFW_KEY_D)) position += right * velocity;
    if (pressed_keys_.contains(GLFW_KEY_A)) position -= right * velocity;
    if (pressed_keys_.contains(GLFW_KEY_SPACE)) position += world_up * velocity;
    if (pressed_keys_.contains(GLFW_KEY_LEFT_CONTROL)) position -= world_up * velocity;

    transform_.SetPosition(position);
  }

  void OnEvent(Event& event) override {
    EventDispatcher dispatcher(event);
    dispatcher.Dispatch<KeyPressEvent>([this](KeyPressEvent& e) {
      pressed_keys_.insert(e.GetKeyCode());
      return false;
    });
    dispatcher.Dispatch<KeyReleaseEvent>([this](KeyReleaseEvent& e) {
      pressed_keys_.erase(e.GetKeyCode());
      return false;
    });
    dispatcher.Dispatch<MouseMoveEvent>([this](MouseMoveEvent& e) {
      if (!mouse_captured_) {
        return false;
      }
      if (first_mouse_) {
        last_mouse_x_ = e.GetX();
        last_mouse_y_ = e.GetY();
        first_mouse_ = false;
        return false;
      }
      // Invert Y: screen +y goes down, but positive pitch should look up.
      float dx = static_cast<float>(e.GetX() - last_mouse_x_);
      float dy = static_cast<float>(last_mouse_y_ - e.GetY());
      last_mouse_x_ = e.GetX();
      last_mouse_y_ = e.GetY();

      yaw_ -= dx * mouse_sensitivity_;
      pitch_ += dy * mouse_sensitivity_;
      pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
      ApplyRotation();
      return false;
    });
    dispatcher.Dispatch<MouseScrollEvent>([this](MouseScrollEvent& e) {
      float fov_deg = glm::degrees(camera_.GetFov()) - static_cast<float>(e.GetYOffset());
      fov_deg = std::clamp(fov_deg, 1.0f, 90.0f);
      camera_.SetFov(glm::radians(fov_deg));
      return false;
    });
  }

private:
  // Compose rotation as yaw-around-world-Y then pitch-around-local-X. No roll.
  void ApplyRotation() {
    glm::quat yaw_quat = glm::angleAxis(glm::radians(yaw_), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat pitch_quat = glm::angleAxis(glm::radians(pitch_), glm::vec3(1.0f, 0.0f, 0.0f));
    transform_.SetRotation(yaw_quat * pitch_quat);
  }

  void SyncAnglesFromTransform() {
    glm::vec3 forward = transform_.GetRotation() * glm::vec3(0.0f, 0.0f, -1.0f);
    pitch_ = glm::degrees(std::asin(std::clamp(forward.y, -1.0f, 1.0f)));
    yaw_ = glm::degrees(std::atan2(forward.x, -forward.z));
  }

  TransformComponent& transform_;
  CameraComponent& camera_;

  std::unordered_set<int> pressed_keys_;

  float movement_speed_ = 2.5f;
  float mouse_sensitivity_ = 0.1f;
  bool mouse_captured_ = true;
  bool first_mouse_ = true;
  double last_mouse_x_ = 0.0;
  double last_mouse_y_ = 0.0;

  float yaw_ = 0.0f;
  float pitch_ = 0.0f;
};

}  // namespace engine
