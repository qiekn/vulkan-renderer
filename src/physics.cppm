module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.physics;

import std;
import engine.scene;

namespace engine {

// ---------------------------------------------------------------------------: Collider

export enum class ColliderType { Box, Sphere };

// Base collider. SetOffset lets users nudge the collider relative to the body
// origin (e.g. a sphere whose center is above the body root). The concrete
// shapes live below and expose shape-specific parameters.
export class Collider {
 public:
  virtual ~Collider() = default;
  virtual ColliderType GetType() const = 0;

  void SetOffset(const glm::vec3& offset) { offset_ = offset; }
  const glm::vec3& GetOffset() const { return offset_; }

 protected:
  glm::vec3 offset_{0.0f};
};

export class SphereCollider : public Collider {
 public:
  explicit SphereCollider(float radius) : radius_(radius) {}
  ColliderType GetType() const override { return ColliderType::Sphere; }
  float GetRadius() const { return radius_; }
  void SetRadius(float r) { radius_ = r; }

 private:
  float radius_ = 0.5f;
};

// Axis-aligned box collider — body rotation is ignored on BoxColliders in ep8.
// Good enough for static ground planes and axis-aligned walls; OBB support is
// a follow-up (requires SAT in collision detection).
export class BoxCollider : public Collider {
 public:
  explicit BoxCollider(const glm::vec3& half_extents) : half_extents_(half_extents) {}
  ColliderType GetType() const override { return ColliderType::Box; }
  const glm::vec3& GetHalfExtents() const { return half_extents_; }
  void SetHalfExtents(const glm::vec3& he) { half_extents_ = he; }

 private:
  glm::vec3 half_extents_{0.5f};
};

// ---------------------------------------------------------------------------: RigidBody

// Rigid body state + forces. Position/rotation are authoritative during physics
// steps; the RigidBodyComponent pushes them onto the owning entity's
// TransformComponent each frame. Angular velocity is integrated (so
// ApplyTorqueImpulse produces visible spin) but collision resolution is
// linear-only — the tutorial does the same and we match.
export class RigidBody {
 public:
  RigidBody() { RebuildInertia(); }

  // --- pose + motion ------------------------------------------------------
  void SetPosition(const glm::vec3& p) { position_ = p; }
  void SetRotation(const glm::quat& r) { rotation_ = r; }
  void SetLinearVelocity(const glm::vec3& v) { linear_velocity_ = v; }
  void SetAngularVelocity(const glm::vec3& v) { angular_velocity_ = v; }

  const glm::vec3& GetPosition() const { return position_; }
  const glm::quat& GetRotation() const { return rotation_; }
  const glm::vec3& GetLinearVelocity() const { return linear_velocity_; }
  const glm::vec3& GetAngularVelocity() const { return angular_velocity_; }

  // --- material properties ------------------------------------------------
  void SetMass(float mass) {
    mass_ = mass;
    inverse_mass_ = mass > 0.0f ? 1.0f / mass : 0.0f;
    RebuildInertia();
  }
  float GetMass() const { return mass_; }
  float GetInverseMass() const { return inverse_mass_; }
  const glm::mat3& GetInverseInertia() const { return inverse_inertia_; }

  void SetRestitution(float e) { restitution_ = e; }
  float GetRestitution() const { return restitution_; }
  void SetFriction(float f) { friction_ = f; }
  float GetFriction() const { return friction_; }

  // --- collider -----------------------------------------------------------
  void SetCollider(std::unique_ptr<Collider> collider) {
    collider_ = std::move(collider);
    RebuildInertia();
  }
  Collider* GetCollider() const { return collider_.get(); }

  // --- forces -------------------------------------------------------------
  void ApplyForce(const glm::vec3& f) { accumulated_force_ += f; }
  void ApplyImpulse(const glm::vec3& j) {
    if (!is_kinematic_) {
      linear_velocity_ += j * inverse_mass_;
    }
  }
  void ApplyTorque(const glm::vec3& t) { accumulated_torque_ += t; }
  void ApplyTorqueImpulse(const glm::vec3& ti) {
    if (!is_kinematic_) {
      angular_velocity_ += inverse_inertia_ * ti;
    }
  }

  // --- flags --------------------------------------------------------------
  void SetKinematic(bool k) { is_kinematic_ = k; }
  bool IsKinematic() const { return is_kinematic_; }
  void SetGravityEnabled(bool g) { use_gravity_ = g; }
  bool IsGravityEnabled() const { return use_gravity_; }

 private:
  friend class PhysicsSystem;

  // Crude diagonal inertia tensor from current mass + collider. Sphere:
  // I = (2/5) m r². Box: I = (m/3)(he_y² + he_z², he_x² + he_z², he_x² + he_y²).
  // Good enough for a toy physics system; a proper engine would let users
  // provide a custom tensor.
  void RebuildInertia() {
    inverse_inertia_ = glm::mat3(0.0f);
    if (mass_ <= 0.0f || collider_ == nullptr) {
      return;
    }
    glm::mat3 inertia(0.0f);
    if (collider_->GetType() == ColliderType::Sphere) {
      auto* sphere = static_cast<SphereCollider*>(collider_.get());
      float i = 0.4f * mass_ * sphere->GetRadius() * sphere->GetRadius();
      inertia = glm::mat3(i);
    } else {
      auto* box = static_cast<BoxCollider*>(collider_.get());
      const glm::vec3 he = box->GetHalfExtents();
      float s = mass_ / 3.0f;
      inertia[0][0] = s * (he.y * he.y + he.z * he.z);
      inertia[1][1] = s * (he.x * he.x + he.z * he.z);
      inertia[2][2] = s * (he.x * he.x + he.y * he.y);
    }
    inverse_inertia_ = glm::inverse(inertia);
  }

  glm::vec3 position_{0.0f};
  glm::quat rotation_{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 linear_velocity_{0.0f};
  glm::vec3 angular_velocity_{0.0f};

  glm::vec3 accumulated_force_{0.0f};
  glm::vec3 accumulated_torque_{0.0f};

  float mass_ = 1.0f;
  float inverse_mass_ = 1.0f;
  glm::mat3 inverse_inertia_{0.0f};
  float restitution_ = 0.3f;
  float friction_ = 0.5f;

  std::unique_ptr<Collider> collider_;
  bool is_kinematic_ = false;
  bool use_gravity_ = true;
};

// ---------------------------------------------------------------------------: CollisionInfo

// Populated by PhysicsSystem's per-pair CheckCollision. body_a / body_b are
// non-owning — they point into PhysicsSystem::bodies_. Normal points from A
// toward B (so A gets pushed opposite, B along normal).
export struct CollisionInfo {
  RigidBody* body_a = nullptr;
  RigidBody* body_b = nullptr;
  glm::vec3 contact_point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  float penetration_depth = 0.0f;
};

// ---------------------------------------------------------------------------: PhysicsSystem

// CPU rigid-body simulation. One step: accumulate forces → integrate forces →
// detect collisions (naive O(n²)) → resolve collisions (impulse + positional
// correction, linear-only) → integrate velocities → clear forces. Outer
// Update() drives fixed 1/60 s steps via an accumulator so sim is framerate-
// independent.
export class PhysicsSystem {
 public:
  PhysicsSystem() = default;
  ~PhysicsSystem() = default;

  PhysicsSystem(const PhysicsSystem&) = delete;
  PhysicsSystem& operator=(const PhysicsSystem&) = delete;
  PhysicsSystem(PhysicsSystem&&) = delete;
  PhysicsSystem& operator=(PhysicsSystem&&) = delete;

  // Returns a non-owning pointer. Caller attaches a collider via body->SetCollider.
  RigidBody* CreateRigidBody() {
    bodies_.push_back(std::make_unique<RigidBody>());
    return bodies_.back().get();
  }

  // Removes a body by address. Linear scan — body counts stay small in demos.
  void DestroyRigidBody(RigidBody* body) {
    auto it = std::ranges::find_if(
        bodies_, [body](const auto& up) { return up.get() == body; });
    if (it != bodies_.end()) {
      bodies_.erase(it);
    }
  }

  void SetGravity(const glm::vec3& g) { gravity_ = g; }
  const glm::vec3& GetGravity() const { return gravity_; }

  std::size_t GetBodyCount() const { return bodies_.size(); }

  // Main loop hook. Caps the accumulated delta at 0.25 s so breakpoints or
  // long stalls don't unwind into hundreds of fixed steps at once.
  void Update(float delta_time) {
    constexpr float kFixedStep = 1.0f / 60.0f;
    constexpr float kMaxAccum = 0.25f;
    accumulator_ += std::min(delta_time, kMaxAccum);
    while (accumulator_ >= kFixedStep) {
      Step(kFixedStep);
      accumulator_ -= kFixedStep;
    }
  }

 private:
  void Step(float dt) {
    AccumulateForces();
    IntegrateForces(dt);
    std::vector<CollisionInfo> collisions;
    DetectCollisions(collisions);
    ResolveCollisions(collisions);
    IntegrateVelocities(dt);
    ClearForces();
  }

  void AccumulateForces() {
    for (auto& body : bodies_) {
      if (body->is_kinematic_ || !body->use_gravity_) {
        continue;
      }
      body->accumulated_force_ += gravity_ * body->mass_;
    }
  }

  void IntegrateForces(float dt) {
    constexpr float kLinearDamping = 0.01f;
    constexpr float kAngularDamping = 0.01f;
    for (auto& body : bodies_) {
      if (body->is_kinematic_) {
        continue;
      }
      body->linear_velocity_ += body->accumulated_force_ * body->inverse_mass_ * dt;
      body->angular_velocity_ += body->inverse_inertia_ * body->accumulated_torque_ * dt;
      body->linear_velocity_ *= (1.0f - kLinearDamping);
      body->angular_velocity_ *= (1.0f - kAngularDamping);
    }
  }

  void DetectCollisions(std::vector<CollisionInfo>& out) {
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
      for (std::size_t j = i + 1; j < bodies_.size(); ++j) {
        RigidBody& a = *bodies_[i];
        RigidBody& b = *bodies_[j];
        if (a.is_kinematic_ && b.is_kinematic_) {
          continue;
        }
        if (a.collider_ == nullptr || b.collider_ == nullptr) {
          continue;
        }
        CollisionInfo info;
        if (CheckCollision(a, b, info)) {
          info.body_a = &a;
          info.body_b = &b;
          out.push_back(info);
        }
      }
    }
  }

  void ResolveCollisions(const std::vector<CollisionInfo>& collisions) {
    for (const auto& c : collisions) {
      RigidBody& a = *c.body_a;
      RigidBody& b = *c.body_b;
      float inv_a = a.is_kinematic_ ? 0.0f : a.inverse_mass_;
      float inv_b = b.is_kinematic_ ? 0.0f : b.inverse_mass_;
      float inv_sum = inv_a + inv_b;
      if (inv_sum <= 0.0f) {
        continue;
      }

      glm::vec3 rel_vel = b.linear_velocity_ - a.linear_velocity_;
      float vel_along_normal = glm::dot(rel_vel, c.normal);
      if (vel_along_normal > 0.0f) {
        continue;  // separating — no impulse
      }

      float e = std::min(a.restitution_, b.restitution_);
      float j = -(1.0f + e) * vel_along_normal / inv_sum;
      glm::vec3 impulse = c.normal * j;
      if (!a.is_kinematic_) {
        a.linear_velocity_ -= impulse * inv_a;
      }
      if (!b.is_kinematic_) {
        b.linear_velocity_ += impulse * inv_b;
      }

      // Positional correction: pushes bodies out of penetration so they don't
      // sink through each other. `slop` avoids jitter at rest.
      constexpr float kCorrectPercent = 0.2f;
      constexpr float kSlop = 0.01f;
      float pen = std::max(c.penetration_depth - kSlop, 0.0f);
      glm::vec3 correction = (pen / inv_sum) * kCorrectPercent * c.normal;
      if (!a.is_kinematic_) {
        a.position_ -= correction * inv_a;
      }
      if (!b.is_kinematic_) {
        b.position_ += correction * inv_b;
      }
    }
  }

  void IntegrateVelocities(float dt) {
    for (auto& body : bodies_) {
      if (body->is_kinematic_) {
        continue;
      }
      body->position_ += body->linear_velocity_ * dt;
      // Quaternion derivative q̇ = ½ ω q. Normalize each step to keep drift
      // bounded (tutorial pattern).
      glm::quat omega_q(0.0f, body->angular_velocity_.x, body->angular_velocity_.y,
                        body->angular_velocity_.z);
      body->rotation_ += (omega_q * body->rotation_) * (0.5f * dt);
      body->rotation_ = glm::normalize(body->rotation_);
    }
  }

  void ClearForces() {
    for (auto& body : bodies_) {
      body->accumulated_force_ = glm::vec3(0.0f);
      body->accumulated_torque_ = glm::vec3(0.0f);
    }
  }

  bool CheckCollision(RigidBody& a, RigidBody& b, CollisionInfo& out) {
    auto* ca = a.collider_.get();
    auto* cb = b.collider_.get();
    if (ca->GetType() == ColliderType::Sphere && cb->GetType() == ColliderType::Sphere) {
      return SphereVsSphere(a, b, out);
    }
    if (ca->GetType() == ColliderType::Sphere && cb->GetType() == ColliderType::Box) {
      return SphereVsAabb(a, b, out);
    }
    if (ca->GetType() == ColliderType::Box && cb->GetType() == ColliderType::Sphere) {
      bool hit = SphereVsAabb(b, a, out);
      if (hit) {
        // Normal was computed from sphere toward box center; our contract is
        // normal goes a→b, so flip since we swapped roles.
        out.normal = -out.normal;
      }
      return hit;
    }
    // Box×Box not implemented — scoped out of ep8.
    return false;
  }

  bool SphereVsSphere(RigidBody& a, RigidBody& b, CollisionInfo& out) {
    auto* sa = static_cast<SphereCollider*>(a.collider_.get());
    auto* sb = static_cast<SphereCollider*>(b.collider_.get());
    glm::vec3 pa = a.position_ + sa->GetOffset();
    glm::vec3 pb = b.position_ + sb->GetOffset();
    float ra = sa->GetRadius();
    float rb = sb->GetRadius();
    glm::vec3 diff = pb - pa;
    float dist_sq = glm::dot(diff, diff);
    float min_dist = ra + rb;
    if (dist_sq >= min_dist * min_dist) {
      return false;
    }
    float dist = std::sqrt(dist_sq);
    glm::vec3 normal = dist > 1e-5f ? diff / dist : glm::vec3(0.0f, 1.0f, 0.0f);
    out.normal = normal;
    out.contact_point = pa + normal * ra;
    out.penetration_depth = min_dist - dist;
    return true;
  }

  // Sphere at `sphere` vs AABB at `box`. Assumes box rotation is identity —
  // documented as a BoxCollider limitation above.
  bool SphereVsAabb(RigidBody& sphere, RigidBody& box, CollisionInfo& out) {
    auto* s = static_cast<SphereCollider*>(sphere.collider_.get());
    auto* b = static_cast<BoxCollider*>(box.collider_.get());
    glm::vec3 sp = sphere.position_ + s->GetOffset();
    glm::vec3 bp = box.position_ + b->GetOffset();
    glm::vec3 he = b->GetHalfExtents();
    float radius = s->GetRadius();

    glm::vec3 local = sp - bp;
    glm::vec3 clamped = glm::clamp(local, -he, he);
    glm::vec3 diff = local - clamped;
    float dist_sq = glm::dot(diff, diff);
    if (dist_sq >= radius * radius) {
      return false;
    }

    if (dist_sq > 1e-6f) {
      float dist = std::sqrt(dist_sq);
      out.normal = diff / dist;
      out.penetration_depth = radius - dist;
      out.contact_point = bp + clamped;
    } else {
      // Sphere center inside the box — pick the nearest face to push out on.
      glm::vec3 face_dist = he - glm::abs(local);  // +ve = distance to face
      int axis = 0;
      if (face_dist.y < face_dist[axis]) axis = 1;
      if (face_dist.z < face_dist[axis]) axis = 2;
      glm::vec3 n(0.0f);
      n[axis] = local[axis] >= 0.0f ? 1.0f : -1.0f;
      out.normal = n;
      out.penetration_depth = radius + face_dist[axis];
      out.contact_point = sp;
    }
    return true;
  }

  std::vector<std::unique_ptr<RigidBody>> bodies_;
  glm::vec3 gravity_{0.0f, -9.81f, 0.0f};
  float accumulator_ = 0.0f;
};

// ---------------------------------------------------------------------------: RigidBodyComponent

// Bridges a physics body into the scene graph: each frame writes body pose
// onto the owning entity's TransformComponent. Flow is one-way (physics →
// transform); mutate physics via ApplyForce/Impulse on the underlying body,
// never by moving the transform directly.
export class RigidBodyComponent : public Component {
 public:
  explicit RigidBodyComponent(RigidBody* body) : body_(body) {}

  void Update(float /*delta_time*/) override {
    auto* transform = GetOwner()->GetComponent<TransformComponent>();
    if (transform == nullptr || body_ == nullptr) {
      return;
    }
    transform->SetPosition(body_->GetPosition());
    transform->SetRotation(body_->GetRotation());
  }

  RigidBody* Get() const { return body_; }

 private:
  RigidBody* body_ = nullptr;
};

}  // namespace engine
