// GPU rigid-body step. Each thread advances one body: accumulate gravity into
// the external-force channel, integrate linear + angular velocity, damp, then
// integrate pose. After integration we resolve the only collision the GPU path
// knows about — sphere vs. a horizontal ground plane at y = ground_y. Sphere-
// vs-sphere and sphere-vs-oriented-box are skipped; on the GPU path the demo
// helmets never interact with each other in mid-air, only with the ground.
// Full broad + narrow + resolve shaders are a follow-up (see
// docs/Subsystems/05_vulkan_physics.adoc).
//
// Layout matches engine::GpuPhysicsData (128 bytes per body, std430).

struct PhysicsData {
    float4 position;          // xyz, w = inv_mass
    float4 rotation;          // quaternion (x, y, z, w)
    float4 linear_velocity;   // xyz, w = restitution
    float4 angular_velocity;  // xyz, w = friction
    float4 force;             // xyz = external force, w = is_kinematic flag
    float4 torque;            // xyz = external torque, w = use_gravity flag
    float4 collider;          // sphere: (radius, 0, 0, type); box: (he, type)
    float4 collider2;         // collider offset xyz
};

struct Params {
    float dt;
    float ground_y;           // world-space y of the ground plane (top surface)
    uint body_count;
    float _pad0;
    float4 gravity;           // xyz = gravity, w unused. vec4 sidesteps any
                              // vec3-padding disagreement between the C++ host
                              // struct and the shader's std430 push-constant
                              // layout.
};

[[vk::binding(0, 0)]] RWStructuredBuffer<PhysicsData> bodies;
[[vk::push_constant]] Params params;

float4 quat_mul(float4 a, float4 b) {
    return float4(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz)
    );
}

float4 quat_normalize(float4 q) {
    float len = length(q);
    return len > 1e-6 ? q / len : float4(0, 0, 0, 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void stepMain(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= params.body_count) return;

    PhysicsData body = bodies[i];

    // Kinematic bodies integrate nothing — host keeps them pinned.
    if (body.force.w > 0.5) return;

    float inv_mass = body.position.w;
    float restitution = body.linear_velocity.w;
    bool use_gravity = body.torque.w > 0.5;

    float3 pos = body.position.xyz;
    float4 rot = body.rotation;
    float3 lin_vel = body.linear_velocity.xyz;
    float3 ang_vel = body.angular_velocity.xyz;
    float3 f = body.force.xyz;
    float3 t = body.torque.xyz;

    // Integrate linear: gravity is acceleration, so it maps directly onto vel.
    // External forces still go through mass (v += F/m * dt).
    if (inv_mass > 0.0) {
        lin_vel += f * inv_mass * params.dt;
        if (use_gravity) {
            lin_vel += params.gravity.xyz * params.dt;
        }
    }

    // Integrate angular: approximate sphere inertia tensor I = (2/5) m r², so
    // inv(I) = (5/2) / (m r²) = 2.5 * inv_mass / r². Boxes skip torque on GPU.
    int type = int(body.collider.w);
    if (type == 1 /* sphere */) {
        float r = body.collider.x;
        if (r > 1e-4 && inv_mass > 0.0) {
            float inv_inertia = 2.5 * inv_mass / (r * r);
            ang_vel += t * inv_inertia * params.dt;
        }
    }

    // Damping — matches CPU PhysicsSystem::IntegrateForces constants.
    const float k_lin_damp = 0.01;
    const float k_ang_damp = 0.01;
    lin_vel *= 1.0 - k_lin_damp;
    ang_vel *= 1.0 - k_ang_damp;

    // Integrate pose.
    pos += lin_vel * params.dt;
    float4 omega_q = float4(ang_vel * 0.5, 0.0);
    rot = quat_normalize(rot + quat_mul(omega_q, rot) * params.dt);

    // Sphere vs ground plane (normal = +Y). Clamp to surface and reflect vy
    // using restitution with a rest-speed threshold so bodies settle instead of
    // micro-bouncing — same trick as PhysicsSystem::ResolveCollisions.
    if (type == 1) {
        float radius = body.collider.x;
        float d = pos.y - params.ground_y;
        if (d < radius) {
            pos.y = params.ground_y + radius;
            if (lin_vel.y < 0.0) {
                const float rest_speed_threshold = 1.0;
                float e = (-lin_vel.y) < rest_speed_threshold ? 0.0 : restitution;
                lin_vel.y = -lin_vel.y * e;
            }
        }
    }

    // Write back — clear external force/torque (GPU analogue of ClearForces)
    // while preserving the kinematic / use-gravity flags in the w channels.
    body.position = float4(pos, inv_mass);
    body.rotation = rot;
    body.linear_velocity = float4(lin_vel, restitution);
    body.angular_velocity = float4(ang_vel, body.angular_velocity.w);
    body.force = float4(0.0, 0.0, 0.0, body.force.w);
    body.torque = float4(0.0, 0.0, 0.0, body.torque.w);
    bodies[i] = body;
}
