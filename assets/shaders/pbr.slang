// Simplified PBR: Cook-Torrance BRDF with 4 point lights, Reinhard tonemap,
// gamma 2.2. Material comes in via push constants (baseColor/metallic/roughness),
// no textures yet -- those land in the Loading_Models chapter with UV + tangent.

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4   light_positions[4];  // xyz = world-space position, w unused
    float4   light_colors[4];     // rgb = radiant intensity, a unused
    float4   cam_pos;             // xyz = world-space camera, w unused
    float    exposure;
    float    gamma;
    float    _pad0;
    float    _pad1;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

struct Material {
    float4 base_color;
    float  metallic;
    float  roughness;
    float  _pad0;
    float  _pad1;
};

[[vk::push_constant]] Material material;

struct VertexInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
};

struct VertexOutput {
    float4 pos         : SV_Position;
    float3 world_pos;
    float3 world_normal;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    float4 world = mul(ubo.model, float4(input.position, 1.0));
    output.world_pos = world.xyz;
    // Cube is uniformly scaled so model's upper-left 3x3 works directly as a
    // normal matrix. Swap to transpose(inverse(model)) once meshes get non-
    // uniform scale.
    output.world_normal = normalize(mul((float3x3)ubo.model, input.normal));
    output.pos = mul(ubo.proj, mul(ubo.view, world));
    return output;
}

static const float kPi = 3.14159265359;

float DistributionGGX(float3 n, float3 h, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(n, h), 0.0);
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (kPi * denom * denom);
}

float GeometrySchlickGGX(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

float GeometrySmith(float3 n, float3 v, float3 l, float roughness) {
    float n_dot_v = max(dot(n, v), 0.0);
    float n_dot_l = max(dot(n, l), 0.0);
    return GeometrySchlickGGX(n_dot_v, roughness) * GeometrySchlickGGX(n_dot_l, roughness);
}

float3 FresnelSchlick(float cos_theta, float3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target {
    float3 n = normalize(input.world_normal);
    float3 v = normalize(ubo.cam_pos.xyz - input.world_pos);

    // f0 is 0.04 for dielectrics, the base color for metals.
    float3 albedo   = material.base_color.rgb;
    float  metal    = material.metallic;
    float  rough    = material.roughness;
    float3 f0       = lerp(float3(0.04, 0.04, 0.04), albedo, metal);

    float3 lo = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        float3 light_pos = ubo.light_positions[i].xyz;
        float3 light_col = ubo.light_colors[i].rgb;

        float3 l = normalize(light_pos - input.world_pos);
        float3 h = normalize(v + l);
        float  distance    = length(light_pos - input.world_pos);
        float  attenuation = 1.0 / (distance * distance);
        float3 radiance    = light_col * attenuation;

        float  ndf = DistributionGGX(n, h, rough);
        float  g   = GeometrySmith(n, v, l, rough);
        float3 f   = FresnelSchlick(max(dot(h, v), 0.0), f0);

        float3 numerator   = ndf * g * f;
        float  denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001;
        float3 specular    = numerator / denominator;

        // ks = f, kd = (1 - ks) * (1 - metallic) — metals have no diffuse term.
        float3 ks = f;
        float3 kd = (float3(1.0, 1.0, 1.0) - ks) * (1.0 - metal);

        float n_dot_l = max(dot(n, l), 0.0);
        lo += (kd * albedo / kPi + specular) * radiance * n_dot_l;
    }

    // Tiny ambient term so back faces aren't pitch black before IBL lands.
    float3 ambient = float3(0.03, 0.03, 0.03) * albedo;
    float3 color   = ambient + lo;

    // Exposure + Reinhard tonemap, then gamma.
    color = color * ubo.exposure;
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, material.base_color.a);
}
