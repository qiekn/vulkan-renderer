// PBR shader for glTF models: Cook-Torrance BRDF, 4 point lights, textured
// base color / metallic-roughness / normal / occlusion / emissive. Material
// factors and per-map set flags come in via push constants; the fallback
// texture lets us sample every binding unconditionally even when a map is
// missing, and the flag tells the shader whether to actually use the result.

struct UniformBufferObject {
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
    float4 base_color_factor;
    float4 emissive_factor;       // xyz used
    float  metallic_factor;
    float  roughness_factor;
    int    base_color_tex_set;
    int    metallic_roughness_tex_set;
    int    normal_tex_set;
    int    occlusion_tex_set;
    int    emissive_tex_set;
    float  _pad0;
};

// One 128-byte push constant block: vertex stage uses `model`, fragment stage
// uses `material`. Vulkan's minimum push constant guarantee is 128 bytes, so
// this fits everywhere without reaching for larger ranges.
struct DrawPushConstants {
    float4x4 model;
    Material material;
};

[[vk::push_constant]] DrawPushConstants draw;

// Per-material set: five combined image-samplers in a fixed order. Bind every
// slot on the C++ side, substituting a 1x1 white fallback where the glTF
// material has no texture.
[[vk::binding(0, 1)]] Sampler2D base_color_map;
[[vk::binding(1, 1)]] Sampler2D metallic_roughness_map;
[[vk::binding(2, 1)]] Sampler2D normal_map;
[[vk::binding(3, 1)]] Sampler2D occlusion_map;
[[vk::binding(4, 1)]] Sampler2D emissive_map;

struct VertexInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 tangent  : TANGENT;
};

struct VertexOutput {
    float4 pos         : SV_Position;
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float4 world_tangent;  // xyz in world space, w = bitangent sign from glTF
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    float4 world = mul(draw.model, float4(input.position, 1.0));
    output.world_pos = world.xyz;
    // Upper-left 3x3 works as a normal matrix while scale stays uniform; once
    // nodes get non-uniform scale we'll want transpose(inverse(model)).
    float3x3 upper = (float3x3)draw.model;
    output.world_normal = normalize(mul(upper, input.normal));
    output.world_tangent = float4(normalize(mul(upper, input.tangent.xyz)), input.tangent.w);
    output.uv = input.uv;
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

// Reconstructs a world-space normal from the tangent-space normal map sample
// when the material actually supplies one; otherwise returns the interpolated
// vertex normal unchanged.
float3 GetNormal(VertexOutput input) {
    float3 n = normalize(input.world_normal);
    if (draw.material.normal_tex_set < 0) {
        return n;
    }
    float3 sampled = normal_map.Sample(input.uv).xyz * 2.0 - 1.0;
    float3 t = normalize(input.world_tangent.xyz);
    float3 b = normalize(cross(n, t)) * input.world_tangent.w;
    float3x3 tbn = float3x3(t, b, n);
    return normalize(mul(sampled, tbn));
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target {
    // Base color: sample + factor when the map is present, factor alone otherwise.
    float4 base_color = draw.material.base_color_factor;
    if (draw.material.base_color_tex_set >= 0) {
        base_color *= base_color_map.Sample(input.uv);
    }

    // glTF metallic-roughness convention: B = metallic, G = roughness.
    float metallic = draw.material.metallic_factor;
    float roughness = draw.material.roughness_factor;
    if (draw.material.metallic_roughness_tex_set >= 0) {
        float4 mr = metallic_roughness_map.Sample(input.uv);
        metallic *= mr.b;
        roughness *= mr.g;
    }

    float ao = 1.0;
    if (draw.material.occlusion_tex_set >= 0) {
        ao = occlusion_map.Sample(input.uv).r;
    }

    float3 emissive = draw.material.emissive_factor.rgb;
    if (draw.material.emissive_tex_set >= 0) {
        emissive *= emissive_map.Sample(input.uv).rgb;
    }

    float3 n = GetNormal(input);
    float3 v = normalize(ubo.cam_pos.xyz - input.world_pos);

    float3 albedo = base_color.rgb;
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 lo = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        float3 light_pos = ubo.light_positions[i].xyz;
        float3 light_col = ubo.light_colors[i].rgb;

        float3 l = normalize(light_pos - input.world_pos);
        float3 h = normalize(v + l);
        float  distance    = length(light_pos - input.world_pos);
        float  attenuation = 1.0 / (distance * distance);
        float3 radiance    = light_col * attenuation;

        float  ndf = DistributionGGX(n, h, roughness);
        float  g   = GeometrySmith(n, v, l, roughness);
        float3 f   = FresnelSchlick(max(dot(h, v), 0.0), f0);

        float3 numerator   = ndf * g * f;
        float  denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001;
        float3 specular    = numerator / denominator;

        float3 ks = f;
        float3 kd = (float3(1.0, 1.0, 1.0) - ks) * (1.0 - metallic);

        float n_dot_l = max(dot(n, l), 0.0);
        lo += (kd * albedo / kPi + specular) * radiance * n_dot_l;
    }

    float3 ambient = float3(0.03, 0.03, 0.03) * albedo * ao;
    float3 color   = ambient + lo + emissive;

    color = color * ubo.exposure;
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, base_color.a);
}
