// MVP-transformed triangle: 3 vertices hardcoded in world space via SV_VertexID.
// Vertex positions are fed through the per-frame model/view/proj UBO, so moving
// the camera or rotating the model changes what we see.

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

struct VertexOutput {
    float4 pos : SV_Position;
    float3 color;
};

static const float3 kPositions[3] = {
    float3( 0.0,  0.5, 0.0),
    float3( 0.5, -0.5, 0.0),
    float3(-0.5, -0.5, 0.0),
};

static const float3 kColors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VertexOutput vertMain(uint vertex_id : SV_VertexID) {
    VertexOutput output;
    float4 world = mul(ubo.model, float4(kPositions[vertex_id], 1.0));
    float4 view  = mul(ubo.view, world);
    output.pos   = mul(ubo.proj, view);
    output.color = kColors[vertex_id];
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
