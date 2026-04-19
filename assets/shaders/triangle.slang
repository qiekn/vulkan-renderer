// Minimal triangle: 3 vertices hardcoded by SV_VertexID, no vertex buffer,
// no uniforms. Produces a colored triangle through dynamic rendering.

struct VertexOutput {
    float4 pos : SV_Position;
    float3 color;
};

static const float2 kPositions[3] = {
    float2( 0.0, -0.5),
    float2( 0.5,  0.5),
    float2(-0.5,  0.5),
};

static const float3 kColors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VertexOutput vertMain(uint vertex_id : SV_VertexID) {
    VertexOutput output;
    output.pos   = float4(kPositions[vertex_id], 0.0, 1.0);
    output.color = kColors[vertex_id];
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
