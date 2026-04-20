// HRTF convolution compute shader. Each thread computes one output stereo frame
// by convolving the trailing 256 mono input samples with the per-ear IR. Output
// is interleaved (L, R, L, R, ...) so miniaudio's default stereo data source
// layout (ma_format_f32, 2 channels) can consume it directly.
//
// Dispatch: ceil(frame_count / 64) workgroups of 64 threads. Buffers are std430
// host-visible + coherent on the CPU side so we can skip transfer-queue staging
// for this first-pass implementation.

struct HrtfParams {
    uint frame_count;  // number of mono input frames = N
};

// Mono input PCM, N floats. RW so we can reuse the same binding layout for
// upload; shader only reads.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> input_mono;

// Interleaved stereo output PCM, 2*N floats.
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_stereo;

// HRTF impulse response pair, 512 floats: left taps [0..255] then right taps
// [256..511].
[[vk::binding(2, 0)]] RWStructuredBuffer<float> ir;

[[vk::push_constant]] HrtfParams params;

static const uint kIrTaps = 256u;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= params.frame_count) {
        return;
    }

    float left = 0.0;
    float right = 0.0;
    for (uint tap = 0u; tap < kIrTaps; ++tap) {
        int src = int(i) - int(tap);
        if (src < 0) break;
        float s = input_mono[src];
        left  += s * ir[tap];
        right += s * ir[kIrTaps + tap];
    }

    output_stereo[2u * i + 0u] = left;
    output_stereo[2u * i + 1u] = right;
}
