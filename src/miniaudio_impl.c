// Translation unit that stamps out miniaudio's heavy implementation exactly
// once, with OGG Vorbis support bolted on. Upstream's miniaudio.c ships
// without Vorbis, so this TU does the canonical stb_vorbis dance:
//
//   1. Include stb_vorbis.c as header-only — gets the decls, no body, no `C`
//      macro yet.
//   2. Pull in miniaudio's implementation; it references the decls and also
//      includes <windows.h> (whose winnt.h uses an unparenthesized `C`
//      identifier that would clash with stb_vorbis's `#define C`).
//   3. Include stb_vorbis.c again for its implementation — by now windows.h
//      is done, so the macro no longer collides.
#define STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "extras/stb_vorbis.c"
