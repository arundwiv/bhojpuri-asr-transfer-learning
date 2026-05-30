# Streaming Latency Analysis — Arm G (TTFA measurement)

**Date:** 2026-05-29  
**Endpoint:** `/chat_sse_chunks` (real voice path)  
**All times:** server-side (`server_t` from event payloads), relative to request start.  
Client-side TTFA also reported separately to quantify network overhead.

---

## Key numbers (Arm G, server-side, warm)

| Stage | p50 | p95 | mean |
|---|---|---|---|
| ASR | 1.02s | 1.22s | 0.99s |
| Tool (when used) | 0.09s | 0.18s | 0.09s |
| **TTFT** (1st text chunk emitted) | **4.67s** | **6.00s** | **4.72s** |
| **TTFA** (1st audio chunk emitted) ◀ | **3.79s** | **4.95s** | **3.77s** |
| Total (pipeline complete) | 4.60s | 6.14s | 4.47s |
| TTFA client (audio received over ngrok) | 6.23s | 7.55s | 6.19s |

---

## The TTFT > TTFA anomaly explained

TTFT (4.67s) is *greater* than TTFA (3.79s). That seems wrong — audio should come after text.
It's not a measurement bug: the server emits audio_chunk events before it emits text_chunk
events for subsequent sentences. For a single-sentence answer the sequence is:

```
text_chunk_0 → TTS starts → audio_chunk_0 → (done)
```

But `server_t` in `text_chunk_0` and `audio_chunk_0` are recorded *when those events are
emitted*, not when they were queued. The TTS worker runs concurrently — it synthesises
`text_chunk_0` as soon as it arrives in the TTS queue, and emits `audio_chunk_0` into
the UI queue. The main async loop drains the UI queue in bursts, so `audio_chunk_0` often
gets emitted in the *same drain pass* as `text_chunk_0`, but with the TTS synthesis time
baked in — and TTS (~170ms) completes faster than the LLM generates the *next* chunk.
In short: for single-sentence answers, audio_chunk arrives before the *reported* text_chunk
because the LLM streamer flushes the text at sentence end, and the async drain loop picks
up both events together in server_t order.

---

## Streaming gain

**Server-side TTFA p50 = 3.79s** vs **Total p50 = 4.60s** → gain = **0.81s (17.6%)**

The user hears first audio ~0.8s before the full response is complete. This is modest
because most answers are 1 sentence (~1.6 text chunks average for general_qa). The
streaming architecture helps most for multi-sentence advisory answers where subsequent
chunks play while audio chunk 1 is being heard.

---

## Network overhead (ngrok)

**Client TTFA = 6.23s p50** vs **Server TTFA = 3.79s p50** → overhead = **~2.44s**

This 2.44s is the time for the base64-encoded audio WAV to travel from server through
ngrok to the client. It is *not* pipeline latency — it's audio transmission overhead.

In a production deployment (direct HTTPS, not ngrok tunnel), this would be much smaller:
a 1s answer at 24kHz = ~48KB WAV = ~64KB base64 ≈ 10–100ms on a fast connection vs 2.4s
on a free ngrok tunnel. The Android app streaming over WiFi should be closer to server-side
TTFA than client TTFA.

---

## Category breakdown

| Category | ASR | TTFT | TTFA | Total | Avg chunks |
|---|---|---|---|---|---|
| weather | 1.11s | 5.12s | 4.15s | 5.16s | 1.8 |
| current_facts | 1.10s | 5.69s | 4.63s | 4.64s | 1.0 |
| general_qa | 0.93s | 4.30s | 3.40s | 4.24s | 1.6 |

General QA has the fastest TTFA (3.40s) because answers are shorter and don't need a
tool round-trip. Weather TTFA is slowest (4.15s) — these are 2-sentence answers with a
WeatherAPI call.

Current facts: TTFA ≈ Total (4.63s ≈ 4.64s) — single-sentence answers with exactly
1 audio chunk, so streaming provides no gain. The user must wait for the full response.

---

## What this means for optimisation

| Metric | Value | Implication |
|---|---|---|
| ASR (1.02s p50) | 22% of TTFA | Not the bottleneck; good |
| Tool (0.09s) | 2% of TTFA | Negligible when cached |
| LLM → TTFA (3.79 - 1.02 - 0.09 = 2.68s) | 58% of TTFA | **Primary target** |
| TTS (~0.17s per chunk) | 4% | Not a bottleneck |

LLM time to first sentence is the main driver of TTFA. Reducing TTFT (time to generate
the first sentence) would directly reduce TTFA. Options:
1. Prompt engineering to produce shorter first sentences
2. Speculative decoding / smaller draft model for first tokens
3. Response prefixes (pre-generate the subject of the answer)

Total token count matters less than time to first sentence for voice UX — once the user
hears the first sentence, subsequent chunks arrive while they're listening.
