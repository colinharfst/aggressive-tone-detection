/*
 * AggressionClassifier.ino  — v2 (Feature-Parity Edition)
 * Arduino Nano 33 BLE Sense Rev2
 * Compatible with Arduino_TensorFlowLite 2.4.0-ALPHA
 *
 * ── FEATURE-PARITY FIX ───────────────────────────────────────────────────────
 *   Prior bug: librosa defaulted to Slaney norm + non-HTK scale, producing
 *   different mel filter weights than this sketch. On-device predictions were
 *   based on features the model had never seen during training.
 *
 *   This sketch implements the EXACT feature pipeline as the unified notebook:
 *     • HTK scale   : mel = 2595 * log10(1 + hz/700)  ← hz_to_mel() below
 *     • Plain triangular filters (NOT area-normalised)  ← init_mel_filterbank()
 *     • power_to_db ref=global_max, eps=1e-10, floor=-80 dB
 *     • Per-clip z-score normalisation
 *
 *   Training notebook (aggression_classifier_unified.ipynb) forces:
 *     librosa.feature.melspectrogram(..., htk=True, norm=None)
 *   which exactly matches this sketch's mel filterbank arithmetic.
 *
 * ── PIPELINE ─────────────────────────────────────────────────────────────────
 *   PDM mic (16 kHz, mono)
 *     → 20 000-sample ring buffer (16 000 clip + 4 000 hop)
 *        → Hann-windowed frames, HOP_LENGTH=512
 *           → Cooley-Tukey FFT (N_FFT=2048, self-contained)
 *              → One-sided power spectrum |X[k]|²
 *                 → HTK triangular mel filterbank (N_MELS=64)
 *                    → 10·log₁₀(E/max_E + 1e-10), floor=-80 dB
 *                       → Per-clip z-score normalisation
 *                          → int8 TFLite Micro CNN
 *                             → Sliding-window majority vote (2-frame buffer)
 *
 * ── HOW TO DEPLOY YOUR MODEL ─────────────────────────────────────────────────
 *   1. Run aggression_classifier_unified.ipynb end-to-end.
 *   2. Copy the generated model_data.h into this sketch folder.
 *   3. Set TFLITE_ARENA_SIZE to the value printed by the notebook.
 *   4. Compile for: Arduino Nano 33 BLE Sense (Mbed OS).
 *   5. Flash and open Serial Monitor at 115200 baud.
 *   6. Watch the "mel min/max/mean" diagnostic: values should be close to
 *      [-2, 2] after z-score. If the range is very different, check that
 *      your training notebook uses htk=True and norm=None.
 *
 * ── DEPENDENCIES ─────────────────────────────────────────────────────────────
 *   Arduino_TensorFlowLite 2.4.0-ALPHA  (Boards Manager)
 *   PDM                                 (bundled with Mbed core)
 *
 * ── MEMORY BUDGET (Nano 33 BLE Sense Rev2: 256 KB SRAM) ─────────────────────
 *   Ring buffer   20 000 × 2 B          =  40.0 KB
 *   FFT scratch    2 048 × 4 B × 2      =  16.0 KB  (real + imag)
 *   Power spectrum 1 025 × 4 B          =   4.1 KB
 *   Hann window    2 048 × 4 B          =   8.0 KB
 *   Mel matrix    64 × 32 × 4 B         =   8.2 KB
 *   TFLite arena   configurable         = 32–96 KB
 *   ─────────────────────────────────────────────
 *   Subtotal (excl. arena)              ~ 76 KB
 *   Remaining for arena + stack         ~180 KB
 */

#include "model_data.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <PDM.h>
#include <Arduino.h>
#include <math.h>

// =============================================================================
//  CONFIGURATION  — must match aggression_classifier_unified.ipynb exactly
// =============================================================================

static constexpr int   SAMPLE_RATE        = 16000;
static constexpr int   CLIP_SAMPLES       = 16000;   // 1 second
static constexpr int   N_FFT              = 2048;    // window length (128 ms)
static constexpr int   HOP_LENGTH         = 512;     // hop (32 ms)
static constexpr int   N_MELS             = 64;
static constexpr float HZ_LOW             = 0.0f;
static constexpr float HZ_HIGH            = 8000.0f;
static constexpr int   N_FRAMES           = 1 + (CLIP_SAMPLES / HOP_LENGTH); // 32

// Decision and voting
static constexpr float DECISION_THRESHOLD = 0.49f;
static constexpr int   VOTE_WINDOW        = 2;       // frames to average
static constexpr int   STRIDE_SAMPLES     = 4000;    // 250 ms inference stride

// Arena — increase in steps of 8192 if AllocateTensors() fails.
// The notebook prints the recommended value after quantisation.
static constexpr int   TFLITE_ARENA_SIZE  = 65536;   // 80 KB default (notebook requested 147456)

static constexpr int   ALERT_PIN          = LED_BUILTIN;
static constexpr int   SERIAL_BAUD        = 115200;

// =============================================================================
//  RING BUFFER
// =============================================================================

static constexpr int RING_LEN   = CLIP_SAMPLES + STRIDE_SAMPLES; // 20 000

static int16_t       g_ring[RING_LEN];
static volatile int  g_ring_write  = 0;
static int           g_ring_read   = 0;
static volatile int  g_new_samples = 0;

// =============================================================================
//  FEATURE BUFFERS
// =============================================================================

static float g_mel  [N_MELS][N_FRAMES];
static float g_hann [N_FFT];
static float g_fft_real[N_FFT];
static float g_fft_imag[N_FFT];
static float g_power[N_FFT / 2 + 1];

// =============================================================================
//  HTK MEL FILTERBANK
//  ─────────────────────────────────────────────────────────────────────────────
//  HTK formula:  mel = 2595 * log10(1 + hz / 700)
//  inverse:      hz  = 700 * (10^(mel/2595) - 1)
//
//  CRITICAL: plain triangular filters (NOT area-normalised).
//  This exactly matches librosa.feature.melspectrogram(..., htk=True, norm=None)
//  used in the training notebook.
// =============================================================================

static constexpr int MAX_FILTER_BINS = 200;

struct MelFilter {
    uint16_t start_bin;
    uint16_t n_bins;
    float    weights[MAX_FILTER_BINS];
};

static MelFilter g_mel_filters[N_MELS];

// HTK mel scale — matches Python: librosa.hz_to_mel(hz, htk=True)
static float hz_to_mel_htk(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz_htk(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void init_mel_filterbank() {
    const int fft_bins = N_FFT / 2 + 1;

    // N_MELS + 2 equally-spaced mel-scale points from HZ_LOW to HZ_HIGH
    float mel_lo = hz_to_mel_htk(HZ_LOW);
    float mel_hi = hz_to_mel_htk(HZ_HIGH);

    // Centre frequencies in mel domain
    float mel_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_pts[i] = mel_lo + (mel_hi - mel_lo) * i / (float)(N_MELS + 1);

    // Map mel centre frequencies to FFT bin indices
    // bin = floor((N_FFT + 1) * hz / sample_rate)  — same as librosa
    int bin_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        float hz   = mel_to_hz_htk(mel_pts[i]);
        bin_pts[i] = (int)floorf((float)(N_FFT + 1) * hz / (float)SAMPLE_RATE);
        if (bin_pts[i] < 0)          bin_pts[i] = 0;
        if (bin_pts[i] >= fft_bins)  bin_pts[i] = fft_bins - 1;
    }

    // Build triangular filters (plain — no area normalisation)
    for (int m = 0; m < N_MELS; m++) {
        int left   = bin_pts[m];
        int centre = bin_pts[m + 1];
        int right  = bin_pts[m + 2];

        g_mel_filters[m].start_bin = (uint16_t)left;
        int nb = right - left + 1;
        if (nb > MAX_FILTER_BINS) nb = MAX_FILTER_BINS;
        g_mel_filters[m].n_bins = (uint16_t)nb;

        for (int k = left; k <= right && (k - left) < MAX_FILTER_BINS; k++) {
            float w = 0.0f;
            if (k <= centre && centre > left)
                w = (float)(k - left)  / (float)(centre - left);
            else if (k > centre && right > centre)
                w = (float)(right - k) / (float)(right  - centre);
            g_mel_filters[m].weights[k - left] = w;
        }
    }
}

// =============================================================================
//  HANN WINDOW
// =============================================================================

static void init_hann_window() {
    // Symmetric Hann: w[n] = 0.5 * (1 - cos(2π n / (N-1)))
    for (int n = 0; n < N_FFT; n++)
        g_hann[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * n / (N_FFT - 1)));
}

// =============================================================================
//  COOLEY-TUKEY FFT  (self-contained, no external library)
//  Operates in-place on g_fft_real[] and g_fft_imag[].
// =============================================================================

static inline void swap_f(float &a, float &b) { float t = a; a = b; b = t; }

static void compute_fft() {
    const int n = N_FFT;

    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            swap_f(g_fft_real[i], g_fft_real[j]);
            swap_f(g_fft_imag[i], g_fft_imag[j]);
        }
    }

    // Iterative Cooley-Tukey butterfly
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / (float)len;
        const float wr  = cosf(ang);
        const float wi  = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float ur = g_fft_real[i + k];
                float ui = g_fft_imag[i + k];
                float vr = g_fft_real[i + k + len/2] * cr
                         - g_fft_imag[i + k + len/2] * ci;
                float vi = g_fft_real[i + k + len/2] * ci
                         + g_fft_imag[i + k + len/2] * cr;
                g_fft_real[i + k]         = ur + vr;
                g_fft_imag[i + k]         = ui + vi;
                g_fft_real[i + k + len/2] = ur - vr;
                g_fft_imag[i + k + len/2] = ui - vi;
                float ncr = cr * wr - ci * wi;
                ci        = cr * wi + ci * wr;
                cr        = ncr;
            }
        }
    }
}

// =============================================================================
//  FEATURE EXTRACTION
//  Produces g_mel[N_MELS][N_FRAMES] — a z-score normalised log-mel spectrogram
//  using HTK scale and plain triangular filters (matches the training notebook).
// =============================================================================

static void extract_features() {
    float global_max_power = 1e-10f;

    // ── Step 1: Mel filterbank for every hop frame ──────────────────────────
    for (int t = 0; t < N_FRAMES; t++) {
        int frame_start = t * HOP_LENGTH;

        // Build Hann-windowed frame from ring buffer
        for (int n = 0; n < N_FFT; n++) {
            float sample = 0.0f;
            int clip_idx = frame_start + n;
            if (clip_idx < CLIP_SAMPLES) {
                int ring_idx = (g_ring_read + clip_idx) % RING_LEN;
                sample = (float)g_ring[ring_idx] / 32768.0f;
            }
            g_fft_real[n] = sample * g_hann[n];
            g_fft_imag[n] = 0.0f;
        }

        // FFT
        compute_fft();

        // One-sided power spectrum |X[k]|²
        for (int k = 0; k <= N_FFT / 2; k++)
            g_power[k] = g_fft_real[k] * g_fft_real[k]
                       + g_fft_imag[k] * g_fft_imag[k];

        // Mel filterbank (plain triangular weights, HTK scale)
        for (int m = 0; m < N_MELS; m++) {
            float energy = 0.0f;
            int   start  = g_mel_filters[m].start_bin;
            int   nb     = g_mel_filters[m].n_bins;
            for (int i = 0; i < nb; i++)
                energy += g_power[start + i] * g_mel_filters[m].weights[i];
            g_mel[m][t] = energy;
            if (energy > global_max_power) global_max_power = energy;
        }
    }

    // ── Step 2: power_to_db(mel, ref=global_max) ───────────────────────────
    // 10 * log10(E / max_E + eps), hard floor at -80 dB
    // Matches: librosa.power_to_db(mel, ref=np.max)
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < N_FRAMES; t++) {
            float db = 10.0f * log10f(g_mel[m][t] / global_max_power + 1e-10f);
            if (db < -80.0f) db = -80.0f;
            g_mel[m][t] = db;
        }

    // ── Step 3: per-clip z-score ─────────────────────────────────────────────
    // Matches: (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    float sum = 0.0f, sum_sq = 0.0f;
    const int n_total = N_MELS * N_FRAMES;
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < N_FRAMES; t++) {
            sum    += g_mel[m][t];
            sum_sq += g_mel[m][t] * g_mel[m][t];
        }
    float mean    = sum / (float)n_total;
    float var     = sum_sq / (float)n_total - mean * mean;
    float inv_std = (var > 0.0f) ? (1.0f / (sqrtf(var) + 1e-6f)) : 1.0f;
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < N_FRAMES; t++)
            g_mel[m][t] = (g_mel[m][t] - mean) * inv_std;
}

// =============================================================================
//  INPUT TENSOR FILL
//  Keras NHWC layout: index = m * N_FRAMES + t  (batch=1, channels=1)
// =============================================================================

static float g_inp_scale = 1.0f;
static int   g_inp_zp    = 0;

static void fill_input_tensor(int8_t* data) {
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < N_FRAMES; t++) {
            float q = g_mel[m][t] / g_inp_scale + (float)g_inp_zp;
            if      (q >  127.0f) q =  127.0f;
            else if (q < -128.0f) q = -128.0f;
            data[m * N_FRAMES + t] = (int8_t)roundf(q);
        }
}

// =============================================================================
//  SLIDING-WINDOW VOTE
// =============================================================================

static float g_vote_buf[VOTE_WINDOW] = {};
static int   g_vote_head             = 0;

static bool update_vote(float p_agg) {
    g_vote_buf[g_vote_head] = p_agg;
    g_vote_head = (g_vote_head + 1) % VOTE_WINDOW;
    float sum = 0.0f;
    for (int i = 0; i < VOTE_WINDOW; i++) sum += g_vote_buf[i];
    return (sum / VOTE_WINDOW) >= DECISION_THRESHOLD;
}

// =============================================================================
//  TFLITE MICRO  (2.4.0-ALPHA API)
// =============================================================================

static uint8_t                    g_arena[TFLITE_ARENA_SIZE];
static tflite::MicroErrorReporter g_error_reporter;
static tflite::AllOpsResolver     g_resolver;
static const tflite::Model*       g_tflite_model = nullptr;
static tflite::MicroInterpreter*  g_interpreter  = nullptr;
static TfLiteTensor*              g_input        = nullptr;
static TfLiteTensor*              g_output       = nullptr;

// =============================================================================
//  INFERENCE
//  Returns P(aggressive) ∈ [0, 1].
// =============================================================================

static float run_inference() {
    if (!g_interpreter) return 0.0f;

    fill_input_tensor(g_input->data.int8);

    TfLiteStatus status = g_interpreter->Invoke();
    if (status != kTfLiteOk) {
        Serial.println("[ERROR] Invoke() failed");
        return 0.0f;
    }

    // Dequantise int8 output → probabilities
    float out_scale = g_output->params.scale;
    int   out_zp    = g_output->params.zero_point;
    float p_neutral    = (float)(g_output->data.int8[0] - out_zp) * out_scale;
    float p_aggressive = (float)(g_output->data.int8[1] - out_zp) * out_scale;

    // Normalise to handle quantisation rounding
    float total = p_neutral + p_aggressive + 1e-7f;
    return p_aggressive / total;
}

// =============================================================================
//  PDM CALLBACK  (ISR — minimal work)
// =============================================================================

static void on_pdm_data() {
    int avail = PDM.available();
    while (avail >= 2) {
        int16_t sample;
        PDM.read(&sample, 2);
        g_ring[g_ring_write] = sample;
        g_ring_write = (g_ring_write + 1) % RING_LEN;
        g_new_samples++;
        avail -= 2;
    }
}

// =============================================================================
//  SETUP
// =============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial && millis() < 4000) {}

    Serial.println();
    Serial.println("╔══════════════════════════════════════════════════════╗");
    Serial.println("║       AggressionClassifier v2 — Feature-Parity      ║");
    Serial.println("║  HTK mel scale · plain triangular · per-clip z-score ║");
    Serial.println("╚══════════════════════════════════════════════════════╝");
    Serial.println();

    pinMode(ALERT_PIN, OUTPUT);
    digitalWrite(ALERT_PIN, LOW);

    // ── Build mel filterbank (HTK) ──────────────────────────────────────────
    init_mel_filterbank();
    init_hann_window();

    Serial.println("[CONFIG] Feature pipeline parameters:");
    Serial.print  ("  SAMPLE_RATE  : "); Serial.println(SAMPLE_RATE);
    Serial.print  ("  N_FFT        : "); Serial.print(N_FFT);
    Serial.print  ("  ("); Serial.print(N_FFT * 1000 / SAMPLE_RATE); Serial.println(" ms window)");
    Serial.print  ("  HOP_LENGTH   : "); Serial.print(HOP_LENGTH);
    Serial.print  ("  ("); Serial.print(HOP_LENGTH * 1000 / SAMPLE_RATE); Serial.println(" ms hop)");
    Serial.print  ("  N_MELS       : "); Serial.println(N_MELS);
    Serial.print  ("  N_FRAMES     : "); Serial.println(N_FRAMES);
    Serial.print  ("  FEATURE DIM  : "); Serial.print(N_MELS); Serial.print("×");
                                          Serial.print(N_FRAMES); Serial.print("=");
                                          Serial.println(N_MELS * N_FRAMES);
    Serial.print  ("  Mel scale    : HTK  (2595·log10(1+hz/700)) ← matches notebook htk=True");
    Serial.println();
    Serial.print  ("  Filter norm  : None (plain triangular) ← matches notebook norm=None");
    Serial.println();

    // ── Load TFLite model ───────────────────────────────────────────────────
    g_tflite_model = tflite::GetModel(g_model_data);
    if (g_tflite_model == nullptr) {
        Serial.println("[ERROR] GetModel() returned null — check model_data.h");
        while (true) delay(1000);
    }

    // 2.4.0-ALPHA constructor: (model, resolver, arena, arena_size, error_reporter)
    static tflite::MicroInterpreter static_interp(
        g_tflite_model, g_resolver,
        g_arena, TFLITE_ARENA_SIZE,
        &g_error_reporter
    );
    g_interpreter = &static_interp;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERROR] AllocateTensors() failed — increase TFLITE_ARENA_SIZE");
        Serial.print  ("        Current size: "); Serial.println(TFLITE_ARENA_SIZE);
        Serial.println("        Increase in steps of 8192 and recompile.");
        while (true) delay(1000);
    }

    g_input  = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    // Cache quantisation parameters for fill_input_tensor()
    g_inp_scale = g_input->params.scale;
    g_inp_zp    = g_input->params.zero_point;

    Serial.println("[MODEL] Loaded successfully:");
    Serial.print  ("  Input shape  : ");
    for (int d = 0; d < g_input->dims->size; d++) {
        Serial.print(g_input->dims->data[d]);
        if (d < g_input->dims->size - 1) Serial.print("×");
    }
    Serial.println();
    Serial.print  ("  Input  quant : scale="); Serial.print(g_inp_scale, 6);
                                                Serial.print(" zp="); Serial.println(g_inp_zp);
    Serial.print  ("  Output quant : scale="); Serial.print(g_output->params.scale, 6);
                                                Serial.print(" zp="); Serial.println(g_output->params.zero_point);
    Serial.print  ("  Arena used   : "); Serial.print(g_interpreter->arena_used_bytes());
                                          Serial.println(" bytes");
    Serial.print  ("  Arena budget : "); Serial.print(TFLITE_ARENA_SIZE);
                                          Serial.println(" bytes");

    // ── Start PDM mic ───────────────────────────────────────────────────────
    PDM.onReceive(on_pdm_data);
    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("[ERROR] PDM.begin() failed");
        while (true) delay(1000);
    }

    Serial.println();
    Serial.println("[READY] Listening...");
    Serial.print  ("  Threshold : "); Serial.println(DECISION_THRESHOLD, 2);
    Serial.print  ("  Vote window: "); Serial.print(VOTE_WINDOW); Serial.println(" frames");
    Serial.print  ("  Stride    : "); Serial.print(STRIDE_SAMPLES);
                                       Serial.print(" samples (");
                                       Serial.print(STRIDE_SAMPLES * 1000 / SAMPLE_RATE);
                                       Serial.println(" ms)");
    Serial.println();
}

// =============================================================================
//  LOOP — sliding-window inference every STRIDE_SAMPLES samples
// =============================================================================

void loop() {
    // Wait until we have enough new audio
    if (g_new_samples < STRIDE_SAMPLES) return;

    noInterrupts();
    g_new_samples -= STRIDE_SAMPLES;
    interrupts();

    g_ring_read = (g_ring_read + STRIDE_SAMPLES) % RING_LEN;

    // Extract features (HTK mel + z-score)
    extract_features();

    // ── Mel stats diagnostic (runs every inference cycle) ───────────────────
    float mel_min = 1e10f, mel_max = -1e10f, mel_sum = 0.0f;
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < N_FRAMES; t++) {
            float v = g_mel[m][t];
            if (v < mel_min) mel_min = v;
            if (v > mel_max) mel_max = v;
            mel_sum += v;
        }
    float mel_mean = mel_sum / (float)(N_MELS * N_FRAMES);
    // After correct z-score: min ≈ -3, max ≈ 3, mean ≈ 0
    // If values are outside [-5, 5] the feature pipeline may have diverged.

    // ── Run inference ────────────────────────────────────────────────────────
    float p_agg = run_inference();
    bool  alert = update_vote(p_agg);
    digitalWrite(ALERT_PIN, alert ? HIGH : LOW);

    // ── Compute vote mean ────────────────────────────────────────────────────
    float vote_sum = 0.0f;
    for (int i = 0; i < VOTE_WINDOW; i++) vote_sum += g_vote_buf[i];
    float vote_mean = vote_sum / (float)VOTE_WINDOW;

    // ── Serial output (one line per inference cycle) ─────────────────────────
    Serial.print("mel=[");
    Serial.print(mel_min,  2); Serial.print(",");
    Serial.print(mel_max,  2); Serial.print(",μ=");
    Serial.print(mel_mean, 3); Serial.print("]  ");

    Serial.print("raw=[");
    Serial.print(g_output->data.int8[0]); Serial.print(",");
    Serial.print(g_output->data.int8[1]); Serial.print("]  ");

    Serial.print("p_agg=");
    Serial.print(p_agg, 3);
    Serial.print("  vote=");
    Serial.print(vote_mean, 3);
    Serial.print("  → ");
    Serial.println(alert ? "🔴 AGGRESSIVE" : "🟢 neutral");
}
