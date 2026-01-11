# AI Detection API Technical Reference

This document provides a comprehensive breakdown of the `/detect` API response. Every field is designed to help the mobile app make informed decisions about content authenticity.

---

## 1. Top-Level Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| **`summary`** | `String` | The system's plain-English conclusion (e.g., `"Likely AI"`, `"Verified Human"`, or `"Suspicious (Uncertain)"`). |
| **`confidence_score`** | `Float` | The final aggregated probability [0.0 - 1.0] that the content is AI-generated. |
| **`suspicious`** | `Boolean` | **CRITICAL**: Set to `true` if the signal is in the "gray area" (conflicting metadata vs. pixel artifacts). Treat this as a trigger for a "Warning" UI. |
| **`gpu_bypassed`** | `Boolean` | `true` if a verdict was reached instantly via metadata signatures (e.g., C2PA) without running heavy GPU neural networks. |
| **`gpu_time_ms`** | `Float` | The actual time in milliseconds the request spent on the GPU. (Use this for performance monitoring). |

---

## 2. Detection Layers (`layers`)

The engine analyzes files in two stages.

### A. Layer 1: Metadata (`layer1_metadata`)
Focuses on cryptographic signatures, system provenance (EXIF), and C2PA.

*   **`status`** (`String`): `"verified_ai"`, `"verified_human"`, or `"not_found"`.
*   **`provider`** (`String`): The entity that signed the file (e.g., `"Google C2PA SDK"`, `"OpenAI"`).
*   **`description`** (`String`): Details on why this status was chosen.
*   **`human_score`** (`Float`): Confidence that headers indicate a real device.
*   **`ai_score`** (`Float`): Confidence that headers indicate AI software.
*   **`signals`** (`Array`): Specific tags found, such as `"Trusted device manufacturer"` or `"AI-typical dimensions"`.

### B. Layer 2: Forensics (`layer2_forensics`)
Deep visual analysis of pixel-level artifacts and frequency distributions.

*   **`status`** (`String`): `"detected"`, `"not_detected"`, or **`"skipped"`** (if `gpu_bypassed` is true).
*   **`probability`** (`Float`): Raw neural network probability of AI manipulation.
*   **`signals`** (`Array`): Types of visual evidence found (e.g., `"FFT artifacts"`, `"High frequency noise"`).

---

## 3. Metadata Summary Detail (`metadata`)

A detailed audit for debugging or advanced transparency features.

*   **`extracted`** (`Dict`): Raw key-value pairs of EXIF and header data.
*   **`bypass_reason`** (`String`): Narrative explanation for why GPU inference was skipped.
*   **`signals`** (`Array`): Combined list of all metadata-level markers found.

---

## 💡 Best Practices for App Developers
1.  **Trust `suspicious`**: Even if `confidence_score` is only `0.5`, if `suspicious` is `true`, show a warning. This flag is the result of a specialized "Suspicion Window" logic in the server.
2.  **Verify `gpu_bypassed`**: High-trust applications should prioritize results where `gpu_bypassed` is `true`, as these are mathematically verified through signatures rather than just heuristic estimation.
3.  **Local Metadata**: Ensure the app sends `captured_in_app: true` to maximize the chance of instant results.
