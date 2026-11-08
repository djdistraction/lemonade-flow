# 🌐 Lemonade-Flow
### High-Throughput, Asynchronous Heterogeneous Orchestration Framework for Cross-Modal Local AI

[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen%20AI-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://amd.com)
[![Lemonade Server](https://img.shields.io/badge/Lemonade-Server-FFD700?style=for-the-badge)](https://lemonade-server.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## ⚡ The Architectural Vision

**Lemonade-Flow** is an event-driven, non-blocking developer framework designed explicitly to orchestrate real-time, multi-modal pipelines on local consumer silicon.

While the unified local server ecosystem removes the friction of locally hosting text, speech, and image models via a single endpoint (`localhost:13305`), standard API client architectures handle network requests synchronously. In highly interactive applications—such as live voice dictation or real-time gaming loops—the user's spoken streaming rate easily outpaces local GPU rendering latency. Without structural safeguards, the processing pipeline bottlenecks, leading to **VRAM memory spikes, context window overflows, and severe system thread starvation.**

**Lemonade-Flow solves local pipeline degradation.** By introducing thread-isolated atomic slot buffers, LIFO preemption gates, and an adaptive sliding token-window memory system, it delivers true **heterogeneous load-balancing** across your local AMD chipsets without locking up system event loops.

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                       LEMONADE-FLOW                                       │
│                                                                                           │
│   ┌───────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────┐  │
│   │   Asynchronous Ingestion  │ │  Adaptive Context Core    │ │   LIFO Preemption Gate │  │
│   │  (Dropless CPU Audio STT) │ │  (NPU Token Guardrail)    │ │   (GPU VRAM Protection)│  │
│   └─────────────┬─────────────┘ └─────────────┬─────────────┘ └───────────┬────────────┘  │
└─────────────────┼─────────────────────────────┼───────────────────────────┼───────────────┘
                  ▼                             ▼                           ▼
        Lemonade whisper.cpp           Lemonade FastFlowLM           Lemonade SD.cpp
          [AMD Ryzen CPU]                [AMD XDNA NPU]               [AMD Radeon GPU]
```

---

## ✨ Core Engineering Innovations

### 1. Backpressure Guarding & LIFO Preemption Gate
When streaming real-time input, text models extract visual entities significantly faster than local graphics processors can render frames. Lemonade-Flow replaces linear task queues with an **atomic, single-slot LIFO preemption gate**. If a new visual frame command arrives while the GPU worker thread is processing a past matrix multiplication, the stale task is immediately dropped. This completely isolates VRAM heap spaces from expanding uncontrollably, preserving consistent millisecond latency targets.

### 2. Adaptive Sliding-Window Context Core
Long-running conversational sessions quickly bloat local context frames, degrading NPU token ingestion speeds. The `AdaptiveContext` module actively intercepts system arrays, running an efficient proxy token estimation workflow (`tiktoken`). If total memory payloads breach strict developer ceilings, the framework systematically crops out the oldest text slices while caching prompt directive anchors. This preserves microsecond response speeds indefinitely.

### 3. Automated Heterogeneous Hardware Prober
At startup, Lemonade-Flow queries active local model configuration endpoints. It performs custom parsing routines to dynamically identify and visually confirm the exact hardware execution blocks mapping across the system, assuring developers that text tasks target the **NPU**, voice models leverage **CPU thread arrays**, and image synthesis isolates onto the **GPU**.

---

## 🛠️ Quick Start & Setup

### Prerequisites
* **Lemonade Server** initialized and listening on `http://localhost:13305`.
* Python 3.10+ installed locally.

### Setup Instructions
```bash
git clone https://github.com/djdistraction/lemonade-flow.git
cd lemonade-flow
pip install -r requirements.txt
```

### Launching the Dashboard
```bash
streamlit run dashboard.py
```

---

## 📊 Performance Benchmark Objectives

* **Zero Event Loop Starvation:** Asynchronous audio ingestion lines maintain a smooth capture cadence without latency degradation during high-overhead visual matrix generation tasks on the GPU.
* **Bounded VRAM Isolation:** The task queue buffer size is strictly capped at `1`. Overwrites on the atomic rendering slot occur instantly, ensuring memory layouts remain entirely static during high-frequency speech input.
* **Development Efficiency Multiplication:** Replaces hundreds of lines of raw multi-threaded subprocess and locking logic with elegant, declarative pythonic parameters.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
