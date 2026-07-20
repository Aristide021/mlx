# MLX Codebase — Forensic Architecture Analysis

> **Purpose**: Comprehensive technical analysis of the MLX framework to inform design of a competing Apple Silicon GPU targeting framework.
> **Method**: Source code reading and tracing — not a code review but forensic architecture documentation.
> **Date**: April 2026
> **MLX Version**: Git commit `a8776b7bbd086bc37dbf9a5a6b72bec7c21a7b94`

---

## 1. MEMORY MODEL

### 1.1 Core Type Definitions

**File: `mlx/array.h`, lines 22-24**

```cpp
using ShapeElem = int32_t;
using Shape = SmallVector<ShapeElem>;
using Strides = SmallVector<int64_t>;
```

This is the **most critical architectural decision** in the entire codebase:

| Type | Storage | Max Value | Implication |
|------|---------|-----------|-------------|
| `ShapeElem` | `int32_t` | 2,147,483,647 | No single dimension can exceed ~2.1B elements |
| `Shape` | `SmallVector<int32_t>` | default capacity=10, dynamic if more dims | Stored on stack for ≤10 dimensions |
| `Strides` | `SmallVector<int64_t>` | ~9.2 quintillion (but limited by ShapeElem) | Can represent byte offsets beyond 2GB |
| `offset` (ArrayDesc) | `int64_t` (line 489) | ~9.2 quintillion | Sub-array offset into parent buffer |
| `size` (ArrayDesc) | `size_t` (line 471) | platform-dependent | Total element count returned by `a.size()` |
| `data_size` (ArrayDesc) | `size_t` (line 492) | platform-dependent | Actual buffer span used by array |

**File: `mlx/small_vector.h`, lines 92-93**

```cpp
template <typename T, size_t kSize = 10, typename Allocator = std::allocator<T>>
class SmallVector {
```

- Default inline capacity is 10 elements (set by `MAX_NDIM`)
- Uses inline storage (`char inline_storage_[sizeof(T) * kSize]`) then grows to heap via `grow()` method
- `grow()` doubles capacity, rounds up to power of 2 (lines 446-471)

### 1.2 Downstream Consumers of ShapeElem (int32_t)

**File: `mlx/ops.h`, line 56**
```cpp
MLX_API array as_strided(array a, Shape shape, Strides strides, size_t offset, StreamOrDevice s = {});
```

**File: `mlx/ops.h`, line 133**
```cpp
MLX_API array reshape(const array& a, Shape shape, StreamOrDevice s = {});
```

**File: `mlx/ops.h`, line 175-180**
```cpp
MLX_API array slice(const array& a, Shape start, Shape stop, Shape strides, StreamOrDevice s = {});
```

Every operation that accepts `Shape` implicitly accepts `int32_t` for dimension sizes.

### 1.3 Narrowing Casts (int64_t/size_t → int32_t)

**File: `mlx/array.h`, line 547**
```cpp
explicit array(std::initializer_list<T> data, Dtype dtype = TypeToDtype<T>())
    : array_desc_(std::make_shared<ArrayDesc>(
        Shape{static_cast<ShapeElem>(data.size())}, // <-- size_t → int32_t
        dtype)) {
```

**File: `mlx/backend/metal/matmul.cpp`, lines 183-185**
```cpp
int M,    // signed 32-bit
int N,    // signed 32-bit
int K,    // signed 32-bit
```
The entire matmul dispatch chain uses `int` for matrix dimensions, which overflow at 2GB elements.

**File: `mlx/backend/metal/matmul.cpp`, line 896**
```cpp
M *= batch_shape.back();  // Can overflow int if batch * M > INT_MAX
```

**File: `mlx/backend/metal/reduce.cpp`**
```cpp
bool large = in.size() > INT32_MAX;  // lines 197, 207, 220, 232, 242, 253, 275
```
MLX explicitly tracks `large` (exceeding INT32_MAX) for reduce operations and uses different kernel paths, but this is a workaround, not a fix.

### 1.4 Offset Arithmetic Sites

**File: `mlx/array.h`, lines 372-381**
```cpp
template <typename T>
T* data() {
  return reinterpret_cast<T*>(
      (static_cast<char*>(buffer().raw_ptr()) + array_desc_->offset));
}
```

Offset is `int64_t` — this is correct. However, the **byte computation** uses `array_desc_->offset` as byte offset only for CPU data. GPU kernels have their own offset logic.

### 1.5 Implications for Competing Framework

- **ShapeElem = int32_t is a deliberate ceiling** at ~2.1B elements per dimension. For models with sequence lengths or vocabularies exceeding this, MLX will silently misbehave or fail.
- **Strides are int64_t** but they describe steps in elements, not bytes. When combined with int32_t shapes, the total span is still limited.
- **A competing framework should use `int64_t` for all shape elements** and `size_t` for byte offsets. No narrowing casts exist for int64_t shape arithmetic in MLX by design, not by mistake.

---

## 2. METAL BACKEND AND COMMAND BUFFER SUBMISSION

### 2.1 Command Encoder Architecture

**File: `mlx/backend/metal/device.h`, lines 24-132**
```cpp
class CommandEncoder {
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  size_t buffer_sizes_{0};
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  ...
};
```

**File: `mlx/backend/metal/device.cpp`, lines 252-268**
```cpp
CommandEncoder::CommandEncoder(Device& d, int index, ResidencySet& residency_set)
    : device_(d) {
  queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
}
```

### 2.2 Submission Budget (The "needs_commit" Logic)

**File: `mlx/backend/metal/device.cpp`, lines 434-443**
```cpp
bool CommandEncoder::needs_commit() const {
  auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
  return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
}

void CommandEncoder::commit() {
  buffer_->commit();
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
  buffer_ops_ = 0;
  buffer_sizes_ = 0;
}
```

**File: `mlx/backend/metal/device.cpp`, lines 488-511**
```cpp
switch (arch) {
  case 'p': // phone
    max_ops_per_buffer_ = 20;
    max_mb_per_buffer_ = 40;
    break;
  case 'g': // base, pro
    max_ops_per_buffer_ = 40;
    max_mb_per_buffer_ = 40;
    break;
  case 's': // max
    max_ops_per_buffer_ = 50;
    max_mb_per_buffer_ = 50;
    break;
  case 'd': // ultra
    max_ops_per_buffer_ = 50;
    max_mb_per_buffer_ = 50;
    break;
  default:
    max_ops_per_buffer_ = 40;
    max_mb_per_buffer_ = 40;
}
max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
```

This is MLX's **command buffer splitting mechanism**. When `needs_commit()` returns true, the current command buffer is committed and a new one is created. This prevents GPU watchdog kills.

### 2.3 Evaluation Pipeline

**File: `mlx/backend/metal/eval.cpp`, lines 30-75**
```cpp
void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();
  auto outputs = arr.outputs();
  arr.primitive().eval_gpu(arr.inputs(), outputs);
  
  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(...);
    encoder.commit();
  }
}
```

**File: `mlx/backend/metal/device.cpp`, lines 345-358**
```cpp
void CommandEncoder::dispatch_threadgroups(...) {
  maybeInsertBarrier();
  buffer_ops_++;  // Incremented per dispatch
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
}
```

### 2.4 Key Findings

| Aspect | MLX Behavior | Implication |
|--------|-------------|-------------|
| Buffer splitting | Yes, via `needs_commit()` | Prevents watchdog for batch of ops |
| Timeout awareness | **NO** — `MTLDispatchDispatchTypeConcurrent` has timeout awareness, but `commitAndWaitUntilCompleted` blocks indefinitely | Long single operations CAN kill GPU |
| Per-operation splitting | **NO** — individual kernel dispatches are NOT split. A single `matmul(40000,40000,40000)` is ONE kernel launch with no internal splitting | A single large operation is one command buffer submission |
| `commandBufferWithUnretainedReferences` | Uses unretained refs to avoid retain/release overhead | Buffer pointers remain stable during execution |

### 2.5 Implications for Competing Framework

- MLX splits across operations but NOT within a single operation. A `gemm(M=100000, N=100000, K=100000)` launches as **one kernel** with no tiling or chunking. A competing framework must split large single operations.
- The `commandBufferWithUnretainedReferences` pattern is good for performance but requires careful address management.

---

## 3. LAZY EVALUATION MODEL

### 3.1 Graph Building vs Materialization

**File: `mlx/transforms.cpp`, lines 74-314**
```cpp
array eval_impl(std::vector<array> outputs, bool async) {
  std::deque<array> tape;
  
  // 1. DFS to record degrees of inputs
  std::stack<std::pair<std::reference_wrapper<array>, int>> dfs;
  dfs.emplace(synchronizer, 0);
  while (!dfs.empty()) {
    // Walk inputs, record cache[id] = degree
  }
  
  // 2. BFS to build tape with width limit
  int max_width = env::bfs_max_width();
  
  // 3. Process tape bottom-up
  while (!tape.empty()) {
    auto arr = std::move(tape.back());
    
    if (arr.primitive().device() == Device::gpu) {
      gpu::eval(arr);    // → device.cpp primitive dispatch
    } else {
      cpu::eval(arr);
    }
    
    // Backpressure
    if (scheduler::n_active_tasks() > MAX_ACTIVE_TASKS ||
        (get_active_memory() > get_memory_limit() && ...)) {
      scheduler::wait_for_one();
    }
  }
}
```

### 3.2 Key Data Points

| Constant | Value | File | Line |
|----------|-------|------|------|
| `MAX_ACTIVE_TASKS` | 10 | `transforms.cpp` | 25 |
| `bfs_max_width()` | env-configurable | `transforms.cpp` | 175 |
| Memory backpressure | `get_active_memory() > get_memory_limit()` | `transforms.cpp` | 265-266 |

### 3.3 How Lazy Evaluation Works

1. **`mx.add(a, b)`** creates an `array` with a `Primitive` but does NOT compute.
2. The `array` has `status = unscheduled` and stores the primitive pointer.
3. **`mx.eval(output)`** triggers `eval_impl()` which:
   - Does DFS from outputs to leaves, recording input degrees
   - Builds a topological tape (BFS with width limit)
   - Processes tape in reverse: calls `gpu::eval(arr)` for each node
   - `gpu::eval()` encodes Metal commands via `primitive().eval_gpu()`
4. **Materialization** happens when `eval()` is called or when `item()`, `data()` is accessed (which call `eval()` internally).

**File: `mlx/array.h`, lines 566-572**
```cpp
template <typename T>
T array::item() {
  if (size() != 1) { throw...; }
  eval();
  return *data<T>();
}
```

### 3.4 Graph-to-Command-Buffer Relationship

- Each `gpu::eval(arr)` call adds kernels to the **current** `CommandEncoder`
- Multiple `gpu::eval()` calls can share one `CommandBuffer` until `needs_commit()` triggers split
- The tape ordering ensures dependencies are encoded in correct order
- **Fences** handle inter-stream synchronization (lines 383-421 of `device.cpp`)

### 3.5 Checkpointing / Partial Evaluation

**There is NO mechanism for partial evaluation or checkpointing mid-graph.** The `eval_impl()` processes the entire tape in one synchronous pass. The only interruption is memory-based backpressure at the operation level.

### 3.6 Implications for Competing Framework

- MLX's lazy model is **define-by-run** (like PyTorch Dynamo), not static graph.
- A competing framework could adopt the same approach but with **explicit checkpoints** for memory-constrained devices by inserting `eval()` calls mid-graph.
- The tape-based approach is clean and efficient but assumes the graph fits in memory.

---

## 4. KERNEL LAYER

### 4.1 Directory Structure

```
mlx/backend/metal/kernels/
├── arange.h/.metal
├── binary.h/.metal
├── copy.h/.metal
├── conv.metal            # 2D convolution kernels
├── fp_quantized.h/.metal  # Quantized matmul (non-NAX)
├── fp_quantized_nax.h/.metal  # Quantized matmul (NAX)
├── gemv.metal
├── hadamard.h
├── layer_norm.metal
├── logsumexp.h/.metal
├── quantized.h/.metal
├── quantized_nax.h/.metal
├── random.metal
├── reduce.h/.metal
├── rms_norm.metal
├── rope.metal
├── scaled_dot_product_attention.metal
├── scan.h/.metal
├── sdpa_vector.h         # SDPA vector kernel (see below)
├── softmax.h/.metal
├── sort.h/.metal
├── ternary.h/.metal
├── unary.h/.metal
├── fft/
├── indexing/
├── reduction/
└── steel/                # High-performance GEMM/CONV library
    ├── attn/             # Steel attention (NAX variant)
    │   ├── kernels/steel_attention_nax.metal
    │   └── kernels/steel_attention.metal
    ├── conv/
    │   ├── kernels/steel_conv.metal
    │   ├── kernels/steel_conv_general.metal
    │   └── kernels/steel_conv_3d.metal
    └── gemm/
        ├── gemm.h                    # GEMM kernel template
        ├── gemm_nax.h               # NAX GEMM variant
        ├── kernels/steel_gemm_fused.metal
        ├── kernels/steel_gemm_fused_nax.metal
        ├── kernels/steel_gemm_splitk.metal
        ├── kernels/steel_gemm_splitk_nax.metal
        ├── kernels/steel_gemm_segmented.metal
        ├── kernels/steel_gemm_masked.metal
        └── kernels/steel_gemm_gather.metal
```

### 4.2 Steel GEMM Tile Sizes and Dtype Support

**File: `mlx/backend/metal/matmul.cpp`, lines 88-169 (GEMM_TPARAM_MACRO)**

| Device Type | Dtype | BM | BN | BK | WM | WN | Condition |
|-------------|-------|----|----|----|----|----|-----------|
| g/p (small) | complex64 | 64 | 32 | 8 | 4 | 1 | always |
| g/p (small) | half/bf16 | 64 | 64 | 16 | 1 | 2 | non-complex |
| g/p (small) | float32 | 64 | 32 | 32 | 2 | 2 | NT |
| d (ultra) | half/bf16 | 64 | 64 | 16 | 1 | 2 | large matmul |
| d (ultra) | float32 | 32 | 64 | 16 | 1 | 2 | large matmul |
| s/c/d | half/bf16 | 64 | 64 | 16 | 2 | 2 | small matmul |
| s/c/d | half/bf16 | 64 | 32 | 32 | 2 | 2 | small matmul, NT |

### 4.3 NAX (Neural Accelerator) Kernel Paths

**File: `mlx/backend/metal/device.cpp`, lines 817-835**
```cpp
bool is_nax_available() {
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
}
```

NAX requires:
- macOS/iOS 26.2+
- Gen ≥ 18 for phone (`p`), Gen ≥ 17 for desktop (`g`, `s`, `d`, `c`)

**NAX GEMM split-K dispatch:**
**File: `mlx/backend/metal/matmul.cpp`, lines 958-985**
```cpp
// Case 2: Large K with sufficient M, N
constexpr int min_mn_threshold = 2048 * 2048;  // M*N >= 4M
constexpr int min_k_threshold = 10240;          // K >= 10240
if (metal::is_nax_available() && 
    !issubdtype(a.dtype(), complexfloating) &&
    (env::enable_tf32() || a.dtype() != float32) &&
    int64_t(M) * N >= min_mn_threshold && K >= min_k_threshold &&
    K >= (3 * std::max(M, N))) {
  return steel_gemm_splitk_axpby_nax<...>(...);
}
```

**NAX does NOT support float32** — the condition `env::enable_tf32() || a.dtype() != float32` means:
- float32 matmuls use NAX ONLY when `env::enable_tf32()` is true (converting to TF32)
- Otherwise, float32 uses the non-NAX steel_gemm path

**NAX dtypes instantiated:**
- `bfloat16_t` → yes (via `steel_gemm_fused_nax_bfloat16_...`)
- `float16_t` → yes (via `steel_gemm_fused_nax_float16_...`)
- `float32` → TF32 only (via `enable_tf32()`)
- **float32 NAX gather_mm → DOES NOT EXIST** — `steel_gemm_gather_nax` only has half/bfloat variants

### 4.4 SDPA Kernel

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, lines 16-177**

```cpp
template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    ...
```

Instantiations (scaled_dot_product_attention.metal lines 31-43):
```cpp
#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
```

**Supported head dims**: 64, 96, 128, 256
**Supported dtypes**: float32, float16, bfloat16

**Submission structure**: SDPA is a **single kernel per query sequence**. There is NO splitting across command buffers for long sequences. The kernel loops `for (int i = simd_gid; i < N; i += BN)` — a single workgroup processes the entire KV sequence.

### 4.5 conv_general Writeback Path

**File: `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.h`, lines 179-224**

```cpp
// Store results to device memory
{
  int offset_m = c_row + mma_op.sm;    // int (32-bit)
  int offset_n = c_col + mma_op.sn;    // int (32-bit)
  C += offset_n;                        // pointer arithmetic with int
  
  if (offset_n >= gemm_params->N) return;
  
  for (int i = 0; i < mma_t::TM; i++) {
    int cm = offset_m + i * mma_t::TM_stride;
    int n = cm / jump_params->adj_out_hw;  // int division
    int hw = cm % jump_params->adj_out_hw;
    int oh = (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + base_oh;
    int ow = (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + base_ow;
    
    if (n < params->N && oh < params->oS[0] && ow < params->oS[1]) {
      int offset_cm = n * params->out_strides[0] + oh * params->out_strides[1] + ow * params->out_strides[2];
      C[offset + k] = Epilogue::apply(accum[k]);
    }
  }
}
```

All offset arithmetic uses **`int` (32-bit)** in Metal shaders. For output tensors with more than ~2B elements, `offset_cm` overflows silently.

### 4.6 Fused Kernel Implementations

**File: `mlx/backend/metal/matmul.cpp`**, line 220:
```cpp
kname << "steel_gemm_fused_nax_"
```
The "fused" kernel refers to a single kernel that handles A×B (+ C×alpha×beta variant). It does NOT fuse transformer layers.

**No fused transformer layer kernels exist.** Each operation (matmul, add, layer_norm, softmax) is a separate kernel.

---

## 5. SHAPE AND SIZE ARITHMETIC IN KERNELS

### 5.1 Metal Shader Integer Types

**File: `mlx/backend/metal/kernels/steel/gemm/gemm.h`, lines 165-172**
```cpp
const int c_row = tid_y * BM;
const int c_col = tid_x * BN;
const size_t c_row_long = size_t(c_row);  // Good: extends to size_t for pointer math
const size_t c_col_long = size_t(c_col);

A += transpose_a ? c_row_long : c_row_long * params->lda;
B += transpose_b ? c_col_long * params->ldb : c_col_long;
D += c_row_long * params->ldd + c_col_long;
```

The GEMM kernel **does** extend to `size_t` for pointer arithmetic in the output write, but **not** for the loop index `k`.

**File: `mlx/backend/metal/kernels/steel/gemm/params.h`**
```cpp
struct GEMMParams {
  const int M;           // 32-bit
  const int N;           // 32-bit
  const int K;           // 32-bit
  const int lda;         // 32-bit
  const int ldb;         // 32-bit
  const int ldd;         // 32-bit
  const int tiles_n;     // 32-bit
  const int tiles_m;     // 32-bit
  const int64_t batch_stride_a;  // 64-bit - correct
  const int64_t batch_stride_b;  // 64-bit - correct
  const int64_t batch_stride_d;  // 64-bit - correct
};
```

### 5.2 Convolution Parameters

**File: `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.h`**
- `implicit_gemm_conv_2d_general` uses **`int`** for all tile positioning
- `MLXConvParams<2>` uses int for output strides

### 5.3 SDPA Vector Parameters

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, lines 23-26**
```cpp
const constant size_t& k_head_stride [[buffer(6)]],    // 64-bit
const constant size_t& k_seq_stride [[buffer(7)]],     // 64-bit
const constant size_t& v_head_stride [[buffer(8)]],    // 64-bit
const constant size_t& v_seq_stride [[buffer(9)]],     // 64-bit
```

But `const constant int& N [[buffer(5)]]` — **N is int32_t**. This limits sequence length to 2^31 - 1.

### 5.4 Critical Overflow Locations

| Location | Type | Location | Risk |
|----------|------|----------|------|
| `GEMMParams::M/N/K` | `int` | params.h:12-14 | Matmuls > 2B in any dimension |
| `sdpa_vector: N` | `int` | sdpa_vector.h:22 | Sequence length > 2^31 - 1 |
| `implicit_gemm_conv_2d: C_per_group` | `int` | steel_conv.h:130 | Channels > 2B |
| `steel_conv_general: offset_cm` | `int` | steel_conv_general.h:203 | Output offset with large O×N |

---

## 6. PYTHON/C++ BOUNDARY

### 6.1 Binding Framework

**File: `python/src/array.cpp`** — uses **nanobind** (not pybind11).

```cpp
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
```

### 6.2 Array Construction from Python

**File: `python/src/array.cpp`, lines 295-303**
```cpp
.def(
    "__init__",
    [](mx::array* aptr, nb::object v, std::optional<mx::Dtype> t) {
      new (aptr) mx::array(create_array(v, t));
    },
    "val"_a, "dtype"_a = nb::none(),
    nb::sig("def __init__(self: array, val: Union[scalar, list, tuple, numpy.ndarray, array], dtype: Optional[Dtype] = None)"))
```

### 6.3 Buffer Protocol

**File: `python/src/array.cpp`, lines 284-287**
```cpp
PyType_Slot array_slots[] = {
    {Py_bf_getbuffer, (void*)getbuffer},
    {Py_bf_releasebuffer, (void*)releasebuffer},
    {0, nullptr}
};
```

The buffer protocol is implemented (see `python/src/buffer.h/cpp`) allowing direct pointer access to array data for numpy interop.

### 6.4 DLPack Implementation

**File: `python/src/array.cpp`, lines 499-513**
```cpp
.def("__dlpack__", [](const mx::array& a) { return mlx_to_dlpack(a); })
.def("__dlpack_device__", [](const mx::array& a) {
    if (mx::metal::is_available()) {
        return nb::make_tuple(8, 0);  // kDLMetal = 8
    } else if (mx::cu::is_available()) {
        return nb::make_tuple(13, 0);  // kDLCUDA = 13
    } else {
        return nb::make_tuple(1, 0);  // kDLCPU = 1
    }
})
```

### 6.5 GIL Handling

**File: `python/src/array.cpp`, lines 777-782**
```cpp
.def(
    "__repr__",
    [](mx::array& a) {
      nb::gil_scoped_release nogil;  // GIL released for repr
      std::ostringstream os;
      os << a;
      return os.str();
    })
```

Metal operations do NOT require GIL. The GIL is released before calling `eval()` and `item()`.

### 6.6 Memory Ownership

Arrays are passed by shared_ptr internally. The Python binding uses `nb::rv_policy::none` for in-place operations. The underlying `array::Data` struct manages lifetime.

### 6.7 What is NOT Exposed

- **No raw pointer access** from Python. The `data<T>()` method is not exposed.
- The `buffer()` method is internal C++ only.
- No way to get the `MTL::Buffer*` pointer from Python.
- No `__cuda_array_interface__` — only `__dlpack__`.

### 6.8 Implications for Competing Framework

- A competing framework should **expose `void* data()` via a Python capsule** for raw pointer access.
- MLX's DLPack implementation is standard and can be matched/cleaned-room implemented.

---

## 7. ALLOCATOR AND MEMORY MANAGEMENT

### 7.1 Allocator Interface

**File: `mlx/allocator.h`, lines 33-50**
```cpp
class Allocator {
  virtual Buffer malloc(size_t size) = 0;
  virtual void free(Buffer buffer) = 0;
  virtual size_t size(Buffer buffer) const = 0;
  virtual Buffer make_buffer(void* ptr, size_t size) { return Buffer{nullptr}; }
  virtual void release(Buffer buffer) {}
};
```

### 7.2 Allocation Model

**Dynamic allocation.** Memory is allocated at eval time, not at graph construction time.

**File: `mlx/transforms.cpp`, line 313**
```cpp
return synchronizer;  // The synchronizer is the final node
```

Allocations happen when `gpu::eval(arr)` is called, not when `mx.add(a,b)` is called. The `eval_gpu()` method for each primitive calls `allocator::malloc(out.nbytes())` to allocate the output buffer.

**File: `mlx/backend/metal/matmul.cpp`, line 571**
```cpp
C_split.set_data(allocator::malloc(C_split.nbytes()));
```

### 7.3 Buffer Caching

The allocator maintains a **free list / pool** of previously-allocated buffers to avoid repeated syscalls.

**File: `mlx/memory.h`, lines 29-34**
```cpp
MLX_API size_t get_cache_memory();
// The cache includes memory not currently used that has not been returned
// to the system allocator.
```

### 7.4 Memory Limits

**File: `mlx/memory.h`, lines 36-47**
```cpp
// The memory limit defaults to 1.5x the maximum recommended working set size
MLX_API size_t set_memory_limit(size_t limit);
MLX_API size_t size_t set_memory_limit(size_t limit);
```

### 7.5 Buffer Stability During Command Recording

**File: `mlx/backend/metal/device.cpp`, lines 278-301**
```cpp
void CommandEncoder::set_input_array(const array& a, int idx, int64_t offset) {
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  get_command_encoder()->setBuffer(a_buf, a.offset() + offset, idx);
}
```

**Addresses are NOT guaranteed stable** if the allocator reallocates between recording and execution. MLX handles this by allocating outputs before encoding commands in the same eval call.

**File: `mlx/backend/metal/eval.cpp`, line 31**
```cpp
auto pool = metal::new_scoped_memory_pool();
```
This creates an autorelease pool scoped to the eval call, ensuring buffers are not freed during command recording.

### 7.6 Implications for Competing Framework

- MLX's allocator is **not pre-allocated**. A competing framework with static pre-allocated pools would have lower allocation overhead.
- The pointer stability issue is real: if you need stable addresses, you must allocate **before** recording.

---

## 8. GQA / MULTI-HEAD ATTENTION ARCHITECTURE

### 8.1 SDPA Vector Implementation

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, lines 59-63**
```cpp
const int q_batch_head_idx = tid.x;
const int q_seq_idx = tid.y;
const int kv_head_idx = q_batch_head_idx / gqa_factor;  // GQA mapping
const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
```

### 8.2 GQA Factor

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, line 21**
```cpp
const constant int& gqa_factor [[buffer(4)]],
```

The `gqa_factor` is `num_q_heads / num_kv_heads`. The kernel maps query heads to kv heads by integer division:
```cpp
const int kv_head_idx = q_batch_head_idx / gqa_factor;
```

This is **correct but suboptimal**. The kernel does not handle the case where `gqa_factor` does not evenly divide `num_q_heads`. The threadgrid dimensions assume `tpg.y = num_q_heads * seq_len`.

### 8.3 Sequence Length Handling

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, line 47**
```cpp
int inner_k_stride = BN * int(k_seq_stride);
```

`k_seq_stride` is `size_t` but cast to `int`. For sequence lengths > 2GB, this wraps to negative.

**File: `mlx/backend/metal/kernels/sdpa_vector.h`, line 99**
```cpp
for (int i = simd_gid; i < N; i += BN) {
```

`int i` wraps at 2^31. For N > 2^31, the loop **never terminates** and accesses invalid memory.

### 8.4 Attention Submission Structure

Each SDPA call is **one kernel call** to `sdpa_vector`. There is no tiling across command buffers. The entire attention computation runs in a single threadgroup.

**File: `mlx/backend/metal/scaled_dot_product_attention.cpp`**
- Holds a reference to queries, keys, values through `copies`
- Calls `gpu::eval` once for the attention output

### 8.5 Implications for Competing Framework

- GQA mapping is correct for `q_heads = k * kv_heads` but **no validation** exists.
- The `int N` limit means attention cannot handle sequence lengths > 2^31.
- The kernel **does not split** long sequences — the entire sequence is processed in one pass.

---

## 9. AUTOGRAD

### 9.1 Implementation Type

**Define-by-run (tape-based).** MLX uses a **recording tape** of vector-Jacobian products computed during the backward pass.

**File: `mlx/transforms.cpp`, lines 347-521**
```cpp
std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotans,
    const std::vector<int>& argnums) {
    
  // 1. Create tracers from primals
  std::vector<array> primals_;
  for (auto& p : primals) primals_.push_back(make_tracer(p));
  
  // 2. Run function with tracers
  auto outputs = fun(primals_);
  
  // 3. Topologically sort graph, build tape
  std::vector<array> tape;
  std::function<void(array&)> recurse = [&](auto& a) {
    if (!calc_grad.count(input.id())) { continue; }
    tape.push_back(a);
  };
  
  // 4. Run tape backwards
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    auto& a = *it;
    auto vjps = a.primitive().vjp(a.inputs(), cotangents, argnums, outputs);
    // Accumulate VJPs
    cotan_map[in_id] = add(cotan_map[in_id], vjps[i], s);
  }
}
```

### 9.2 VJP Registration Pattern

Each primitive overrides:
```cpp
virtual std::vector<array> vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs);
```

The `DEFINE_GRADS()` macro (primitives.h lines 18-28) requires both `jvp` and `vjp`.

### 9.3 Gradient Registration

Gradients are registered per-primitive in `mlx/primitives.cpp`:
```cpp
std::vector<array> Matmul::vjp(...) {
  // Returns gradient w.r.t. each arg
  // vjp_a: dL/dA = cotan @ B^T
  // vjp_b: dL/dB = A^T @ cotan
}
```

### 9.4 Overhead Model

- Each primitive on the gradient tape has **one VJP kernel call** per argnum
- Accumulation is an additional `add` kernel call per input
- There is **no tape reuse** — each grad() call rebuilds the tape
- Tracer checking in `make_tracer` avoids unnecessary tracing for cached graphs

### 9.5 Implications for Competing Framework

- MLX's autograd is **cleanly separable** from the forward computation. A competing framework could reuse the forward kernels with a different autograd strategy.
- The "tape rebuilds every time" model is inefficient for training loops. A competing framework could implement **checkpointing** or **tape reuse** for repeated backward passes.

---

## 10. KNOWN BUGS AND STRUCTURAL WEAKNESSES

### 10.1 int32 Overflow Sites (arrays with more than 2^31 elements)

| File | Location | Type | Risk Level |
|------|----------|------|------------|
| `array.h:547` | `static_cast<ShapeElem>(data.size())` | narrowing | LOW (initializers rarely exceed 2B) |
| `steel/gemm/params.h:12-14` | `int M, N, K` | storage | **CRITICAL** — matmuls > 2B elements silently fail |
| `sdpa_vector.h:22` | `const constant int& N` | parameter | **CRITICAL** — attention > 2B seq len |
| `steel_conv_general.h:182-224` | `int offset_cm` in writeback | local | HIGH — large convolutions fail |
| `metal/matmul.cpp:896` | `M *= batch_shape.back()` | multiplication | HIGH — batch × M overflow |
| `metal/reduce.cpp` (multiple) | Checked with `> INT32_MAX` | — | **MITIGATED** — MLX handles this |
| `metal/matmul.cpp:550-551+` | `int _tm = (M + 32 - 1) / 32` all int division | local | MEDIUM |
| `metal/conv.cpp` | `int implicit_M = out.size() / conv_params.O` | narrowing | HIGH — implicit matmul overflow |

### 10.2 Single Large Operations Without Splitting

| Operation | Splitting | Evidence |
|-----------|-----------|----------|
| GEMM (regular) | **NO** — single kernel launch | `matmul.cpp:334 dispatch_threadgroups(grid_dims, group_dims)` |
| GEMM (split-K) | YES — across K dimension | Only triggers for M*N ≤ 2048 and K ≥ max(M,N) |
| GEMM (NAX split-K) | YES — across K dimension (3072 partition) | Only triggers for M*N ≥ 4M and K ≥ 10240 |
| SDPA (attention) | **NO** — single kernel per sequence | `sdpa_vector.h:99 for (int i = simd_gid; i < N; i += BN)` |
| Reduce | Partly — `strided_reduce_2pass` exists | But `bool large = in.size() > INT32_MAX` fallback exists |
| Conv (steel) | **NO** — single launch | `steel_conv_general.h:17-225` one kernel for entire implicit matmul |
| Element-wise | **NO** — one kernel per array | `binary.cpp`, `unary.cpp` — grid_size = array size |

### 10.3 Missing Kernel Instantiations

| Kernel | Dtype/Variant | Status |
|--------|--------------|--------|
| float32 NAX GEMM | float32 | **NOT AVAILABLE** — only TF32 via `enable_tf32()` |
| float32 NAX scatter/gather | float32 with gather | **NOT AVAILABLE** — `steel_gemm_gather_nax` only has half/bfloat |
| 3D Conv NAX | NAX | **NOT AVAILABLE** — `steel_conv_3d` has no NAX variant |
| Complex NAX | complex64 | **NOT AVAILABLE** — NAX disabled for complexfloating |
| Float64 GPU kernels | float64 | **NOT AVAILABLE** — no Metal kernel for float64 GEMM |
| Integer GEMM | int8, int32 | **NOT AVAILABLE** — steel_gemm only has float types |
| Fused transformer layers | N/A | **DOES NOT EXIST** — each operation is separate |
| GRUCell/LSTMCell | N/A | **NOT AVAILABLE** as primitives |

### 10.4 Memory Safety Issues

**Use-after-free risk:**
- **File: `device.cpp:398-405`** — `prev_ce_outputs_` cleanup relies on exact fence matching. If a fence is lost to an exception, outputs remain in the map indefinitely.

**Buffer stability:**
- The `array::Data` struct moves buffers between arrays (copy_shared_buffer). **If a kernel captures `MTL::Buffer*` and then the buffer is reused**, the kernel may write to the wrong memory. This is mitigated by the `add_temporary` system.

**Missing bounds checks:**
- **Steel GEMM** does NOT validate `M, N, K > 0`. If K=0, `gemm_k_iterations_aligned = 0` and the kernel writes uninitialized values.
- **SDPA vector** does not validate `N` or `gqa_factor` against thread grid dimensions.

### 10.5 Python-Only Logic

| Feature | Python Location | C++ Equivalent |
|---------|----------------|----------------|
| `mx.compile` caching | `python/src/transforms.cpp` | C++ `compile` exists but with different semantics |
| `metal_kernel` custom kernels | `mlx/backend/metal/custom_kernel.cpp` | Full C++ support |
| Type promotion rules | `python/src/utils.cpp` | C++ has different type promotion |
| Gradient tape caching | Python-only | No caching in C++ |
| `mx.vjp` with scalar return | `transforms.py` | C++ requires vector cotangents |

---

## ARCHITECTURAL DECISION SUMMARY AND RECOMMENDATIONS

### Decisions Worth Adopting

| Decision | Why | How |
|----------|-----|-----|
| Command buffer splitting (`needs_commit`) | Prevents GPU watchdog kills | Copy `device.cpp:434-443` pattern |
| Fence-based synchronization | Clean inter-stream sync | Copy `device.cpp:366-422` pattern |
| steel_gemm_splitk | Optimizes narrow-large matmul | Adapt `matmul.cpp:527-680` |
| NAX utilization for large matmuls | Hardware acceleration | Adapt `is_nax_available()` check |
| BFS tape building with width limit | Prevents OOM during graph traversal | Adapt `transforms.cpp:174-218` |
| Memory backpressure | `scheduler::wait_for_one()` when limit exceeded | Adapt `transforms.cpp:264-278` |

### Decisions to Avoid

| Decision | Why Avoid | Alternative |
|----------|-----------|------------|
| `ShapeElem = int32_t` | Ceiling at 2B per dimension | Use `int64_t` everywhere |
| Single kernel for large SDPA | GPU watchdog kills | Tile SDPA across blocks |
| Dynamic allocation | Overhead | Pre-allocate pool |
| No float32 NAX matmul | Misses 2x throughput on Apple Silicon NAX | Implement float32 NAX gather and scatter |
| No fused transformer kernels | More kernel launches | Fuse QKV projection + attention |
| `int M, N, K` in GEMMParams | Overflow at 2B | Use `int64_t` or validate |
| No raw pointer exposure | Blocks external kernel integration | Expose `void*` via Python capsule |

### Clean Room Implementation Opportunities

MLX is MIT licensed. The following can be adopted (with clean-room rewriting) without carrying forward problematic design decisions:

1. **steel_gemm** — the template instantiations, threading model, and swizzle logic can be reimplemented with `int64_t` shapes.
2. **scheduler.StreamThread** — the thread-per-stream model is elegant and standalone.
3. **fence-based synchronization** — the fence map pattern is generic and useful.
4. **reduction plan** — the `ContiguousReduce` / `RowReduceSimple` pattern is well-designed.
5. **memory allocator** — the pool allocator in `allocator.cpp` can be rewritten with pre-allocation.

---

## APPENDIX: FILES SUMMARY

| File | Purpose |
|------|---------|
| `mlx/array.h` | Core array type definition, ShapeElem=32_t, ArrayDesc |
| `mlx/small_vector.h` | Inline-then-heap storage for shapes/strides |
| `mlx/allocator.h` | Allocator base class, Buffer wrapper |
| `mlx/transforms.cpp` | eval_impl(), vjp, jvp, vmap |
| `mlx/scheduler.h` | StreamThread, Scheduler, MAX_ACTIVE_TASKS |
| `mlx/memory.h` | get_active_memory, set_memory_limit |
| `mlx/ops.h` | All public C++ API functions |
| `mlx/primitives.h` | Primitive base class, all primitive declarations |
| `mlx/backend/metal/eval.cpp` | GPU eval entry point, command encoder interaction |
| `mlx/backend/metal/device.h/cpp` | Device singleton, CommandEncoder, kernel cache |
| `mlx/backend/metal/matmul.cpp` | steel_matmul dispatch, NAX routing |
| `mlx/backend/metal/kernels/steel/gemm/gemm.h` | GEMM kernel template |
| `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h` | NAX GEMM kernel |
| `mlx/backend/metal/kernels/steel/gemm/params.h` | GEMMParams with int32 dimensions |
| `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h` | Convolution 2D kernel |
| `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.h` | Conv general kernel |
| `mlx/backend/metal/kernels/sdpa_vector.h` | SDPA vector kernel with GQA |
| `python/src/array.cpp` | nanobind Python bindings |

---

*End of forensic architecture analysis.*