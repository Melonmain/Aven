#!/usr/bin/env python3
"""In-process RKLLM backend — runs a .rkllm model on the RK3588 NPU.

Used by llm_server when `llm.backend: rkllm` (fully offline, no network). The
model is loaded once into the llm_server process and generates tokens directly
through librkllmrt, so there is no rkllama HTTP hop.

Why not rkllama? Its ctypes bindings target RKLLM 1.2.x and are binary
incompatible with 1.3.0 (which we need for Gemma 4): RKLLMParam dropped
img_start/img_end/img_content and use_gpu, gained ignore_eos_token, top_k
became int32, and rkllm_init now takes an RKLLMCallback *struct* instead of a
bare function pointer. The structs below come from rkllm.h @ release-v1.3.0.

Chat templates are set explicitly via rkllm_set_chat_template: Gemma 4's
embedded template is not parseable by the runtime, and without this it emits a
single token and stops.
"""

import ctypes
import os
import queue
import threading
import time

# --- enums (rkllm.h) --------------------------------------------------------
RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING, RKLLM_RUN_FINISH, RKLLM_RUN_ERROR = 0, 1, 2, 3
RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [("base_domain_id", ctypes.c_int32),
                ("embed_flash", ctypes.c_int8),
                ("enabled_cpus_num", ctypes.c_int8),
                ("enabled_cpus_mask", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint8),
                ("use_cross_attn", ctypes.c_int8),
                ("reserved", ctypes.c_uint8 * 104)]


class RKLLMParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p),
                ("max_context_len", ctypes.c_int32),
                ("max_new_tokens", ctypes.c_int32),
                ("top_k", ctypes.c_int32),
                ("n_keep", ctypes.c_int32),
                ("top_p", ctypes.c_float),
                ("temperature", ctypes.c_float),
                ("repeat_penalty", ctypes.c_float),
                ("frequency_penalty", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int32),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("skip_special_token", ctypes.c_bool),
                ("ignore_eos_token", ctypes.c_bool),
                ("is_async", ctypes.c_bool),
                ("extend_param", RKLLMExtendParam)]


class _LastHidden(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class _Logits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float), ("prefill_tokens", ctypes.c_int),
                ("generate_time_ms", ctypes.c_float), ("generate_tokens", ctypes.c_int),
                ("memory_usage_mb", ctypes.c_float)]


class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int32),
                ("last_hidden_layer", _LastHidden), ("logits", _Logits),
                ("perf", RKLLMPerfStat)]


class _EmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]


class _TokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]


class _Image(ctypes.Structure):
    _fields_ = [("image_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_image_tokens", ctypes.c_size_t), ("n_image", ctypes.c_size_t),
                ("image_start", ctypes.c_char_p), ("image_end", ctypes.c_char_p),
                ("image_content", ctypes.c_char_p),
                ("image_width", ctypes.c_size_t), ("image_height", ctypes.c_size_t)]


class _Video(ctypes.Structure):
    _fields_ = [("video_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_frame_tokens", ctypes.c_size_t), ("n_frame_per_video", ctypes.c_size_t),
                ("n_video", ctypes.c_size_t), ("video_start", ctypes.c_char_p),
                ("video_end", ctypes.c_char_p), ("video_content", ctypes.c_char_p),
                ("frame_width", ctypes.c_size_t), ("frame_height", ctypes.c_size_t)]


class _MultiModal(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p), ("image", _Image), ("video", _Video)]


class _InputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p), ("embed_input", _EmbedInput),
                ("token_input", _TokenInput), ("multimodal_input", _MultiModal)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool),
                ("input_type", ctypes.c_int), ("input_data", _InputUnion)]


class _LoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class _PromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int), ("prompt_cache_path", ctypes.c_char_p)]


class _SamplingParam(ctypes.Structure):
    _fields_ = [("top_k", ctypes.c_int32), ("top_p", ctypes.c_float),
                ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float),
                ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int),
                ("lora_params", ctypes.POINTER(_LoraParam)),
                ("prompt_cache_params", ctypes.POINTER(_PromptCacheParam)),
                ("sampling_params", ctypes.POINTER(_SamplingParam)),
                ("keep_history", ctypes.c_int),
                ("max_new_tokens", ctypes.c_int32)]


LLMResultCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult),
                                     ctypes.c_void_p, ctypes.c_int)
_TokenizerCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
_EmbedCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [("result_callback", LLMResultCallback), ("result_userdata", ctypes.c_void_p),
                ("tokenizer_callback", _TokenizerCB), ("tokenizer_userdata", ctypes.c_void_p),
                ("embed_callback", _EmbedCB), ("embed_userdata", ctypes.c_void_p)]


# Chat templates (system, prefix, postfix) per model family.
CHAT_TEMPLATES = {
    "gemma": ("", "<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    "qwen": ("", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
    "llama": ("", "<|start_header_id|>user<|end_header_id|>\n",
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"),
}


def family_of(model_path):
    name = os.path.basename(model_path).lower()
    for fam in CHAT_TEMPLATES:
        if fam in name:
            return fam
    return "gemma"


class RKLLMModel:
    """One loaded .rkllm model. generate() streams tokens; calls are serialized."""

    def __init__(self, model_path, lib_path, max_context=4096, max_new_tokens=512,
                 temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1,
                 family=None, system_prompt=""):
        self.model_path = model_path
        self.family = family or family_of(model_path)
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self._q = None                       # active generation's token queue

        lib = ctypes.CDLL(lib_path)
        lib.rkllm_createDefaultParam.restype = RKLLMParam
        lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        lib.rkllm_init.restype = ctypes.c_int
        lib.rkllm_set_chat_template.argtypes = [ctypes.c_void_p] + [ctypes.c_char_p] * 3
        lib.rkllm_set_chat_template.restype = ctypes.c_int
        lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput),
                                  ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        lib.rkllm_run.restype = ctypes.c_int
        lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib = lib

        p = lib.rkllm_createDefaultParam()
        p.model_path = model_path.encode()
        p.max_context_len = int(max_context)
        p.max_new_tokens = int(max_new_tokens)
        p.top_k = int(top_k)
        p.top_p = float(top_p)
        p.temperature = float(temperature)
        p.repeat_penalty = float(repeat_penalty)
        p.skip_special_token = True
        p.is_async = False
        p.extend_param.base_domain_id = 1
        # Keep the (very large) token-embedding table on flash rather than RAM.
        # Loading it into RAM instead is measurably WORSE on this board, even
        # with 12 GB free — benchmarked on Gemma 4 E2B:
        #   embed_flash=1: 9.01 tok/s, 230 ms TTFT,  8.6 s load, 2.6 GB resident
        #   embed_flash=0: 6.55 tok/s, 305 ms TTFT, 87.9 s load, 12.2 GB resident
        # Generation is ~27% slower in RAM: the embedding table is read once per
        # token, but resident it evicts the page cache serving the weights that
        # every layer hits, and NPU inference here is memory-bandwidth bound.
        p.extend_param.embed_flash = 1
        p.extend_param.n_batch = 1
        p.extend_param.use_cross_attn = 0
        # RK3588 is big.LITTLE; pin inference to the four fast A76 cores.
        p.extend_param.enabled_cpus_num = 4
        p.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        self._cb = LLMResultCallback(self._on_token)   # keep a ref: C holds this pointer
        cbs = RKLLMCallback()
        ctypes.memset(ctypes.byref(cbs), 0, ctypes.sizeof(cbs))
        cbs.result_callback = self._cb
        self._cbs = cbs

        self.handle = ctypes.c_void_p()
        t0 = time.perf_counter()
        ret = lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(p), ctypes.byref(cbs))
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed ({ret}) for {model_path}")
        self.load_seconds = time.perf_counter() - t0

        sys_p, pre, post = CHAT_TEMPLATES[self.family]
        lib.rkllm_set_chat_template(self.handle, (system_prompt or sys_p).encode(),
                                    pre.encode(), post.encode())
        self.last_perf = None

    # -- generation ---------------------------------------------------------
    def _on_token(self, result_ptr, userdata, state):
        q = self._q
        if q is None:
            return 0
        if state == RKLLM_RUN_NORMAL:
            txt = result_ptr.contents.text
            if txt:
                q.put(("text", txt.decode("utf-8", "ignore")))
        elif state == RKLLM_RUN_FINISH:
            pf = result_ptr.contents.perf
            self.last_perf = {"prefill_tokens": pf.prefill_tokens,
                              "prefill_ms": pf.prefill_time_ms,
                              "generate_tokens": pf.generate_tokens,
                              "generate_ms": pf.generate_time_ms}
            q.put(("done", None))
        elif state == RKLLM_RUN_ERROR:
            q.put(("error", "RKLLM run error"))
        return 0

    def generate(self, prompt, max_new_tokens=None):
        """Yield generated text chunks for one prompt (serialized per model)."""
        with self._lock:
            q: queue.Queue = queue.Queue()
            self._q = q

            inp = RKLLMInput()
            ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(inp))
            inp.role = b"user"
            inp.enable_thinking = False
            inp.input_type = RKLLM_INPUT_PROMPT
            inp.input_data.prompt_input = prompt.encode()

            infer = RKLLMInferParam()
            ctypes.memset(ctypes.byref(infer), 0, ctypes.sizeof(infer))
            infer.mode = RKLLM_INFER_GENERATE
            infer.keep_history = 0
            infer.max_new_tokens = int(max_new_tokens or self.max_new_tokens)

            # rkllm_run blocks and calls back on this thread, so run it in a
            # worker and stream from the queue as tokens arrive.
            err = {}

            def run():
                try:
                    rc = self.lib.rkllm_run(self.handle, ctypes.byref(inp),
                                            ctypes.byref(infer), None)
                    if rc != 0:
                        err["rc"] = rc
                        q.put(("error", f"rkllm_run rc={rc}"))
                except Exception as exc:  # noqa: BLE001
                    q.put(("error", str(exc)))

            th = threading.Thread(target=run, daemon=True)
            th.start()
            try:
                while True:
                    kind, val = q.get()
                    if kind == "text":
                        yield val
                    elif kind == "error":
                        print(f"[rkllm] {val}", flush=True)
                        break
                    else:
                        break
            finally:
                th.join(timeout=30)
                self._q = None

    def close(self):
        if getattr(self, "handle", None):
            try:
                self.lib.rkllm_destroy(self.handle)
            except Exception:  # noqa: BLE001
                pass
            self.handle = None
