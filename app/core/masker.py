import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import tiktoken
from interegular.fsm import FSM

NEG_INF = float("-inf")

_FSM_STATE_TRANSITION_TABLE: Dict[int, Dict[int, Dict[int, int]]] = {}


class VocabularyMapper:
    def __init__(self, tokenizer_name: str = "gpt-4o"):
        self.tokenizer_name = tokenizer_name
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")

        self.vocab_size = self.encoding.n_vocab
        self.eos_token_id = self.encoding.eot_token

        self.token_id_to_str: Dict[int, str] = {}
        for token_id in range(min(self.vocab_size, 100256)):
            try:
                decoded_bytes = self.encoding.decode_single_token_bytes(token_id)
                self.token_id_to_str[token_id] = decoded_bytes.decode("utf-8", errors="replace")
            except Exception:
                continue

        self.single_char_tokens: Dict[str, int] = {}
        for tid, tstr in self.token_id_to_str.items():
            if len(tstr) == 1:
                self.single_char_tokens[tstr] = tid

        self.multi_char_tokens: Dict[str, List[Tuple[int, str]]] = {}
        for tid, tstr in self.token_id_to_str.items():
            if 1 < len(tstr) <= 6 and tid < 20000:
                c0 = tstr[0]
                self.multi_char_tokens.setdefault(c0, []).append((tid, tstr))

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.encoding.decode(token_ids)


default_vocab = VocabularyMapper()


class TokenLogitMasker:
    def __init__(
        self,
        fsm: FSM,
        vocab: Optional[VocabularyMapper] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.fsm = fsm
        self.vocab = vocab or default_vocab
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.vocab.eos_token_id
        self.current_state = fsm.initial

        fsm_id = id(self.fsm)
        if fsm_id not in _FSM_STATE_TRANSITION_TABLE:
            _FSM_STATE_TRANSITION_TABLE[fsm_id] = {}

        self._state_transition_cache = _FSM_STATE_TRANSITION_TABLE[fsm_id]

    def is_finished(self) -> bool:
        return self.current_state in self.fsm.finals

    def _compute_valid_transitions_for_state(self, state: int) -> Dict[int, int]:
        valid_transitions: Dict[int, int] = {}
        st_map = self.fsm.map.get(state, {})
        if not st_map:
            return valid_transitions

        for char, tid in self.vocab.single_char_tokens.items():
            sym = self.fsm.alphabet.get(char)
            if sym is not None and sym in st_map:
                valid_transitions[tid] = st_map[sym]

        for c0, candidates in self.vocab.multi_char_tokens.items():
            sym0 = self.fsm.alphabet.get(c0)
            if sym0 is not None and sym0 in st_map:
                nxt = st_map[sym0]
                for tid, tstr in candidates:
                    curr = nxt
                    valid = True
                    for c in tstr[1:]:
                        s = self.fsm.alphabet.get(c)
                        if s is None or curr not in self.fsm.map or s not in self.fsm.map[curr]:
                            valid = False
                            break
                        curr = self.fsm.map[curr][s]
                    if valid:
                        valid_transitions[tid] = curr

        return valid_transitions

    def get_valid_tokens_for_state(self, state: int) -> Dict[int, int]:
        if state not in self._state_transition_cache:
            self._state_transition_cache[state] = self._compute_valid_transitions_for_state(state)
        return self._state_transition_cache[state]

    def get_valid_token_ids(self) -> Set[int]:
        valid_map = self.get_valid_tokens_for_state(self.current_state)
        valid_tokens = set(valid_map.keys())

        if self.is_finished():
            valid_tokens.add(self.eos_token_id)

        return valid_tokens

    def apply_mask_to_logits(self, logits: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        valid_tokens = self.get_valid_token_ids()

        masked_logits = np.full_like(logits, fill_value=NEG_INF)
        for token_id in valid_tokens:
            if token_id < len(logits):
                masked_logits[token_id] = logits[token_id]

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return masked_logits, latency_ms

    def sample_valid_token(
        self,
        raw_logits: np.ndarray,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> Tuple[int, float]:
        t0 = time.perf_counter()
        valid_tokens = list(self.get_valid_token_ids())
        if not valid_tokens:
            raise RuntimeError("FSM deadlock: no valid transitions from current state")

        valid_tokens_arr = np.array([tid for tid in valid_tokens if tid < len(raw_logits)])
        if len(valid_tokens_arr) == 0:
            raise RuntimeError("Valid token indices out of bounds")

        sub_logits = raw_logits[valid_tokens_arr]

        if temperature <= 1e-4:
            chosen_token = int(valid_tokens_arr[np.argmax(sub_logits)])
        else:
            scaled_logits = sub_logits / max(temperature, 1e-4)
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)

            if top_p < 1.0:
                sorted_idx = np.argsort(probs)[::-1]
                cum_probs = np.cumsum(probs[sorted_idx])
                cutoff = cum_probs <= top_p
                if not np.any(cutoff):
                    cutoff[0] = True
                valid_tokens_arr = valid_tokens_arr[sorted_idx[cutoff]]
                probs = probs[sorted_idx[cutoff]]
                probs = probs / np.sum(probs)

            choice = np.random.choice(len(valid_tokens_arr), p=probs)
            chosen_token = int(valid_tokens_arr[choice])

        t1 = time.perf_counter()
        return chosen_token, (t1 - t0) * 1000.0

    def step(self, token_id: int) -> bool:
        if token_id == self.eos_token_id:
            return self.is_finished()

        valid_map = self.get_valid_tokens_for_state(self.current_state)
        if token_id in valid_map:
            self.current_state = valid_map[token_id]
            return True
        return False

    def reset(self):
        self.current_state = self.fsm.initial


try:
    import torch
    from transformers import LogitsProcessor

    class FSMLogitsProcessor(LogitsProcessor):
        def __init__(self, masker: TokenLogitMasker):
            self.masker = masker

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            for batch_idx in range(scores.shape[0]):
                valid_token_ids = self.masker.get_valid_token_ids()
                mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
                for tid in valid_token_ids:
                    if tid < len(mask):
                        mask[tid] = False
                scores[batch_idx, mask] = NEG_INF

            if scores.shape[0] == 1 and input_ids.shape[1] > 0:
                last_token = input_ids[0, -1].item()
                self.masker.step(last_token)

            return scores

except ImportError:
    FSMLogitsProcessor = None  # type: ignore
