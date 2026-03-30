from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizerFast,
)


# ---------------------------------------------------------------------------
# Generic sentence-level extraction (original cortexflow helper)
# ---------------------------------------------------------------------------

def extract_transformer_features(
    lines: List[str],
    model_name: str,
    layer: int,
    batch_size: int,
    max_length: int,
    pooling: str = "mean",
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    feats = []
    with torch.no_grad():
        for start in range(0, len(lines), batch_size):
            batch = lines[start : start + batch_size]
            inp = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states[layer]
            mask = inp["attention_mask"].unsqueeze(-1)

            if pooling == "last":
                last_idx = inp["attention_mask"].sum(dim=1) - 1
                pooled = hs[torch.arange(hs.shape[0]), last_idx]
            else:
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)

            feats.append(pooled.cpu().numpy())

    return np.vstack(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Word-level extraction – ported from extract_llm_activations.py
# (Bonnasse-Gahot & Pallier, NeurIPS 2024)
# ---------------------------------------------------------------------------

_PUNCT = ["-", "'", "\\", "\u2019", "\u201c", "\u00ab", "\u00bb", "\u2014"]
_CHINESE_PUNCT = [
    "\uff0c", "\u300a", "\u300b", "\u3002", "\uff1a",
    "\u201c", "\u201d", "\uff1b", "\uff1f", "\uff01",
    "\u2026", "\u3001",
]


def simplify_word(word: str) -> str:
    """Normalise a word for fuzzy token matching.

    Ported from extract_llm_activations.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).
    """
    word = word.lower().replace(" ", "")
    # Some annotation rows contain escape artifacts like "na\\ive".
    # Strip the slash so these rows can still align to tokenized text.
    word = word.replace("\\", "")
    for p in _PUNCT:
        word = word.replace(p, "")
    for p in _CHINESE_PUNCT:
        word = word.replace(p, "")
    return word


def do_word_match(word_in_list: str, word_in_text: str, lang: str = "en") -> bool:
    """Fuzzy word/token matching used during word-to-token alignment.

    Ported from extract_llm_activations.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).
    """
    word_in_text = simplify_word(word_in_text)
    if len(word_in_text) > 0 and word_in_list.startswith(word_in_text):
        return True
    if len(word_in_list) > (1 - (lang == "cn")) and word_in_list in word_in_text:
        return True
    if len(word_in_text) > (1 - (lang == "cn")) and word_in_text in word_in_list:
        return True
    # EN ad-hoc fixes
    if word_in_list == "one" and word_in_text == "1":
        return True
    if word_in_list == "did" and word_in_text == "didn":
        return True
    if word_in_list == "nt" and word_in_text == "t":
        return True
    if word_in_list == "does" and word_in_text == "doesn":
        return True
    if word_in_list == "do" and word_in_text == "don":
        return True
    if word_in_list == "is" and word_in_text == "isn":
        return True
    if word_in_list == "threetwofive" and word_in_text == "3":
        return True
    if word_in_list == "threetwosix" and word_in_text == "3":
        return True
    if word_in_list == "threetwoseven" and word_in_text == "3":
        return True
    if word_in_list == "threetwoeight" and word_in_text == "3":
        return True
    if word_in_list == "threetwonine" and word_in_text == "3":
        return True
    if word_in_list == "threethreezero" and word_in_text == "3":
        return True
    if word_in_list == "na\u00efve" and word_in_text == "naive":
        return True
    # FR ad-hoc fixes
    if word_in_list == "repondit" and word_in_text == "répondit":
        return True
    if word_in_list == "oeuvre" and word_in_text == "œuvre":
        return True
    if word_in_list == "oeil" and word_in_text == "œil":
        return True
    if word_in_list == "a" and word_in_text == "à":
        return True
    if word_in_list == "coeur" and word_in_text == "cœur":
        return True
    return False


def load_llm_model(
    model_name: str,
    access_token: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer, int, int, int]:
    """Load an LLM with hidden-state output enabled.

    Ported from extract_llm_activations.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    Returns
    -------
    model, tokenizer, n_layers, maxlen, stride
        n_layers includes the embedding layer (+1 over num_hidden_layers).
        stride = maxlen - 64 (sliding-window overlap).
    """
    if model_name.endswith("chinese"):
        full_name = f"ckiplab/{model_name}"
    elif model_name.endswith("french"):
        full_name = f"ClassCat/{model_name}"
    elif model_name.startswith("gpt2"):
        full_name = f"openai-community/{model_name}"
    elif model_name.startswith("opt"):
        full_name = f"facebook/{model_name}"
    elif model_name.startswith("Mistral"):
        full_name = f"mistralai/{model_name}"
    elif model_name.startswith("gemma"):
        full_name = f"google/{model_name}"
    elif model_name.startswith("mamba"):
        full_name = f"state-spaces/{model_name}"
    elif model_name.startswith("stablelm"):
        full_name = f"stabilityai/{model_name}"
    elif model_name.startswith("Llama"):
        full_name = f"meta-llama/{model_name}"
    elif model_name.startswith("Qwen"):
        full_name = f"Qwen/{model_name}"
    elif model_name.startswith("pythia"):
        full_name = f"EleutherAI/{model_name}"
    else:
        raise ValueError(f"Unrecognised model name: {model_name!r}")

    if model_name.startswith(("Meta", "gemma", "Mistral", "Llama")):
        assert access_token is not None, (
            "Provide access_token for gated HuggingFace models."
        )
        model = AutoModelForCausalLM.from_pretrained(
            full_name, output_hidden_states=True, token=access_token
        )
        tokenizer = AutoTokenizer.from_pretrained(full_name, token=access_token)
    elif model_name.startswith("pythia"):
        base, step = full_name.split("_step")
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=f"step{step}", output_hidden_states=True
        )
    elif model_name.endswith("chinese"):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        model = AutoModel.from_pretrained(full_name, output_hidden_states=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            full_name, output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(full_name)

    n_layers: int = model.config.num_hidden_layers + 1  # +1 for embedding layer
    try:
        maxlen: int = model.config.max_position_embeddings
    except AttributeError:
        maxlen = 32000  # SSMs / infinite context – treat as large

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stride = maxlen - 64

    return model, tokenizer, n_layers, maxlen, stride


def extract_word_activations_run(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    word_list: List[str],
    fulltext_run: str,
    n_layers: int,
    maxlen: int,
    stride: int,
    lang: str = "en",
) -> List[List[np.ndarray]]:
    """Extract word-level hidden-state activations for one run.

    Ported from extract_llm_activations.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    The full run text is tokenised with a sliding window (stride = maxlen - 64).
    Each word's activation at every layer is the mean of hidden states of the
    tokens belonging to that word.

    Parameters
    ----------
    model, tokenizer : from :func:`load_llm_model`
    word_list        : list of words in the run (from annotation CSV)
    fulltext_run     : full text of the run as a single string
    n_layers, maxlen, stride : from :func:`load_llm_model`
    lang             : "en", "fr", or "cn"

    Returns
    -------
    layers_words_activations : list of length n_layers, each element is a list
        of length n_words containing a 1-D numpy array of n_neurons values.
    """
    model.eval()
    inputs = tokenizer(
        fulltext_run,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=maxlen,
        return_overflowing_tokens=True,
        stride=stride,
    )

    # ---- map each word to its starting (batch, token) position ----
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token: List[Tuple[int, int]] = []

    for idx_word, word in enumerate(word_list):
        word_s = simplify_word(word)
        if idx_token == maxlen:
            idx_batch += 1
            idx_token = stride
        i_start, i_stop = inputs["offset_mapping"][idx_batch, idx_token].numpy()

        n = 0
        while n < 20 and not do_word_match(
            word_s, fulltext_run[i_start:i_stop], lang=lang
        ):
            idx_token += 1
            if idx_token == maxlen:
                idx_batch += 1
                idx_token = stride
            i_start, i_stop = inputs["offset_mapping"][idx_batch, idx_token].numpy()
            n += 1
        if n == 20:
            if lang == "cn":
                # Chinese annotation/token boundaries can occasionally diverge.
                # Keep progress with a best-effort alignment instead of failing
                # the whole run.
                idx_word_to_idx_token.append((idx_batch, idx_token))
                idx_token += 1
                continue
            raise RuntimeError(
                f"No matching token for word {idx_word} ({word!r})"
            )

        idx_word_to_idx_token.append((idx_batch, idx_token))

        if lang != "cn":
            idx_token += 1
        elif (
            idx_word < len(word_list) - 1
            and not do_word_match(
                word_list[idx_word + 1], fulltext_run[i_start:i_stop], lang=lang
            )
        ):
            idx_token += 1

    # ---- run model over each batch ----
    batch_size = inputs["input_ids"].shape[0]
    hidden_states: List[List[np.ndarray]] = []
    for k in range(batch_size):
        with torch.no_grad():
            outputs = model(
                inputs["input_ids"][k : k + 1],
                attention_mask=inputs["attention_mask"][k : k + 1],
            )
        hidden_states.append(
            [outputs["hidden_states"][lyr][0].numpy() for lyr in range(n_layers)]
        )

    # ---- pool hidden states per word ----
    layers_words_activations: List[List[np.ndarray]] = [[] for _ in range(n_layers)]

    for idx_word in range(len(word_list) - 1):
        idx_b, idx_t = idx_word_to_idx_token[idx_word]
        idx_b_next, idx_t_next = idx_word_to_idx_token[idx_word + 1]

        emb_layers: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
        if idx_b == idx_b_next:
            for i in range(idx_t, idx_t_next):
                for lyr in range(n_layers):
                    emb_layers[lyr].append(hidden_states[idx_b][lyr][i])
        else:
            # span a batch boundary: go to end of current batch …
            for i in range(idx_t, min(maxlen, inputs["input_ids"][idx_b].shape[0])):
                for lyr in range(n_layers):
                    emb_layers[lyr].append(hidden_states[idx_b][lyr][i])
            # … then continue from stride in the next batch
            for i in range(stride, idx_t_next):
                for lyr in range(n_layers):
                    emb_layers[lyr].append(hidden_states[idx_b_next][lyr][i])

        for lyr in range(n_layers):
            # Some CN words can collapse to an empty token span when two
            # consecutive words align to the same tokenizer position.
            # Fall back to the current token representation to keep a
            # consistent (n_neurons,) shape for every word.
            if len(emb_layers[lyr]) == 0:
                emb_layers[lyr].append(hidden_states[idx_b][lyr][idx_t])
            layers_words_activations[lyr].append(np.mean(emb_layers[lyr], axis=0))

    # last word
    idx_b, idx_t = idx_word_to_idx_token[-1]
    emb_layers = [[] for _ in range(n_layers)]
    for i in range(idx_t, min(maxlen, inputs["input_ids"][idx_b].shape[0])):
        token = inputs["input_ids"][idx_b, i]
        if token == tokenizer.eos_token_id:
            break
        for lyr in range(n_layers):
            emb_layers[lyr].append(hidden_states[idx_b][lyr][i])
    for lyr in range(n_layers):
        if len(emb_layers[lyr]) == 0:
            emb_layers[lyr].append(hidden_states[idx_b][lyr][idx_t])
        layers_words_activations[lyr].append(np.mean(emb_layers[lyr], axis=0))

    return layers_words_activations
