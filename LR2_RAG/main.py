
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # reduce noisy TF/CUDA logs from deps

import re
import argparse
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from llama_cpp import Llama

from sentence_transformers import SentenceTransformer


# -----------------------------
# Utils
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

def normalize_ru(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'«».,:;!?()\[\]{}]", "", s)
    return s

def tokenize(text: str) -> List[str]:
    text = (text or "").lower().replace("ё", "е")
    return _WORD_RE.findall(text)

def safe_read_text(path: str) -> str:
    for enc in ["utf-8-sig", "utf-8", "cp1251"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def chunk_text(text: str, chunk_chars: int = 900, overlap_chars: int = 180) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_chars)
        cut = end
        for sep in ["\n\n", "\n", ". ", "! ", "? "]:
            j = text.rfind(sep, i + int(chunk_chars * 0.6), end)
            if j != -1:
                cut = j + len(sep)
                break
        chunk = text[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        nxt = max(i + chunk_chars - overlap_chars, cut)
        if nxt <= i:
            nxt = cut
        i = nxt
    return chunks

def trim_contexts_to_budget(contexts: List[str], max_chars: int = 12000) -> List[str]:
    out = []
    s = 0
    for c in contexts:
        if s + len(c) > max_chars:
            break
        out.append(c)
        s += len(c)
    return out


# -----------------------------
# Book mapping
# -----------------------------
BOOK_TO_FILES = {
    "ася": ["dataset/turgenev/Turgenev_Asya.txt"],
    "муму": ["dataset/turgenev/Turgenev_Mumu.txt"],
    "записки охотника": ["dataset/turgenev/Turgenev_ZapiskiOhotnika.txt"],

    "война и мир": ["dataset/tolstoy/Tolstoy_VoynaIMir1.txt", "dataset/tolstoy/Tolstoy_VoynaIMir2.txt"],

    "город без памяти": ["dataset/bulichev/Bulichev_Gorod_bez_pamyati.txt"],
    "лиловый шар": ["dataset/bulichev/Bulichev_Lilovii_shar.txt"],
    "сто лет тому вперед": ["dataset/bulichev/Bulichev_Sto_let_tomu_vpered.txt"],

    "как поссорился иван иванович с иваном никифоровичем": ["dataset/gogol/Gogol_KakPossorilsyaIvanIvanovichSIvanomNikiforovichem.txt"],
    "мертвые души": ["dataset/gogol/Gogol_MertvieDushi.txt"],
    "тарас бульба": ["dataset/gogol/Gogol_TarasBulba.txt"],
    "вий": ["dataset/gogol/Gogol_Vii.txt"],
    "старосветских помещиках": ["dataset/gogol/Gogol_StarosvetskiyePomeshchiki.txt"],

    "роковые яйца": ["dataset/bulgakov/Bulgakov_RokovyeYayca.txt"],
    "мастер и маргарита": ["dataset/bulgakov/Bulgakov_MasterIMargarita.txt.txt"],
    "морфий": ["dataset/bulgakov/Bulgakov_Morfiyi.txt"],
    "полотенце с петухом": ["dataset/bulgakov/Bulgakov_PolotenceSPetuhom.txt"],
    "стальное горло": ["dataset/bulgakov/Bulgakov_StalnoeGorlo.txt"],
    "звездная сыпь": ["dataset/bulgakov/Bulgakov_ZvezdnayaSyp.txt"],
    "тьма египетская": ["dataset/bulgakov/Bulgakov_TmaEgipetskaya.txt"],
    "приключения покойника": ["dataset/bulgakov/Bulgakov_PriklyucheniyaPokoynika.txt"],
    "крещение поворотом": ["dataset/bulgakov/Bulgakov_KrescheniePovorotom.txt"],

    "княгиня лиговская": ["dataset/lermontov/Lermontov_KnyaginyaLigovskaya.txt"],
    "дело артамоновых": ["dataset/gorkiyi/Gorkiyi_DeloArtamonovyih.txt"],
}

BOOK_ALIASES = {
    "сто лет тому вперёд": "сто лет тому вперед",
    "крещении поворотом": "крещение поворотом",
    "тарaс бульба": "тарас бульба",
}


# -----------------------------
# Robust path resolving
# -----------------------------
def build_basename_index(book_texts: Dict[str, str]) -> Dict[str, str]:
    m = {}
    for rel in book_texts.keys():
        base = os.path.basename(rel)
        if base not in m:
            m[base] = rel
    return m

def resolve_path(rel_path: str, book_texts: Dict[str, str], basename_map: Dict[str, str]) -> Optional[str]:
    if rel_path in book_texts:
        return rel_path

    if rel_path.startswith("dataset/"):
        p2 = rel_path[len("dataset/"):]
        if p2 in book_texts:
            return p2

    p3 = "dataset/" + rel_path
    if p3 in book_texts:
        return p3

    base = os.path.basename(rel_path)
    if base in basename_map:
        return basename_map[base]

    return None


# -----------------------------
# Retrieval structures
# -----------------------------
@dataclass
class BookIndex:
    name_norm: str
    chunks: List[str]
    tokenized_chunks: List[List[str]]
    bm25: BM25Okapi

def build_bm25_index(chunks: List[str]) -> BookIndex:
    tok = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tok)
    return BookIndex(name_norm="", chunks=chunks, tokenized_chunks=tok, bm25=bm25)


# -----------------------------
# Embeddings helpers
# -----------------------------
def is_e5_model(name: str) -> bool:
    return "e5" in (name or "").lower()

def encode_query(embedder: SentenceTransformer, text: str, e5: bool) -> np.ndarray:
    t = f"query: {text}" if e5 else text
    return embedder.encode([t], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

def encode_passages(embedder: SentenceTransformer, texts: List[str], e5: bool) -> np.ndarray:
    tt = [f"passage: {x}" if e5 else x for x in texts]
    return embedder.encode(
        tt,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)


# -----------------------------
# Better retrieval: option-aware + per-option evidence + safe semantic rerank
# -----------------------------
def make_retrieval_queries(question: str, options: List[str]) -> List[str]:
    q = (question or "").strip()
    opts = [str(o).strip() for o in options if o is not None and str(o).strip()]
    queries = [q]
    queries += [f"{q} {o}" for o in opts]
    if opts:
        queries.append(f"{q} " + " ".join(opts))
    return dedup_keep_order([x for x in queries if x.strip()])

def bm25_top_indices(book_index: BookIndex, query: str, top_k: int) -> List[int]:
    q_tok = tokenize(query)
    scores = book_index.bm25.get_scores(q_tok)
    idxs = np.argsort(-scores)[:top_k]
    return [int(i) for i in idxs]

def overlap_score(chunk_tokens: List[str], option_text: str) -> int:
    opt = set(tokenize(option_text))
    if not opt:
        return 0
    ct = set(chunk_tokens)
    return sum(1 for t in opt if t in ct)

def retrieve_strong(
    book_index: BookIndex,
    question: str,
    options: List[str],
    top_k: int = 8,
    cand_k: int = 120,
    per_option_k: int = 2,
    # embeddings
    embedder: Optional[SentenceTransformer] = None,
    e5: bool = False,
    emb_weight: float = 1.4,
) -> List[str]:
    """
    1) BM25 candidates from expanded queries
    2) Force per-option evidence chunks (BM25(question+option))
    3) Rerank candidates by:
       - BM25 RRF-like rank signal over multiple queries
       - option overlap signal
       - optional semantic similarity (rerank only within BM25 candidates)
    """
    queries = make_retrieval_queries(question, options)

    # Collect BM25 top lists for each query
    top_lists = []
    for q in queries:
        top_lists.append(bm25_top_indices(book_index, q, top_k=min(cand_k, len(book_index.chunks))))

    # Candidate pool
    cand_idxs = dedup_keep_order([i for lst in top_lists for i in lst])[:cand_k]

    # Must-have per-option evidence
    must = []
    for opt in options:
        qopt = f"{question} {opt}"
        must.extend(bm25_top_indices(book_index, qopt, top_k=per_option_k))
    must = dedup_keep_order(must)

    # Score candidates with BM25 rank fusion (RRF-like) + overlap + optional semantic
    rrf_k = 60.0
    rrf_score = {i: 0.0 for i in cand_idxs}
    for lst in top_lists:
        for r, idx in enumerate(lst, start=1):
            if idx in rrf_score:
                rrf_score[idx] += 1.0 / (rrf_k + r)

    # overlap signal
    ov_score = {}
    for i in cand_idxs:
        ov_score[i] = max(overlap_score(book_index.tokenized_chunks[i], opt) for opt in options)

    # semantic signal (only within candidates)
    sem_score = {i: 0.0 for i in cand_idxs}
    if embedder is not None and cand_idxs:
        # Use max similarity over (question) and (question+option) queries
        q_embs = [encode_query(embedder, question, e5=e5)]
        for opt in options:
            q_embs.append(encode_query(embedder, f"{question} {opt}", e5=e5))
        q_mat = np.stack(q_embs, axis=0)  # (Q, D)

        passages = [book_index.chunks[i] for i in cand_idxs]
        p_emb = encode_passages(embedder, passages, e5=e5)  # (C, D)
        sims = p_emb @ q_mat.T  # (C, Q)
        best = np.max(sims, axis=1)  # (C,)
        for j, i in enumerate(cand_idxs):
            sem_score[i] = float(best[j])

    # final score
    scored = []
    for i in cand_idxs:
        # overlap is discrete; scale softly
        s = 10.0 * rrf_score[i] + 0.25 * ov_score[i] + emb_weight * sem_score[i]
        scored.append((s, i))
    scored.sort(reverse=True, key=lambda x: x[0])

    # Build final contexts: must + best scored
    picked = dedup_keep_order(must + [i for _, i in scored])[:top_k]
    return [book_index.chunks[i] for i in picked]


# -----------------------------
# Prompting & Answering
# -----------------------------
SYSTEM_RU = (
    "Ты — аккуратный помощник по литературе. "
    "Тебе дан вопрос и 4 варианта ответа. "
    "Также дан набор фрагментов текста произведения (контекст). "
    "Выбери правильный вариант на основе контекста. "
    "Если в контексте ответа нет напрямую, выбери наиболее вероятный вариант по смыслу произведения. "
    "Отвечай СТРОГО одной цифрой: 1, 2, 3 или 4. Без пояснений."
)

def make_prompt(question: str, options: List[str], contexts: List[str]) -> str:
    a, b, c, d = options
    ctx_block = "\n\n".join([f"[Фрагмент {i+1}]\n{t}" for i, t in enumerate(contexts)])
    return (
        f"{SYSTEM_RU}\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Варианты:\n"
        f"1) {a}\n"
        f"2) {b}\n"
        f"3) {c}\n"
        f"4) {d}\n\n"
        f"Контекст:\n{ctx_block}\n\n"
        f"Ответ (только 1-4):"
    )

def make_notes_prompt(question: str, options: List[str], contexts: List[str]) -> str:
    a, b, c, d = options
    ctx_block = "\n\n".join([f"[Фрагмент {i+1}]\n{t}" for i, t in enumerate(contexts)])
    return (
        "Ты помогаешь выбрать правильный вариант ответа по контексту произведения.\n"
        "Сделай короткие заметки (5-8 строк):\n"
        "- какие факты из контекста важны\n"
        "- какой вариант лучше всего подходит и почему\n"
        "НЕ ПИШИ цифры 1-4 отдельно в ответе, не делай нумерацию.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Варианты:\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n\n"
        f"Контекст:\n{ctx_block}\n\n"
        "Заметки:"
    )

def parse_answer(text: str) -> Optional[int]:
    t = (text or "").strip()
    m = re.search(r"\b([1-4])\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"([1-4])", t)
    if m:
        return int(m.group(1))
    return None

def vote_answers(ans_list: List[int]) -> int:
    c = Counter(ans_list)
    best, best_cnt = c.most_common(1)[0]
    ties = [k for k, v in c.items() if v == best_cnt]
    return min(ties)

def try_build_digit_grammar():
    """
    Prefer strict grammar if llama_cpp provides it.
    """
    try:
        from llama_cpp import LlamaGrammar  # type: ignore
        # Minimal grammar allowing only a single digit
        g = LlamaGrammar.from_string(
            r"""
            root ::= ("1" | "2" | "3" | "4")
            """,
            verbose=False
        )
        return g
    except Exception:
        return None

_DIGIT_GRAMMAR = try_build_digit_grammar()

def llm_choose_digit(llm: Llama, prompt: str, seed: int = 12345) -> Optional[int]:
    """
    Deterministic choose digit 1-4. Uses grammar if available.
    """
    try:
        out = llm(
            prompt,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            seed=seed,
            stop=["\n", "\r", "</s>"],
            grammar=_DIGIT_GRAMMAR,  # None is fine
        )
        txt = out["choices"][0]["text"].strip()
        return parse_answer(txt)
    except Exception:
        return None

def llm_notes(llm: Llama, prompt: str, seed: int = 12345) -> str:
    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.05,
        seed=seed,
        stop=["</s>"],
    )
    return out["choices"][0]["text"].strip()

def heuristic_fallback(question: str, options: List[str], contexts: List[str]) -> int:
    base = ((" ".join(contexts)) + " " + (question or "")).lower().replace("ё", "е")
    scores = []
    for opt in options:
        toks = set(tokenize(opt))
        s = sum(1 for t in toks if t in base)
        scores.append(s)
    return int(np.argmax(scores)) + 1


# -----------------------------
# Corpus handling (zip OR dir)
# -----------------------------
def extract_zip_or_use_dir(corpus_path: str, out_dir: str) -> str:
    # Exact dir
    if corpus_path and os.path.isdir(corpus_path):
        return corpus_path

    # Zip
    if corpus_path and corpus_path.lower().endswith(".zip") and os.path.isfile(corpus_path):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(corpus_path, "r") as z:
            z.extractall(out_dir)
        return out_dir

    # Smart fallback if user passed ...zip but it's a folder
    if corpus_path and corpus_path.lower().endswith(".zip"):
        no_zip = corpus_path[:-4]
        if os.path.isdir(no_zip):
            return no_zip
        parent = os.path.dirname(corpus_path)
        if os.path.isdir(parent):
            return parent

    raise FileNotFoundError(
        f"Corpus path not found or unsupported: {corpus_path}\n"
        f"Tip: pass a directory that contains .txt files."
    )

def load_book_texts(root_dir: str) -> Dict[str, str]:
    texts = {}
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, root_dir).replace("\\", "/")
                texts[rel] = safe_read_text(full)
    return texts

def build_indices(book_texts: Dict[str, str], chunk_chars: int, overlap_chars: int):
    basename_map = build_basename_index(book_texts)

    by_book: Dict[str, BookIndex] = {}
    all_chunks: List[str] = []

    # Global index (all texts)
    for _, txt in book_texts.items():
        all_chunks.extend(chunk_text(txt, chunk_chars=chunk_chars, overlap_chars=overlap_chars))

    # Per-book indices
    for book_norm, rel_paths in BOOK_TO_FILES.items():
        combined = []
        for rp in rel_paths:
            resolved = resolve_path(rp, book_texts, basename_map)
            if resolved is not None:
                combined.append(book_texts[resolved])
        if not combined:
            continue

        text = "\n\n".join(combined)
        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        if chunks:
            idx = build_bm25_index(chunks)
            idx.name_norm = book_norm
            by_book[book_norm] = idx

    if not all_chunks:
        raise RuntimeError("No text chunks found. Check corpus_path: it must contain .txt files.")

    global_index = build_bm25_index(all_chunks)
    global_index.name_norm = "__GLOBAL__"
    return by_book, global_index


# -----------------------------
# Main pipeline
# -----------------------------
def run_inference(
    questions_csv: str,
    corpus_path: str,
    output_csv: str,
    model_path: str,

    # retrieval
    top_k: int = 8,
    cand_k: int = 120,
    per_option_k: int = 2,

    # chunking
    chunk_chars: int = 900,
    overlap_chars: int = 180,

    # embeddings
    use_embeddings: int = 1,
    embed_model: str = "intfloat/multilingual-e5-small",
    emb_weight: float = 1.4,

    # llm
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    vote: int = 1,        # grammar makes it deterministic; voting usually unnecessary
    two_pass: int = 1,    # notes -> final digit
    ctx_budget_chars: int = 12000,
):
    work_dir = os.path.join(os.path.dirname(output_csv) or ".", "corpus_extracted")
    corpus_root = extract_zip_or_use_dir(corpus_path, work_dir)

    book_texts = load_book_texts(corpus_root)

    # if texts are in nested dataset/
    if len(book_texts) == 0:
        candidate = os.path.join(corpus_root, "dataset")
        if os.path.isdir(candidate):
            book_texts = load_book_texts(candidate)
            if len(book_texts) > 0:
                corpus_root = candidate

    by_book, global_index = build_indices(book_texts, chunk_chars, overlap_chars)

    # embedding model
    embedder = None
    e5 = False
    if use_embeddings:
        embedder = SentenceTransformer(embed_model)
        e5 = is_e5_model(embed_model)

    df = pd.read_csv(questions_csv)

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    preds = []
    base_seed = 12345

    for _, row in df.iterrows():
        qid = int(row["Unnamed: 0"])
        question = str(row["question"])
        options = [str(row["answer a"]), str(row["answer b"]), str(row["answer c"]), str(row["answer d"])]
        book = str(row["book"])

        book_norm = normalize_ru(book)
        book_norm = BOOK_ALIASES.get(book_norm, book_norm)
        idx = by_book.get(book_norm, global_index)

        contexts = retrieve_strong(
            idx,
            question=question,
            options=options,
            top_k=top_k,
            cand_k=cand_k,
            per_option_k=per_option_k,
            embedder=embedder,
            e5=e5,
            emb_weight=emb_weight,
        )
        contexts = trim_contexts_to_budget(contexts, max_chars=ctx_budget_chars)

        # optional 2-pass reasoning
        seed0 = base_seed + qid * 31
        notes_text = ""
        if two_pass:
            notes_prompt = make_notes_prompt(question, options, contexts)
            notes_text = llm_notes(llm, notes_prompt, seed=seed0)

        prompt = make_prompt(question, options, contexts)
        if notes_text:
            prompt = (
                f"{SYSTEM_RU}\n\n"
                f"Заметки (помогают выбрать вариант):\n{notes_text}\n\n"
                + prompt
            )

        answers = []
        for j in range(max(1, vote)):
            ans = llm_choose_digit(llm, prompt, seed=seed0 + j)
            if ans is not None:
                answers.append(ans)

        if answers:
            ans_final = vote_answers(answers)
        else:
            ans_final = heuristic_fallback(question, options, contexts)

        preds.append({"Unnamed: 0": qid, "answer": int(ans_final)})

    out_df = pd.DataFrame(preds).sort_values("Unnamed: 0")
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} ({len(out_df)} rows)")
    print(f"Book-specific indices built: {len(by_book)} (fallback global for others)")
    if use_embeddings:
        print(f"Embeddings: {embed_model} | e5={e5} | emb_weight={emb_weight}")
    else:
        print("Embeddings: OFF")


# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_predictions(questions_csv: str, pred_csv: str, true_csv: str, print_per_book: int = 1):
    pred = pd.read_csv(pred_csv)
    true = pd.read_csv(true_csv)
    q = pd.read_csv(questions_csv)[["Unnamed: 0", "book"]]
    df = q.merge(pred, on="Unnamed: 0").merge(true, on="Unnamed: 0", suffixes=("_pred","_true"))
    df["ok"] = df["answer_pred"] == df["answer_true"]
    acc = float(df["ok"].mean())
    print(f"Accuracy: {acc:.4f} ({int(df['ok'].sum())}/{len(df)})")
    if print_per_book:
        print(df.groupby("book")["ok"].mean().sort_values())
    return acc


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--corpus_path", type=str, default="")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model", type=str, default="")

    # retrieval
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--cand_k", type=int, default=120)
    ap.add_argument("--per_option_k", type=int, default=2)

    # chunking
    ap.add_argument("--chunk_chars", type=int, default=900)
    ap.add_argument("--overlap_chars", type=int, default=180)

    # embeddings
    ap.add_argument("--use_embeddings", type=int, default=1)
    ap.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-small")
    ap.add_argument("--emb_weight", type=float, default=1.4)

    # llm
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--vote", type=int, default=1)
    ap.add_argument("--two_pass", type=int, default=1)
    ap.add_argument("--ctx_budget_chars", type=int, default=12000)

    # eval
    ap.add_argument("--eval_true", type=str, default="")
    ap.add_argument("--print_per_book", type=int, default=1)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If eval mode and output exists -> evaluate only
    if args.eval_true and os.path.isfile(args.output):
        evaluate_predictions(args.questions, args.output, args.eval_true, print_per_book=args.print_per_book)
    else:
        if not args.corpus_path or not args.model:
            raise ValueError("For inference you must pass --corpus_path (dir or zip) and --model (GGUF).")

        run_inference(
            questions_csv=args.questions,
            corpus_path=args.corpus_path,
            output_csv=args.output,
            model_path=args.model,
            top_k=args.top_k,
            cand_k=args.cand_k,
            per_option_k=args.per_option_k,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            use_embeddings=args.use_embeddings,
            embed_model=args.embed_model,
            emb_weight=args.emb_weight,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            vote=args.vote,
            two_pass=args.two_pass,
            ctx_budget_chars=args.ctx_budget_chars,
        )
