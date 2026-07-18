"""benchmarks/spatialscore.py — SpatialScore adapter.

HF: haoningwu/SpatialScore (dataset)
    SpatialScore_benchmark.ndjson   questions/annotations (~7.6 MB)
    SpatialScore_benchmark.zip      images (~15.8 GB) 

ndjson image_paths are relative ("./CV-Bench/img/..."), and the zip extracts under data_dir
into SpatialScore_benchmark/ -> absolute image path = data_dir/SpatialScore_benchmark/<rel>.

Scoring: shell out to evaluate_results.py (incl. LLM-judge). A pinned copy is vendored
         verbatim at benchmarks/scorers/spatialscore/ (evaluate_results.py + utils/util.py)
         so scoring is self-contained; SS_SCORER overrides it. Needs vllm + util.py's
         torch/torchvision/matplotlib in the active env (see README).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter, register, swift_record

HF_REPO = "haoningwu/SpatialScore"
NDJSON_NAME = "SpatialScore_benchmark.ndjson"
ZIP_NAME = "SpatialScore_benchmark.zip"
IMAGES_SUBDIR = "SpatialScore_benchmark"      
SENTINEL = ".images_extracted"                # mark only when images are successfully unzipped


@register
class SpatialScoreAdapter(BenchmarkAdapter):
    name = "spatialscore"
    # scoring shells out to the vendored evaluate_results.py (vllm judge) + utils/util.py
    # (torch/torchvision/matplotlib) — activate an env that has these (see README).

    _OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

    # Download data if it's not there
    def ensure_data(self) -> None:
        from huggingface_hub import hf_hub_download

        root = self.data_dir
        root.mkdir(parents=True, exist_ok=True)

        # 1) SpatialScore.benchmark.ndjson download
        if not (root / NDJSON_NAME).exists():
            print(f"[spatialscore] downloading {NDJSON_NAME} ...")
            hf_hub_download(HF_REPO, NDJSON_NAME, repo_type="dataset", local_dir=root)

        # 2) Image download
        sentinel = root / SENTINEL
        if sentinel.exists():
            print("[spatialscore] images already extracted")
            return

        print(f"[spatialscore] downloading {ZIP_NAME} (~15.8 GB) ... this can take a while")
        zip_path = Path(hf_hub_download(HF_REPO, ZIP_NAME, repo_type="dataset", local_dir=root))
        print(f"[spatialscore] extracting {zip_path.name} -> {root} ...")
        import zipfile

        with zipfile.ZipFile(zip_path) as zf: # Unzip image zip file
            zf.extractall(root)

        sentinel.write_text("")  # Remove mark only after a success unzip
        try:  # Remove zip file to secure storage
            zip_path.unlink()
            print(f"[spatialscore] removed {zip_path.name} (already unzipped)")
        except OSError as e:
            print(f"[spatialscore] could not remove zip: {e}")

    # Load raw data from the dataset
    def load_raw(self) -> list[dict[str, Any]]:
        nd = self.data_dir / NDJSON_NAME
        if not nd.exists():
            raise FileNotFoundError(
                f"{nd} not found — run `python data_preparation.py spatialscore` first."
            )
        rows: list[dict[str, Any]] = []
        with open(nd, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # For multi-choice, fold the options into the question text as "(A) ... / (B) ..." so the prompt is self-contained as test_qwen.py from SpatialScore
                if item.get("question_type") == "multi-choice" and item.get("options"):
                    q = item.get("question", "")
                    opts = item["options"]
                    # Label each option with a letter, capped at the available letters (A..F).
                    formatted = [
                        f"({self._OPTION_LETTERS[i]}) {o}"
                        for i, o in enumerate(opts)
                        if i < len(self._OPTION_LETTERS)
                    ]
                    if formatted:
                        # Append the lettered options below the question, one per line.
                        item["question"] = q + "\n" + "\n".join(formatted)
                rows.append(item)
        return rows

    # per-question-type instruction (originally from SpatialScore repository)
    @staticmethod
    def _assistant_prompt(sample: dict[str, Any]) -> str:
        qtype = (sample.get("question_type") or "").lower()
        if qtype == "multi-choice":
            return (
                "**Please select the most appropriate answer from the given options.**\n"
                "**Respond ONLY with the capital letter and its parentheses.**\n\nQuestion: "
            )
        if qtype == "judgement":
            return (
                "**Please answer with yes or no based on the image.**\n"
                "**Respond ONLY with 'yes' or 'no'.**\n\nQuestion: "
            )
        if qtype == "open-ended":
            extra = sample.get("extra_info") or {}
            if "answer_value" in extra and "answer_unit" in extra:
                return (
                    "Please answer the question by measuring the precise distance in 3D "
                    "space through 2D images or videos.\nRespond ONLY with a numeric answer "
                    "consisting of a scalar and a distance unit in the format of "
                    "**\\scalar{scalar} \\distance_unit{distance unit}**.\n\nQuestion: "
                )
            return (
                "**Please answer the question based on the given image or video.**\n"
                "**Respond ONLY with a concise and accurate scalar or a scalar with "
                "corresponding unit.**\n\nQuestion: "
            )
        return "Question: "

    def _abs(self, rel: str) -> str:
        # relative ndjson path -> absolute path (ms-swift needs absolute)
        rel = rel[2:] if rel.startswith("./") else rel   # drop leading "./"
        return str((self.data_dir / IMAGES_SUBDIR / rel).resolve())

    # one raw row -> one ms-swift row
    def to_messages(self, row: dict[str, Any]) -> dict[str, Any] | None:
        
        image_paths = row.get("image_paths") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images = [self._abs(p) for p in image_paths if p] 
        

        meta = {
            "id": row.get("id"),
            "answer": row.get("answer"),
            "question_type": row.get("question_type"),
            "category": row.get("category"),
            "task": row.get("task"),
            "sub_task": row.get("sub_task"),
            "input_modality": row.get("input_modality"),
            "source_dataset": row.get("source_dataset"),
            "extra_info": row.get("extra_info") or {},
            "options": row.get("options") or [],
        }

        # Build the user-turn content, mirroring test_qwen.py (see eval_spatialscore_vllm.py):
        #   - interleaved case (question already has one <image> per image, e.g.
        #     view_matching "A) <image> B) <image> ..."): keep the tokens in place so
        #     the option->image mapping is preserved. images stay in token order.
        #   - otherwise: prepend one <image> per image and drop any stray tokens.
        q = row.get("question") or ""
        n_tok = q.count("<image>")
        if n_tok >= 1 and n_tok == len(images):
            user_content = q
        else:
            user_content = "<image>" * len(images) + q.replace("<image>", "")

        # test_qwen.py protocol: the per-question-type instruction goes in a leading
        # assistant-role turn (NOT concatenated into the question).
        messages = [
            {"role": "assistant", "content": self._assistant_prompt(row)},
            {"role": "user", "content": user_content},
        ]
        return swift_record(row.get("id"), images=images, meta=meta, messages=messages)

    # Reshape ms-swift output jsonl -> test_qwen.py (from official SpatialScore) style json
    def reshape(self, preds_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)                 # ensure results dir exists (idempotent)
        entries: list[dict[str, Any]] = []                         # rows to dump into all_results.json

        with open(preds_path, encoding="utf-8") as f:              # ms-swift writes one JSON object per line
            for line in f:                                         # stream predictions line by line
                line = line.strip()                                # trim newline / surrounding whitespace
                if not line:                                       # tolerate stray blank lines
                    continue                                       # skip them

                p = json.loads(line)                               # parse one swift prediction record
                meta = p.get("meta") or {}                         # our scoring payload (from to_messages) lives here

                pred = p.get("response")                           # swift stores the model generation under "response"
                if pred is None:                                   # fallback for versions that only append to messages
                    msgs = p.get("messages") or []                 # the full turn list incl. the model's reply
                    pred = next(                                   # walk turns from the end...
                        (m.get("content", "")                      # ...taking the content of...
                         for m in reversed(msgs)                   # ...the last...
                         if m.get("role") == "assistant"),         # ...assistant turn
                        "",                                        # default to empty string if none found
                    )

                entries.append({                                   # one all_results.json row == test_qwen.py result_entry
                    "id":             p.get("id", meta.get("id")), # sample id (top-level, meta as fallback) -> dedup key
                    "source_dataset": meta.get("source_dataset"),  # aggregation group in the scorer
                    "category":       meta.get("category"),        # aggregation group
                    "task":           meta.get("task"),            # aggregation group
                    "sub_task":       meta.get("sub_task"),        # aggregation group + open-type (counting) detection
                    "input_modality": meta.get("input_modality"),  # single-image / multi-image / video (provenance)
                    "question_type":  meta.get("question_type"),   # multi-choice / judgement / open-ended -> scorer branch
                    "options":        meta.get("options") or [],   # option list -> maps a numeric/text answer back to a letter
                    "extra_info":     meta.get("extra_info") or {},# carries answer_value/answer_unit for distance questions
                    "img_paths":      p.get("images") or [],       # absolute image paths (echoed from the swift record)
                    "video_path":     (p.get("videos") or [None])[0],  # first video if any, else None
                    "gt_answer":      meta.get("answer"),          # ground truth — renamed answer -> gt_answer
                    "pred_answer":    pred,                        # model prediction — renamed response -> pred_answer
                    # NOTE: `question` is intentionally omitted. The scorer only reads it to
                    # recover options when `options` is empty; we supply `options` directly,
                    # so that fallback never triggers and we don't need the folded question.
                })

        out = out_dir / "all_results.json"                        # scorer hardcodes this filename under its --input dir
        with open(out, "w", encoding="utf-8") as f:               # write the reshaped results
            json.dump(entries, f, ensure_ascii=False, indent=2)   # JSON array (evaluate_results.py also accepts ndjson)
        print(f"[spatialscore reshape] {len(entries)} rows -> {out}")  # trace: how many rows were written and where

    def score(self, in_dir: Path) -> dict[str, Any]:
        import os                                                   # stdlib; local import keeps the module top minimal
        import sys                                                  # reuse the active env's python interpreter
        import subprocess                                           # to run the external scorer

        # Resolve the scorer script + its dir (cwd must be that dir so `import utils.util` works).
        # Default: the copy vendored into this repo (benchmarks/scorers/spatialscore/); no machine
        # path is baked in. Override with SS_SCORER only to point at a different checkout.
        vendored = Path(__file__).resolve().parent / "scorers" / "spatialscore" / "evaluate_results.py"
        scorer = Path(os.environ.get("SS_SCORER", str(vendored))).resolve()  # env override else vendored
        if not scorer.exists():                                     # missing -> say exactly what's wrong
            raise FileNotFoundError(f"scorer not found: {scorer} (SS_SCORER override, or vendored copy missing)")
        repo = scorer.parent                                        # utils/ lives next to evaluate_results.py

        # Build the command. --input is the dir that holds all_results.json (== our results_dir).
        cmd = [sys.executable, str(scorer), "--input", str(in_dir)] # base invocation
        if os.environ.get("SS_NO_LLM"):                            # rule-only fast path (no Stage-2 judge)
            cmd.append("--no_llm")                                  # skip the vllm LLM entirely
        else:                                                       # otherwise wire the Stage-2 judge LLM
            cmd += ["--llm_path", os.environ.get(                   # judge LLM: HF repo id (cache-resolved), not a local path
                "SS_LLM_PATH", "openai/gpt-oss-20b")]
            cmd += ["--tp_size", os.environ.get("SS_TP_SIZE", "1")]              # judge tensor-parallel size
            cmd += ["--gpu_memory_utilization", os.environ.get("SS_GPU_MEM", "0.5")]  # judge gpu mem util

        print(f"[spatialscore score] running: {' '.join(cmd)} (cwd={repo})")  # trace the exact invocation
        subprocess.run(cmd, cwd=repo, check=True)                   # run scorer; raise if it exits non-zero

        # evaluate_results.py wrote summary_report.json into --input; read it back as our metrics dict.
        summary_path = in_dir / "summary_report.json"              # {category,task,sub_task,source_dataset,overall}
        if not summary_path.exists():                              # scorer "succeeded" but produced nothing -> surface it
            raise RuntimeError(f"scorer produced no summary_report.json in {in_dir}")
        with open(summary_path, encoding="utf-8") as f:            # load the summary the scorer wrote
            return json.load(f)                                    # return metrics dict (evaluate.py records it)
