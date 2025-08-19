import datetime
import math
import os
import pathlib
from typing import Dict, List

import yaml

try:
    import tiktoken
except ImportError:
    tiktoken = None
import openai

from bibcat import config
from bibcat.llm.openai import OpenAIHelper
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

# create the helper
oa = OpenAIHelper()


class Chunker:
    """Token-aware batch file chunker and batch manager.

    Usage:
        c = Chunker(input_file_path, output_dir=None, model='gpt-4.1-mini', ...)
        c.analyze()
        c.create_chunks()
        c.create_daily_batches()
        c.verify_chunks()

    After running, inspect attributes on the instance for details.
    """

    def __init__(
        self,
        input_file_path: str,
        max_lines: int = 50000,
        max_size_mb: int = 200,
        max_tokens_per_day: int = 40_000_000,
        model: str = "gpt-4.1-mini",
        days_to_distribute: int = 1,
    ) -> None:
        # setup inputs and outputs
        self.input_file_path = pathlib.Path(input_file_path)
        self.output_dir = self.input_file_path.parent/'batch_chunks'
        self.model = model
        self.all_output_files = []
        self.daily_batches = []

        # limits
        self.max_lines = max_lines
        self.max_size_mb = max_size_mb
        self.max_tokens_per_day = max_tokens_per_day
        self.days_to_distribute = days_to_distribute

        # Analysis / state fields
        self.file_size_bytes: int = 0
        self.file_size_mb: float = 0.0
        self.total_lines: int = 0
        self.avg_tokens_per_line: float = 0.0
        self.total_estimated_tokens: int = 0

        self.base_chunks_needed: int = 0
        self.lines_per_chunk: int = 0
        self.estimated_tokens_per_chunk: int = 0
        self.chunks_per_day: int = 0
        self.days_needed: int = 0

        self.total_actual_tokens: int = 0

        # Ensure output dir exists and check for any existing chunks
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_output_files = list(sorted(self.output_dir.glob('*.jsonl')))

        # submission tracking
        self.submitted_chunks: List[str] = []
        self.submission_log: List[Dict] = []

        # If chunks already exist, try to recover plan state from YAML
        if self.all_output_files:
            plan_path = self.output_dir / 'chunk_plan.yaml'
            if plan_path.exists():
                try:
                    self._load_plan(plan_path)
                    logger.info("Loaded plan state from %s", plan_path)
                except Exception as e:
                    logger.warning("Failed to load plan state from %s: %s", plan_path, e)

    # --- Token estimation helper ---
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text using tiktoken. Falls back gracefully."""
        if tiktoken is None:
            raise ImportError("tiktoken is required for token estimation. Install with: pip install tiktoken")
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    # --- Analysis ---
    def analyze(self, sample_lines: int = 1000) -> Dict:
        """Analyze file: lines, size, and estimate tokens using a sample of lines.

        Returns a dict with analysis results and stores to instance attributes.
        """
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

        self.file_size_bytes = os.path.getsize(self.input_file_path)
        self.file_size_mb = self.file_size_bytes / (1024 * 1024)

        # sample up to `sample_lines` lines
        total_lines = 0
        sample_tokens = 0
        sample_count = 0

        with open(self.input_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines += 1
                if i < sample_lines:
                    stripped = line.strip()
                    if stripped:
                        try:
                            sample_tokens += self._estimate_tokens(stripped)
                        except Exception:
                            # If token estimation fails, treat tokens ~= words
                            sample_tokens += max(1, len(stripped.split()))
                    sample_count += 1

        self.total_lines = total_lines
        if sample_count > 0:
            self.avg_tokens_per_line = sample_tokens / sample_count
            self.total_estimated_tokens = int(self.avg_tokens_per_line * self.total_lines)
        else:
            self.avg_tokens_per_line = 0.0
            self.total_estimated_tokens = 0

        logger.info("File analysis:")
        logger.info("  Total lines: %s", self.total_lines)
        logger.info("  File size: %.2f MB", self.file_size_mb)
        logger.info("  Estimated total tokens: %s", self.total_estimated_tokens)
        logger.info("  Average tokens per line: %.1f", self.avg_tokens_per_line)

        return {
            "total_lines": self.total_lines,
            "file_size_mb": self.file_size_mb,
            "total_estimated_tokens": self.total_estimated_tokens,
            "avg_tokens_per_line": self.avg_tokens_per_line,
        }

    # --- Chunk calculation and creation ---
    def plan_chunks(self) -> Dict:
        """Compute chunking parameters based on constraints and store them."""
        max_size_bytes = self.max_size_mb * 1024 * 1024

        chunks_needed_for_lines = math.ceil(self.total_lines / max(1, self.max_lines))
        chunks_needed_for_size = math.ceil(self.file_size_bytes / max(1, max_size_bytes))

        # Conservative tokens per chunk estimate
        if self.avg_tokens_per_line > 0:
            chunks_needed_for_tokens = math.ceil(self.total_estimated_tokens / max(1, self.max_tokens_per_day))
        else:
            chunks_needed_for_tokens = 1

        self.base_chunks_needed = max(chunks_needed_for_lines, chunks_needed_for_size, chunks_needed_for_tokens, 1)
        self.lines_per_chunk = math.ceil(self.total_lines / self.base_chunks_needed)
        if self.base_chunks_needed > 0:
            self.estimated_tokens_per_chunk = math.ceil(self.total_estimated_tokens / self.base_chunks_needed)
        else:
            self.estimated_tokens_per_chunk = self.total_estimated_tokens

        # Determine chunks_per_day and days needed
        self.chunks_per_day = max(1, min(self.base_chunks_needed, self.max_tokens_per_day // max(1, self.estimated_tokens_per_chunk)))
        self.days_needed = max(self.days_to_distribute, math.ceil(self.base_chunks_needed / self.chunks_per_day))

        logger.info("Chunking plan:")
        logger.info("  Base chunks needed: %s", self.base_chunks_needed)
        logger.info("  Lines per chunk: %s", self.lines_per_chunk)
        logger.info("  Estimated tokens per chunk: %s", self.estimated_tokens_per_chunk)
        logger.info("  Chunks per day: %s", self.chunks_per_day)
        logger.info("  Days needed: %s", self.days_needed)

        return {
            "base_chunks_needed": self.base_chunks_needed,
            "lines_per_chunk": self.lines_per_chunk,
            "estimated_tokens_per_chunk": self.estimated_tokens_per_chunk,
            "chunks_per_day": self.chunks_per_day,
            "days_needed": self.days_needed,
        }

    def create_chunks(self) -> List[str]:
        """Create chunk files on disk according to computed lines_per_chunk."""
        if self.lines_per_chunk <= 0:
            raise ValueError("lines_per_chunk must be computed and > 0; call analyze() and plan_chunks() first")

        base = pathlib.Path(self.input_file_path).stem
        ext = pathlib.Path(self.input_file_path).suffix

        self.all_output_files = []
        chunk_num = 1
        current_count = 0
        current_out = None

        with open(self.input_file_path, "r", encoding="utf-8") as infile:
            try:
                for line in infile:
                    if current_count == 0:
                        if current_out:
                            current_out.close()
                        chunk_filename = f"{base}_chunk_{chunk_num:03d}{ext}"
                        chunk_path = os.path.join(self.output_dir, chunk_filename)
                        current_out = open(chunk_path, "w", encoding="utf-8")
                        self.all_output_files.append(chunk_path)
                        logger.info("Creating chunk %s: %s", chunk_num, chunk_filename)

                    current_out.write(line)
                    current_count += 1

                    if current_count >= self.lines_per_chunk:
                        current_count = 0
                        chunk_num += 1
            finally:
                if current_out:
                    current_out.close()

        logger.info("Created %s chunk files", len(self.all_output_files))

        # save plan state for future recovery
        try:
            self._save_plan(self.output_dir / 'chunk_plan.yaml')
            logger.info("Saved chunk plan to %s", self.output_dir / 'chunk_plan.yaml')
        except Exception as e:
            logger.warning("Failed to save chunk plan: %s", e)

        return self.all_output_files

    # --- Daily batching ---
    def create_daily_batches(self) -> List[List[str]]:
        """Organize chunks into daily batches based on chunks_per_day and days_needed."""
        if not self.all_output_files:
            raise ValueError("No chunk files found; run create_chunks() first")

        self.daily_batches = []
        for day in range(self.days_needed):
            start_idx = day * self.chunks_per_day
            end_idx = min(start_idx + self.chunks_per_day, len(self.all_output_files))
            if start_idx < len(self.all_output_files):
                self.daily_batches.append(self.all_output_files[start_idx:end_idx])

        logger.info("Organized chunks into %s daily batches", len(self.daily_batches))
        return self.daily_batches

    # --- Verification and token accounting ---
    def verify_chunks(self) -> Dict:
        """Verify each chunk against limits and estimate actual tokens per chunk."""
        if not self.all_output_files:
            raise ValueError("No chunks to verify")

        total_tokens = 0
        for i, chunk_path in enumerate(self.all_output_files, start=1):
            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk_lines = sum(1 for _ in f)

            # sample lines for token estimate
            sample_lines = []
            with open(chunk_path, "r", encoding="utf-8") as f:
                for j, line in enumerate(f):
                    if j < 100:
                        sample_lines.append(line.strip())
                    elif j % (max(1, chunk_lines // 100)) == 0 and len(sample_lines) < 200:
                        sample_lines.append(line.strip())

            if sample_lines:
                try:
                    sample_tokens = sum(self._estimate_tokens(line_text) for line_text in sample_lines if line_text)
                except Exception:
                    sample_tokens = sum(max(1, len(line_text.split())) for line_text in sample_lines if line_text)
                chunk_tokens = int((sample_tokens / len(sample_lines)) * chunk_lines)
            else:
                chunk_tokens = 0

            total_tokens += chunk_tokens

            logger.info("  Chunk %s: %s lines, %.2f MB, ~%s tokens", i, chunk_lines, chunk_size_mb, chunk_tokens)

            if chunk_lines > self.max_lines:
                raise ValueError(f"Chunk {i} exceeds line limit: {chunk_lines} > {self.max_lines}")
            if chunk_size_mb > self.max_size_mb:
                raise ValueError(f"Chunk {i} exceeds size limit: {chunk_size_mb:.2f} MB > {self.max_size_mb} MB")

        self.total_actual_tokens = total_tokens
        logger.info("Total estimated tokens across all chunks: %s", self.total_actual_tokens)

        return {"total_actual_tokens": self.total_actual_tokens}

    # --- Utilities for submission tracking ---
    def remaining_chunks_count(self) -> int:
        """Return number of chunks not yet submitted/processed (naive: all chunks)."""
        return len(self.all_output_files)

    def get_submission_schedule(self) -> Dict:
        """Return the planned submission schedule and strategy info."""
        return {
            "total_tokens": self.total_actual_tokens,
            "total_chunks": len(self.all_output_files),
            "days_needed": self.days_needed,
            "chunks_per_day": self.chunks_per_day,
            "strategy": "multi_day_batching" if self.days_needed > 1 else "single_day_chunking",
            "daily_batches": self.daily_batches,
        }

    # # --- Merge responses ---
    # def merge_responses(self, response_files: List[str], output_path: Optional[str] = None) -> str:
    #     """Merge multiple response files (JSONL or json) into a single JSONL file for evaluation.

    #     Returns path to merged file.
    #     """
    #     out_path = output_path or os.path.join(self.output_dir, "merged_responses.jsonl")
    #     with open(out_path, "w", encoding="utf-8") as out:
    #         for rf in response_files:
    #             if not os.path.exists(rf):
    #                 logger.warning("Response file not found, skipping: %s", rf)
    #                 continue
    #             with open(rf, "r", encoding="utf-8") as inf:
    #                 for line in inf:
    #                     out.write(line.rstrip() + "\n")
    #     logger.info("Merged %s response files into %s", len(response_files), out_path)
    #     return out_path

    # def merge_and_evaluate(self, response_files: List[str], evaluator_callable) -> Dict:
    #     """Merge response files and run an evaluator callable over merged data.

    #     evaluator_callable takes merged_file_path and returns evaluation results.
    #     """
    #     merged = self.merge_responses(response_files)
    #     results = evaluator_callable(merged)
    #     return results

    def prepare_all(self, sample_lines: int = 1000, run_verification: bool = True):
        """Run full pipeline: analyze -> plan -> create chunks -> verify -> create daily batches.

        Returns the submission schedule dict.
        """
        self.analyze(sample_lines=sample_lines)
        self.plan_chunks()
        self.create_chunks()
        if run_verification:
            try:
                self.verify_chunks()
            except Exception as e:
                logger.warning("Verification failed: %s", e)
        self.create_daily_batches()
        return self.get_submission_schedule()

    def plan_info(self) -> Dict:
        """Log and return a consolidated plan summary for easy inspection."""
        # Log file analysis in the same format used elsewhere
        logger.info("File analysis:")
        logger.info("  Total lines: %s", self.total_lines)
        logger.info("  File size: %.2f MB", self.file_size_mb)
        logger.info("  Estimated total tokens: %s", self.total_estimated_tokens)
        logger.info("  Average tokens per line: %.1f", self.avg_tokens_per_line)

        # Log chunking plan
        logger.info("Chunking plan:")
        logger.info("  Base chunks needed: %s", self.base_chunks_needed)
        logger.info("  Lines per chunk: %s", self.lines_per_chunk)
        logger.info("  Estimated tokens per chunk: %s", self.estimated_tokens_per_chunk)
        logger.info("  Chunks per day: %s", self.chunks_per_day)
        logger.info("  Days needed: %s", self.days_needed)

        # Log daily batches with per-day file lists
        logger.info("Batch Info:")
        logger.info("  Number of daily batches: %s", len(self.daily_batches))
        logger.info("  Submitted Batches: %s", len(self.submitted_chunks))
        logger.info("  Remaining Batches: %s", len(self._get_pending_batches()))
        logger.info("  Completed Batches: %s", len(self.completed_batches))

        info = {
            "file": str(self.input_file_path),
            "total_lines": self.total_lines,
            "file_size_mb": self.file_size_mb,
            "total_estimated_tokens": self.total_estimated_tokens,
            "avg_tokens_per_line": self.avg_tokens_per_line,
            "base_chunks_needed": self.base_chunks_needed,
            "lines_per_chunk": self.lines_per_chunk,
            "estimated_tokens_per_chunk": self.estimated_tokens_per_chunk,
            "chunks_per_day": self.chunks_per_day,
            "days_needed": self.days_needed,
            "total_actual_tokens": self.total_actual_tokens,
            "total_chunks": len(self.all_output_files),
            "n_daily_batches": len(self.daily_batches),
            "n_submitted_batches": len(self.submitted_chunks),
            "n_remaining_batches": len(self._get_pending_batches()),
            "n_completed_batches": len(self.completed_batches),
        }

        return info

    # --- Submission simulation ---
    def _get_pending_batches(self) -> List[List[str]]:
        """Return list of pending chunk batches (exclude already submitted)."""
        if not self.all_output_files:
            logger.info("No chunk files found, running prepare_all().")
            self.prepare_all()

        pending = [p for p in self.all_output_files if p not in self.submitted_chunks]
        if not pending:
            return []
        return [pending[i : i + self.chunks_per_day] for i in range(0, len(pending), self.chunks_per_day)]

    def get_next_batch(self) -> List[str]:
        """Return the next pending batch (full paths)."""
        batches = self._get_pending_batches()
        return batches[0] if batches else []

    def get_status(self) -> Dict:
        """Return submission status: submitted count, remaining count, and next batch info."""
        total = len(self.all_output_files)
        submitted = len(self.submitted_chunks)
        remaining = total - submitted
        next_batch = [os.path.basename(p) for p in self.get_next_batch()]
        chunks_today = self._chunks_submitted_today()
        # estimate tokens today using estimated_tokens_per_chunk as a guide
        tokens_today_est = int(chunks_today * max(1, self.estimated_tokens_per_chunk))
        return {
            "total_chunks": total,
            "submitted_chunks": submitted,
            "remaining_chunks": remaining,
            "next_batch_files": next_batch,
            "chunks_submitted_today": chunks_today,
            "tokens_submitted_today_estimate": tokens_today_est,
            "tokens_allowed_per_day": self.max_tokens_per_day,
            "tokens_remaining_today_estimate": max(0, self.max_tokens_per_day - tokens_today_est),
        }

    def _chunks_submitted_today(self) -> int:
        """Return number of chunks recorded in submission_log for today's date."""
        today_prefix = datetime.date.today().isoformat()
        return sum(1 for item in self.submission_log if str(item.get("timestamp", "")).startswith(today_prefix))

    def _estimate_chunk_tokens(self, chunk_path: str) -> int:
        """Estimate tokens for a single chunk based on sampled lines and avg_tokens_per_line."""
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
        except OSError:
            return 0
        return int(self.avg_tokens_per_line * lines) if self.avg_tokens_per_line > 0 else 0

    def _submit_chunk(self, chunk_path: str) -> Dict:
        """Submit a single chunk enforcing plan-based per-day limits.

        Uses estimated_tokens_per_chunk and chunks_per_day as the daily safeguards.
        """
        # use estimates rather than exact token counts
        est_tokens = max(1, int(self.estimated_tokens_per_chunk))

        chunks_today = self._chunks_submitted_today()
        if self.chunks_per_day > 0 and chunks_today >= self.chunks_per_day:
            msg = f"Daily chunk submission limit reached: {chunks_today} >= {self.chunks_per_day}"
            logger.error(msg)
            return {"chunk": str(chunk_path), "status": "rejected", "error": msg}

        tokens_today_est = chunks_today * est_tokens
        if tokens_today_est + est_tokens > self.max_tokens_per_day:
            msg = (
                f"Estimated submission would exceed daily token limit: today_est={tokens_today_est}, "
                f"chunk_est={est_tokens}, limit={self.max_tokens_per_day}"
            )
            logger.error(msg)
            return {"chunk": str(chunk_path), "status": "rejected", "error": msg}

        #import uuid

        # perform submission
        try:
            oa.submit_batch(batch_file=str(chunk_path))
        except Exception as e:
            logger.error("Error submitting chunk %s: %s", chunk_path, e)
            return {"chunk": str(chunk_path), "status": "error", "error": str(e)}
        else:
            #oa.batch=type('A', (), {'id':str(uuid.uuid1())})()
            self.batch_id = oa.batch.id

        # success: record submission using estimated tokens
        self.submitted_chunks.append(chunk_path)
        entry = {"chunk": os.path.basename(chunk_path), "timestamp": datetime.datetime.utcnow().isoformat(),
                 "tokens": est_tokens, "batch_id": self.batch_id}
        self.submission_log.append(entry)
        try:
            self._save_plan(self.output_dir / 'chunk_plan.yaml')
        except Exception as e:
            logger.warning("Failed to update saved plan after submission: %s", e)
        return {"chunk": str(chunk_path), "status": "submitted", "tokens_estimated": est_tokens, "batch_id": self.batch_id}

    def run_submission_schedule(self, dry_run: bool = True) -> List[Dict]:
        """Run the submission schedule for up to `days` (default 1).

        By default this submits only the next day's batch so you can run once per day
        and resume later. Set days>1 to process more days in one call.
        """
        batch = self.get_next_batch()
        if not batch:
            logger.info("No pending chunks to submit. All %s chunks already submitted.", len(self.submitted_chunks))
            return []

        # get current day
        batch_num = 1 + self.daily_batches.index(batch)
        if self.submission_log:
            first = datetime.datetime.fromisoformat(self.submission_log[0]['timestamp'])
        else:
            first = datetime.datetime.today()
        today = datetime.datetime.today()
        days = 1 + (today.date() - first.date()).days

        batch_tokens = sum(self._estimate_chunk_tokens(chunk) for chunk in batch)
        results: List[Dict] = []
        logger.info("--- Day %s Batch %s: submitting %s chunks (~%s tokens) ---", days, batch_num, len(batch), batch_tokens)
        for chunk in batch:
            #batch_tokens = sum(self._estimate_chunk_tokens(c) for c in batch)

            logger.info("Submitting chunk: %s", os.path.basename(chunk))
            if dry_run:
                results.append({"chunk": str(chunk), "status": "dry-run"})
                continue

            res = self._submit_chunk(chunk)
            results.append(res)

        if results[0]['status'] != 'rejected':
            logger.info("Submission run complete. %s records generated.", len(results))

        return results

    # --- Plan save/load ---
    def _save_plan(self, path: pathlib.Path) -> None:
        """Save current plan state to YAML grouped by category for future recovery."""
        plan = {
            "file_input": {
                "file": str(self.input_file_path),
                "file_size_mb": self.file_size_mb,
                "total_lines": self.total_lines,
                "avg_tokens_per_line": self.avg_tokens_per_line,
                "total_estimated_tokens": self.total_estimated_tokens,
            },
            "chunk_plan": {
                "base_chunks_needed": self.base_chunks_needed,
                "lines_per_chunk": self.lines_per_chunk,
                "estimated_tokens_per_chunk": self.estimated_tokens_per_chunk,
                "total_chunks": len(self.all_output_files),
                "all_chunks": [os.path.basename(p) for p in self.all_output_files],
            },
            "submission_strategy": {
                "chunks_per_day": self.chunks_per_day,
                "days_needed": self.days_needed,
                "total_actual_tokens": self.total_actual_tokens,
                "daily_batches": [[os.path.basename(p) for p in batch] for batch in self.daily_batches],
                "submitted_chunks": [os.path.basename(p) for p in self.submitted_chunks],
                "submission_log": list(self.submission_log),
                "completed_chunks": [os.path.basename(p) for p in self.completed_batches],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(plan, f)

    def _load_plan(self, path: pathlib.Path) -> None:
        """Load plan state from YAML grouped structure and restore instance attributes."""
        with open(path, "r", encoding="utf-8") as f:
            plan = yaml.safe_load(f) or {}

        file_input = plan.get("file_input", {})
        chunk_plan = plan.get("chunk_plan", {})
        submission = plan.get("submission_strategy", {})

        # restore file input info
        self.total_lines = file_input.get("total_lines", self.total_lines)
        self.file_size_mb = file_input.get("file_size_mb", self.file_size_mb)
        self.total_estimated_tokens = file_input.get("total_estimated_tokens", self.total_estimated_tokens)
        self.avg_tokens_per_line = file_input.get("avg_tokens_per_line", self.avg_tokens_per_line)

        # restore chunk plan
        self.base_chunks_needed = chunk_plan.get("base_chunks_needed", self.base_chunks_needed)
        self.lines_per_chunk = chunk_plan.get("lines_per_chunk", self.lines_per_chunk)
        self.estimated_tokens_per_chunk = chunk_plan.get("estimated_tokens_per_chunk", self.estimated_tokens_per_chunk)

        all_chunks = chunk_plan.get("all_chunks", [])
        self.all_output_files = [str(self.output_dir / c) for c in all_chunks]

        # restore submission strategy
        self.chunks_per_day = submission.get("chunks_per_day", self.chunks_per_day)
        self.days_needed = submission.get("days_needed", self.days_needed)
        self.total_actual_tokens = submission.get("total_actual_tokens", self.total_actual_tokens)

        daily = submission.get("daily_batches", [])
        self.daily_batches = [[str(self.output_dir / c) for c in batch] for batch in daily]

        # restore submitted chunks
        submitted = submission.get("submitted_chunks", [])
        self.submitted_chunks = [str(self.output_dir / c) for c in submitted]

        # restore submission log
        self.submission_log = submission.get("submission_log", [])

        # ensure base_chunks_needed matches actual number of chunks
        self.base_chunks_needed = max(self.base_chunks_needed, len(self.all_output_files))

    def check_batches_status(self) -> List[Dict]:
        """Check status of submitted batches using the OpenAI client.

        Returns a list of dicts: {"chunk": ..., "batch_id": ..., "status": ...}
        """
        remote_batches = {batch.id: batch.status for batch in oa.client.batches.list()}

        results = []
        for entry in self.submission_log:
            if entry['batch_id'] not in remote_batches:
                continue
            results.append({"chunk": entry.get("chunk"), "batch_id": entry.get("batch_id"), "status": remote_batches[entry['batch_id']]})
        return results

    def retrieve_batch_results(self) -> List[Dict]:
        """Retrieve completed batches using OpenAIHelper.retrieve_batch (call left commented).

        For each logged batch_id this will attempt to retrieve the batch results. The actual
        retrieval call is commented out so you can modify oa.retrieve_batch before enabling.
        Returns a list of dicts with chunk, batch_id and a note about retrieval attempt.
        """
        results: List[Dict] = []
        for idx, entry in enumerate(self.submission_log, start=1):
            bid = entry.get("batch_id")
            if not bid:
                continue

            output = self.output_dir/f'{config.llms.prompt_output_file.replace(".json", "")}_chunk_{idx:>03}.json'
            try:
                oa.retrieve_batch(batch_id=bid, output=output)
                results.append({"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": False, "note": "call commented out"})
            except openai.APIStatusError as e:
                logger.error("Error retrieving batch %s: %s", bid, e)
                results.append({"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": False, "error": str(e)})
            else:
                results.append({"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": True, "output": output})

        return results

    @property
    def completed_batches(self) -> List[str]:
        """List of completed batch file paths."""
        return [str(i) for i in sorted(self.output_dir.glob('*chunk*.json'))]
