import datetime
import itertools
import math
import os
import pathlib
import time
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


class ChunkPlanner:
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
        self.output_dir = self.input_file_path.parent / "batch_chunks2"
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
        self.all_output_files = list(sorted(self.output_dir.glob("*.jsonl")))

        # submission tracking
        self.submitted_chunks: List[str] = []
        self.submission_log: List[Dict] = []

        # If chunks already exist, try to recover plan state from YAML
        if self.all_output_files:
            plan_path = self.output_dir / "chunk_plan.yaml"
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

    def _sample_file(self, file, n_sample: int = 100) -> tuple:
        """Sample lines from a file for token estimation.

        Reads a file and samples a specified number of lines
        to estimate the token count. It returns the total number of lines,
        the average tokens per line, and the estimated tokens for the sampled
        lines.

        Parameters
        ----------
        file : _type_
            the input file to sample
        n_sample : int, optional
            the number of lines to sample, by default 100

        Returns
        -------
        tuple
            A tuple of total number of lines, average tokens per line, and estimated tokens.
        """

        # read the file
        with open(file, "r", encoding="utf-8") as f:
            # sample first N lines
            sample_lines = [ln.strip() for ln in itertools.islice(f, n_sample) if ln.strip()]

            # count remaining lines
            sampled_count = len(sample_lines)
            remaining = sum(1 for _ in f)
            total_lines = sampled_count + remaining

            # estimate tokens for the whole sample
            sample_text = "\n".join(sample_lines)
            try:
                sample_tokens = self._estimate_tokens(sample_text)
            except Exception:
                # If token estimation fails, treat tokens ~= words
                sample_tokens = sum(max(1, len(l.split())) for l in sample_lines)

            sample_count = len([l for l in sample_lines if l.strip()])

            # estimate the average tokens per line and total tokens
            avg_tokens_per_line = (sample_tokens / sample_count) if sample_count else 0.0
            estimated_tokens = int(avg_tokens_per_line * total_lines)

            return total_lines, avg_tokens_per_line, estimated_tokens

    # --- File Analysis ---
    def analyze(self, sample_lines: int = 1000) -> dict:
        """Analyze the main batch JSONL

        Analyze the input file to get the number of lines, the filesize,
        and to estimate the number of tokens.  Uses a sample of input
        lines to estimate the tokens.

        Parameters
        ----------
        sample_lines : int, optional
            the number of lines to sample, by default 1000

        Returns
        -------
        dict
            stats on input file

        Raises
        ------
        FileNotFoundError
            when input file doesn't exist
        """
        if not self.input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

        self.file_size_bytes = self.input_file_path.stat().st_size
        self.file_size_mb = self.file_size_bytes / (1024 * 1024)

        # sample the input file
        total_lines, avg_tokens_per_line, estimated_tokens = self._sample_file(self.input_file_path, sample_lines)

        # estimate the average tokens per line and total tokens
        self.total_lines = total_lines
        self.avg_tokens_per_line = avg_tokens_per_line
        self.total_estimated_tokens = estimated_tokens

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
    def plan_chunks(self) -> dict:
        """Plan the chunking of the input file.

        Using the input file estimates, plans best how to split it into
        chunks based on the various constraints and rate limits, such as
        max lines, max size, and estimated tokens.

        Returns
        -------
        dict
            stats on the planned chunks
        """
        # the max filesize in bytes
        max_size_bytes = self.max_size_mb * 1024 * 1024

        # estimate number of chunks based on line and size rate limits
        chunks_needed_for_lines = math.ceil(self.total_lines / max(1, self.max_lines))
        chunks_needed_for_size = math.ceil(self.file_size_bytes / max(1, max_size_bytes))

        # Conservative tokens per chunk estimate
        if self.avg_tokens_per_line > 0:
            chunks_needed_for_tokens = math.ceil(self.total_estimated_tokens / max(1, self.max_tokens_per_day))
        else:
            chunks_needed_for_tokens = 1

        # Estimate optimal number of file chunks needed
        self.base_chunks_needed = max(chunks_needed_for_lines, chunks_needed_for_size, chunks_needed_for_tokens, 1)
        self.lines_per_chunk = math.ceil(self.total_lines / self.base_chunks_needed)
        if self.base_chunks_needed > 0:
            self.estimated_tokens_per_chunk = math.ceil(self.total_estimated_tokens / self.base_chunks_needed)
        else:
            self.estimated_tokens_per_chunk = self.total_estimated_tokens

        # Determine chunks_per_day and days needed
        self.chunks_per_day = max(
            1, min(self.base_chunks_needed, self.max_tokens_per_day // max(1, self.estimated_tokens_per_chunk))
        )
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

    def create_chunks(self) -> list[str]:
        """Split the input file into subset chunks

        Reads the input file and splits it into smaller chunks
        based on the lines_per_chunk attribute. Each chunk is written to a
        separate file.

        Returns
        -------
        list[str]
            the list of output chunked file subsets

        Raises
        ------
        ValueError
            when no plan has been created
        """
        # error if the chunk plan has not been created
        if self.lines_per_chunk <= 0:
            raise ValueError("No chunk plan yet made. Call analyze() and plan_chunks() first")

        # set base output chunk parts
        base = pathlib.Path(self.input_file_path).stem
        ext = pathlib.Path(self.input_file_path).suffix
        self.all_output_files = []

        # chunk the input file into a number of subset files
        with open(self.input_file_path, "r", encoding="utf-8") as infile:
            # iterate over number of needed chunks
            for i in range(self.base_chunks_needed):
                # slice the input file data into lines per chunk
                chunk_lines = list(itertools.islice(infile, self.lines_per_chunk))
                if not chunk_lines:
                    break
                chunk_path = self.output_dir / f"{base}_chunk_{i + 1:03d}{ext}"
                # write chunk
                with open(chunk_path, "w", encoding="utf-8") as out:
                    out.writelines(chunk_lines)
                self.all_output_files.append(chunk_path)
                logger.info("Creating chunk %s: %s", i + 1, chunk_path)

        logger.info("Created %s chunk files", len(self.all_output_files))

        # save plan state to disk
        try:
            self._save_plan(self.output_dir / "chunk_plan.yaml")
            logger.info("Saved chunk plan to %s", self.output_dir / "chunk_plan.yaml")
        except Exception as e:
            logger.warning("Failed to save chunk plan: %s", e)

        return self.all_output_files

    # --- Verification and token accounting ---
    def verify_chunks(self) -> Dict:
        """Verify each chunk against limits and estimate actual tokens per chunk."""
        if not self.all_output_files:
            raise ValueError("No output chunks to verify.")

        n_sample = 50
        total_tokens = 0
        # iterate over all chunks
        for i, chunk_path in enumerate(self.all_output_files, start=1):
            chunk_size_mb = pathlib.Path(chunk_path).stat().st_size / (1024 * 1024)
            # sample file
            chunk_lines, __, chunk_tokens = self._sample_file(chunk_path, n_sample)
            total_tokens += chunk_tokens
            logger.info("  Chunk %s: %s lines, %.2f MB, ~%s tokens", i, chunk_lines, chunk_size_mb, chunk_tokens)

            if chunk_lines > self.max_lines:
                raise ValueError(f"Chunk {i} exceeds line limit: {chunk_lines} > {self.max_lines}")
            if chunk_size_mb > self.max_size_mb:
                raise ValueError(f"Chunk {i} exceeds size limit: {chunk_size_mb:.2f} MB > {self.max_size_mb} MB")

        self.total_actual_tokens = total_tokens
        logger.info("Total estimated tokens across all chunks: %s", self.total_actual_tokens)

        return {"total_actual_tokens": self.total_actual_tokens}

    # --- Daily batching ---
    def create_daily_batches(self) -> list[list[str]]:
        """Organize chunks into daily batches based on chunks_per_day and days_needed.

        Splits the file chunks into daily batches that fit
        within the daily token limits, as a list of lists of files.

        Returns
        -------
        list[list[str]]
            A list of daily batches, each containing chunk file paths.

        Raises
        ------
        ValueError
            when no input chunks are available
        """
        if not self.all_output_files:
            raise ValueError("No chunk files found; run create_chunks() first")

        self.daily_batches = []
        # iterate over days
        for day in range(self.days_needed):
            # split the chunks into daily batches
            start_idx = day * self.chunks_per_day
            end_idx = min(start_idx + self.chunks_per_day, len(self.all_output_files))
            if start_idx < len(self.all_output_files):
                self.daily_batches.append(self.all_output_files[start_idx:end_idx])

        logger.info("Organized chunks into %s daily batches", len(self.daily_batches))
        return self.daily_batches

    def get_submission_schedule(self) -> dict:
        """Return the planned submission schedule and strategy info."""

        return {
            "total_tokens": self.total_actual_tokens,
            "total_chunks": len(self.all_output_files),
            "days_needed": self.days_needed,
            "chunks_per_day": self.chunks_per_day,
            "strategy": "multi_day_batching" if self.days_needed > 1 else "single_day_chunking",
            "daily_batches": self.daily_batches,
        }

    def prepare_all(self, sample_lines: int = 1000) -> dict:
        """Runs the chunk preparation pipeline.

        Parameters
        ----------
        sample_lines : int, optional
            the number of lines to sample, by default 1000

        Returns
        -------
        dict
            stats on the submission plan
        """
        self.analyze(sample_lines=sample_lines)
        self.plan_chunks()
        self.create_chunks()
        self.verify_chunks()
        self.create_daily_batches()
        return self.get_submission_schedule()

    def plan_info(self) -> dict:
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

    # # --- Submission simulation ---
    # def _get_pending_batches(self) -> List[List[str]]:
    #     """Return list of pending chunk batches (exclude already submitted)."""
    #     if not self.all_output_files:
    #         logger.info("No chunk files found, running prepare_all().")
    #         self.prepare_all()

    #     pending = [p for p in self.all_output_files if p not in self.submitted_chunks]
    #     if not pending:
    #         return []
    #     return [pending[i : i + self.chunks_per_day] for i in range(0, len(pending), self.chunks_per_day)]

    # def get_next_batch(self) -> List[str]:
    #     """Return the next pending batch (full paths)."""
    #     batches = self._get_pending_batches()
    #     return batches[0] if batches else []

    # def get_status(self) -> Dict:
    #     """Return submission status: submitted count, remaining count, and next batch info."""
    #     total = len(self.all_output_files)
    #     submitted = len(self.submitted_chunks)
    #     remaining = total - submitted
    #     next_batch = [os.path.basename(p) for p in self.get_next_batch()]
    #     chunks_today = self._chunks_submitted_today()
    #     # estimate tokens today using estimated_tokens_per_chunk as a guide
    #     tokens_today_est = int(chunks_today * max(1, self.estimated_tokens_per_chunk))
    #     return {
    #         "total_chunks": total,
    #         "submitted_chunks": submitted,
    #         "remaining_chunks": remaining,
    #         "next_batch_files": next_batch,
    #         "chunks_submitted_today": chunks_today,
    #         "tokens_submitted_today_estimate": tokens_today_est,
    #         "tokens_allowed_per_day": self.max_tokens_per_day,
    #         "tokens_remaining_today_estimate": max(0, self.max_tokens_per_day - tokens_today_est),
    #     }

    # def _chunks_submitted_today(self) -> int:
    #     """Return number of chunks recorded in submission_log for today's date."""
    #     today_prefix = datetime.date.today().isoformat()
    #     return sum(1 for item in self.submission_log if str(item.get("timestamp", "")).startswith(today_prefix))

    # def _estimate_chunk_tokens(self, chunk_path: str) -> int:
    #     """Estimate tokens for a single chunk based on sampled lines and avg_tokens_per_line."""
    #     try:
    #         with open(chunk_path, "r", encoding="utf-8") as f:
    #             lines = sum(1 for _ in f)
    #     except OSError:
    #         return 0
    #     return int(self.avg_tokens_per_line * lines) if self.avg_tokens_per_line > 0 else 0

    # def _submit_chunk(self, chunk_path: str) -> Dict:
    #     """Submit a single chunk enforcing plan-based per-day limits.

    #     Uses estimated_tokens_per_chunk and chunks_per_day as the daily safeguards.
    #     """
    #     # use estimates rather than exact token counts
    #     est_tokens = max(1, int(self.estimated_tokens_per_chunk))

    #     chunks_today = self._chunks_submitted_today()
    #     if self.chunks_per_day > 0 and chunks_today >= self.chunks_per_day:
    #         msg = f"Daily chunk submission limit reached: {chunks_today} >= {self.chunks_per_day}"
    #         logger.error(msg)
    #         return {"chunk": str(chunk_path), "status": "rejected", "error": msg}

    #     tokens_today_est = chunks_today * est_tokens
    #     if tokens_today_est + est_tokens > self.max_tokens_per_day:
    #         msg = (
    #             f"Estimated submission would exceed daily token limit: today_est={tokens_today_est}, "
    #             f"chunk_est={est_tokens}, limit={self.max_tokens_per_day}"
    #         )
    #         logger.error(msg)
    #         return {"chunk": str(chunk_path), "status": "rejected", "error": msg}

    #     # perform submission
    #     try:
    #         oa.submit_batch(batch_file=str(chunk_path))
    #     except Exception as e:
    #         logger.error("Error submitting chunk %s: %s", chunk_path, e)
    #         return {"chunk": str(chunk_path), "status": "error", "error": str(e)}
    #     else:
    #         #oa.batch=type('A', (), {'id':str(uuid.uuid1())})()
    #         self.batch_id = oa.batch.id

    #     # success: record submission using estimated tokens
    #     self.submitted_chunks.append(chunk_path)
    #     entry = {"chunk": os.path.basename(chunk_path), "timestamp": datetime.datetime.utcnow().isoformat(),
    #              "tokens": est_tokens, "batch_id": self.batch_id}
    #     self.submission_log.append(entry)
    #     try:
    #         self._save_plan(self.output_dir / 'chunk_plan.yaml')
    #     except Exception as e:
    #         logger.warning("Failed to update saved plan after submission: %s", e)
    #     return {"chunk": str(chunk_path), "status": "submitted", "tokens_estimated": est_tokens, "batch_id": self.batch_id}

    # def run_submission_schedule(self, dry_run: bool = True) -> List[Dict]:
    #     """Run the submission schedule for up to `days` (default 1).

    #     By default this submits only the next day's batch so you can run once per day
    #     and resume later. Set days>1 to process more days in one call.
    #     """
    #     batch = self.get_next_batch()
    #     if not batch:
    #         logger.info("No pending chunks to submit. All %s chunks already submitted.", len(self.submitted_chunks))
    #         return []

    #     # get current day
    #     batch_num = 1 + self.daily_batches.index(batch)
    #     if self.submission_log:
    #         first = datetime.datetime.fromisoformat(self.submission_log[0]['timestamp'])
    #     else:
    #         first = datetime.datetime.today()
    #     today = datetime.datetime.today()
    #     days = 1 + (today.date() - first.date()).days

    #     batch_tokens = sum(self._estimate_chunk_tokens(chunk) for chunk in batch)
    #     results: List[Dict] = []
    #     logger.info("--- Day %s Batch %s: submitting %s chunks (~%s tokens) ---", days, batch_num, len(batch), batch_tokens)
    #     for chunk in batch:
    #         #batch_tokens = sum(self._estimate_chunk_tokens(c) for c in batch)

    #         logger.info("Submitting chunk: %s", os.path.basename(chunk))
    #         if dry_run:
    #             results.append({"chunk": str(chunk), "status": "dry-run"})
    #             continue

    #         res = self._submit_chunk(chunk)
    #         results.append(res)

    #     if results[0]['status'] != 'rejected':
    #         logger.info("Submission run complete. %s records generated.", len(results))

    #     return results

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

    @property
    def completed_batches(self) -> List[str]:
        """List of completed batch file paths."""
        return [str(i) for i in sorted(self.output_dir.glob("*chunk*.json"))]

    def _get_pending_batches(self) -> List[List[str]]:
        """Return list of pending chunk batches (exclude already submitted)."""
        if not self.all_output_files:
            logger.info("No chunk files found, running planner.prepare_all().")
            self.prepare_all()

        pending = [p for p in self.all_output_files if p not in self.submitted_chunks]
        if not pending:
            return []
        return [pending[i : i + self.chunks_per_day] for i in range(0, len(pending), self.chunks_per_day)]


# Submission and monitoring manager
class SubmissionManager:
    """Handles submitting chunk files (daily batches) and monitoring batch status.

    Works against a ChunkPlanner instance (planner) which holds chunk metadata and
    persistence helpers. The OpenAIHelper client is created per SubmissionManager
    instance to avoid import-time side effects.
    """

    def __init__(self, planner: ChunkPlanner, verbose: bool = None):
        self.planner = planner
        self.verbose = verbose or config.logging.verbose
        self.oa = OpenAIHelper(verbose=self.verbose)

        # expose some convenient attributes
        self.output_dir = self.planner.output_dir
        self.max_tokens_per_day = self.planner.max_tokens_per_day
        self.chunks_per_day = self.planner.chunks_per_day

    def get_next_batch(self) -> List[str]:
        """Return the next pending batch (full paths)."""
        batches = self._get_pending_batches()
        return batches[0] if batches else []

    def get_status(self) -> Dict:
        """Return submission status: submitted count, remaining count, and next batch info."""
        total = len(self.planner.all_output_files)
        submitted = len(self.planner.submitted_chunks)
        remaining = total - submitted
        next_batch = [os.path.basename(p) for p in self.get_next_batch()]
        chunks_today = self._chunks_submitted_today()
        tokens_today_est = int(chunks_today * max(1, self.planner.estimated_tokens_per_chunk))
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
        """Return number of chunks recorded in planner.submission_log for today's date."""
        today_prefix = datetime.date.today().isoformat()
        return sum(1 for item in self.planner.submission_log if str(item.get("timestamp", "")).startswith(today_prefix))

    def _estimate_chunk_tokens(self, chunk_path: str) -> int:
        """Estimate tokens for a single chunk based on sampled lines and planner.avg_tokens_per_line."""
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
        except OSError:
            return 0
        return int(self.planner.avg_tokens_per_line * lines) if self.planner.avg_tokens_per_line > 0 else 0

    def _submit_chunk(self, chunk_path: str) -> Dict:
        """Submit a single chunk enforcing plan-based per-day limits.

        Uses planner.estimated_tokens_per_chunk and planner.chunks_per_day as the daily safeguards.
        """
        # use estimates rather than exact token counts
        est_tokens = max(1, int(self.planner.estimated_tokens_per_chunk))

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

        # perform submission using OpenAIHelper
        try:
            self.oa.submit_batch(batch_file=str(chunk_path))
        except Exception as e:
            logger.error("Error submitting chunk %s: %s", chunk_path, e)
            return {"chunk": str(chunk_path), "status": "error", "error": str(e)}
        else:
            batch_id = getattr(self.oa.batch, "id", None)

        # success: record submission using estimated tokens
        self.planner.submitted_chunks.append(chunk_path)
        entry = {
            "chunk": os.path.basename(chunk_path),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tokens": est_tokens,
            "batch_id": batch_id,
        }
        self.planner.submission_log.append(entry)
        try:
            self.planner._save_plan(self.planner.output_dir / "chunk_plan.yaml")
        except Exception as e:
            logger.warning("Failed to update saved plan after submission: %s", e)
        return {"chunk": str(chunk_path), "status": "submitted", "tokens_estimated": est_tokens, "batch_id": batch_id}

    def run_submission_schedule(self, dry_run: bool = True, days: int = 1, pause_between_days: int = 0) -> List[Dict]:
        """Run the submission schedule for up to `days` (default 1).

        By default this submits only the next day's batch so you can run once per day
        and resume later. Set days>1 to process more days in one call.
        """
        batches = self._get_pending_batches()
        if not batches:
            logger.info(
                "No pending chunks to submit. All %s chunks already submitted.", len(self.planner.submitted_chunks)
            )
            return []

        batches = batches[: max(1, int(days))]

        results: List[Dict] = []
        for day_idx, batch in enumerate(batches, start=1):
            batch_tokens = sum(self._estimate_chunk_tokens(c) for c in batch)
            logger.info("--- Day %s: submitting %s chunks (~%s tokens) ---", day_idx, len(batch), batch_tokens)

            for chunk in batch:
                logger.info("Submitting chunk: %s", os.path.basename(chunk))
                if dry_run:
                    results.append({"chunk": str(chunk), "status": "dry-run"})
                    continue

                res = self._submit_chunk(chunk)
                results.append(res)

            if pause_between_days and day_idx < len(batches):
                logger.info("Pausing %s seconds before next day", pause_between_days)
                time.sleep(pause_between_days)

        logger.info("Submission run complete. %s records generated.", len(results))
        return results

    def check_batches_status(self) -> List[Dict]:
        """Check status of submitted batches using the OpenAI client.

        Returns a list of dicts: {"chunk": ..., "batch_id": ..., "status": ...}
        """
        try:
            remote_batches = {batch.id: batch.status for batch in self.oa.client.batches.list()}
        except Exception as e:
            logger.error("Error listing remote batches: %s", e)
            remote_batches = {}

        results = []
        for entry in self.planner.submission_log:
            bid = entry.get("batch_id")
            if not bid:
                continue
            status = remote_batches.get(bid, "not_found")
            results.append({"chunk": entry.get("chunk"), "batch_id": bid, "status": status})
        return results

    def retrieve_batch_results(self) -> List[Dict]:
        """Retrieve completed batches using OpenAIHelper.retrieve_batch (call left commented).

        For each logged batch_id this will attempt to retrieve the batch results. The actual
        retrieval call is commented out so you can modify oa.retrieve_batch before enabling.
        Returns a list of dicts with chunk, batch_id and a note about retrieval attempt.
        """
        results: List[Dict] = []
        for idx, entry in enumerate(self.planner.submission_log, start=1):
            bid = entry.get("batch_id")
            if not bid:
                continue

            output = (
                self.planner.output_dir / f"{config.llms.prompt_output_file.replace('.json', '')}_chunk_{idx:>03}.json"
            )
            try:
                # Uncomment to perform real retrievals once OA.retrieve_batch signature is confirmed
                # self.oa.retrieve_batch(batch_id=bid)
                results.append(
                    {"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": False, "note": "call commented out"}
                )
            except openai.APIStatusError as e:
                logger.error("Error retrieving batch %s: %s", bid, e)
                results.append({"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": False, "error": str(e)})
            else:
                results.append({"chunk": entry.get("chunk"), "batch_id": bid, "retrieved": True, "output": output})

        return results
