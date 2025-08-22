

import pytest

from bibcat.llm.chunker import ChunkPlanner, SubmissionManager


@pytest.fixture()
def onebatch(batchfile):
    """fixture for creating a single batch file"""
    yield batchfile(50)


@pytest.fixture()
def twobatch(batchfile):
    """fixture for creating a single batch file"""
    yield batchfile(100)


@pytest.fixture()
def planner(twobatch):
    """fixture for creating a basic plan"""
    yield ChunkPlanner(str(twobatch), max_lines=75)

@pytest.fixture()
def midplan(planner):
    """fixture for creating a plan mid processing"""
    planner.analyze(sample_lines=5)
    planner.plan_chunks()
    planner.create_chunks()
    yield planner


@pytest.fixture()
def fullplan(midplan):
    """fixture for creating a plan full processing"""
    midplan.create_daily_batches()
    midplan.get_submission_schedule()
    yield midplan


def test_no_chunks_needed(onebatch):
    """test no chunking is needed"""
    planner = ChunkPlanner(str(onebatch), max_lines=75)
    res = planner.analyze(sample_lines=5)
    print(res)
    assert res['total_lines'] == 50

    res = planner.plan_chunks()
    assert res['lines_per_chunk'] == 50
    assert res['base_chunks_needed'] == 1
    assert res['days_needed'] == 1
    assert res['chunks_per_day'] == 1


def test_analyze_and_plan(planner):
    """test analyze method"""
    res = planner.analyze(sample_lines=5)
    print(res)
    assert res["total_lines"] == 100

    res = planner.plan_chunks()
    print(res)
    assert res['lines_per_chunk'] == 50
    assert res['base_chunks_needed'] == 2
    assert res['days_needed'] == 1
    assert res['chunks_per_day'] == 2


def test_create_chunks(planner):
    """test we can create chunked files"""
    planner.analyze(sample_lines=5)
    planner.plan_chunks()
    chunks = planner.create_chunks()
    assert len(chunks) == 2
    assert chunks[0].exists()
    assert chunks[1].name == 'batch_chunk_002.jsonl'

    for cc in chunks:
        with open(cc, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
            assert len(lines) == 50

    res = planner.verify_chunks()
    assert res['total_actual_tokens'] > 10
    assert planner.plan_path.exists()


def test_daily_batches(midplan):
    """test we can create daily batches"""
    res = midplan.create_daily_batches()
    print(res)
    assert len(res) == 1
    assert len(res[0]) == 2
    assert res[0][0].name == 'batch_chunk_001.jsonl'


def test_submission_schedule(midplan):
    """test we can return the submission plan"""
    midplan.create_daily_batches()
    res = midplan.get_submission_schedule()
    assert res["total_chunks"] == 2
    assert res["days_needed"] == 1
    assert res["chunks_per_day"] == 2
    assert len(res["daily_batches"][0]) == 2

def test_prepare_all(planner):
    """test that prepare_all runs through everything"""
    res = planner.prepare_all(sample_lines=5)
    assert planner.total_lines == 100
    assert planner.lines_per_chunk == 50
    assert planner.base_chunks_needed == 2
    assert res["total_chunks"] == 2
    assert res["days_needed"] == 1
    assert res["chunks_per_day"] == 2
    assert len(planner.daily_batches[0]) == 2

def test_plan_info(fullplan):
    """test the info returns ok"""
    res = fullplan.plan_info()
    assert "batch.jsonl" in res["file"]
    assert res["total_lines"] == 100
    assert res["base_chunks_needed"] == 2
    assert res["lines_per_chunk"] == 50
    assert res["chunks_per_day"] == 2
    assert res["days_needed"] == 1
    assert res["total_chunks"] == 2
    assert res["n_daily_batches"] == 1
    assert res["n_submitted_batches"] == 0
    assert res["n_remaining_batches"] == 1
    assert res["n_completed_batches"] == 0


def test_load_plan(fullplan):
    """test we can load a plan yaml file"""
    planner = ChunkPlanner(str(fullplan.input_file_path))
    files = [str(i) for i in fullplan.all_output_files]
    assert files == planner.all_output_files
    assert planner.base_chunks_needed == 2
    assert planner.all_output_files == planner.get_pending_batches()[0]


def test_submission_manager(fullplan, twobatch):
    """test we can create the submission manager"""
    sm = SubmissionManager(fullplan)
    assert sm.chunks_per_day == 2
    assert "/batch_chunks" in str(sm.planner.output_dir)
    othersm = SubmissionManager(file=twobatch)
    assert othersm.chunks_per_day == 2


@pytest.fixture()
def sm(fullplan):
    """fixture for creating a submission manager"""
    yield SubmissionManager(fullplan)


def test_get_next_batch(sm):
    batch = sm.get_next_batch()
    assert batch[0].name == "batch_chunk_001.jsonl"


def test_get_status(sm):
    status = sm.get_status()
    assert status["total_chunks"] == 2
    assert status["submitted_chunks"] == 0
    assert status["remaining_chunks"] == 2
    assert status["next_batch_files"] == ["batch_chunk_001.jsonl", "batch_chunk_002.jsonl"]
    assert status["chunks_submitted_today"] == 0
    assert status["tokens_allowed_per_day"] == 40000000
    assert status["tokens_remaining_today_estimate"] == 40000000
