"""Tests for app.evidence (snapshots, clip recorder, pruning)."""

import os
import time
from pathlib import Path

from app.evidence import ClipRecorder, prune_evidence, save_snapshot


def test_save_snapshot_writes_jpeg_and_returns_rel_path(settings_factory, frame):
    settings = settings_factory()
    rel = save_snapshot(frame, settings.evidence_dir, "camera", "HIGH")

    assert rel is not None
    assert rel.endswith(".jpg")
    # rel is "<YYYY-MM-DD>/<source>_<stamp>_<risk>.jpg" (posix, relative).
    assert Path(rel).name.startswith("camera_")
    assert Path(rel).name.endswith("_HIGH.jpg")
    # Relative posix path, not absolute.
    assert not Path(rel).is_absolute()

    absolute = settings.evidence_dir / rel
    assert absolute.exists()
    assert absolute.stat().st_size > 0


def test_save_snapshot_none_frame_returns_none(settings_factory):
    settings = settings_factory()
    assert save_snapshot(None, settings.evidence_dir, "camera", "HIGH") is None


def test_clip_recorder_arms_and_finishes(settings_factory, frame):
    settings = settings_factory(clip_pre_frames=2, clip_post_frames=2, clip_fps=10.0)
    rec = ClipRecorder(settings.evidence_dir, "camera",
                       pre_frames=2, post_frames=2, fps=10.0)

    # Feed the pre-buffer.
    rec.push(frame)
    rec.push(frame)
    assert rec.recording is False

    # Arm the clip.
    assert rec.trigger() is True
    assert rec.recording is True

    # Write the post-roll frames; the recorder finalises once post_frames hit 0.
    rec.push(frame)
    rec.push(frame)

    finished = rec.take_finished()
    assert len(finished) == 1
    clip_rel = finished[0]
    assert clip_rel.endswith("_clip.mp4")
    assert (settings.evidence_dir / clip_rel).exists()
    assert rec.recording is False

    # take_finished() drains the queue.
    assert rec.take_finished() == []


def test_clip_trigger_without_prebuffer_returns_false(settings_factory):
    settings = settings_factory()
    rec = ClipRecorder(settings.evidence_dir, "camera", pre_frames=2, post_frames=2)
    # Nothing pushed yet -> no dimensions -> cannot arm.
    assert rec.trigger() is False
    assert rec.recording is False


def test_clip_zero_post_frames_finalises_on_trigger(settings_factory, frame):
    settings = settings_factory()
    rec = ClipRecorder(settings.evidence_dir, "video", pre_frames=1, post_frames=0)
    rec.push(frame)
    assert rec.trigger() is True
    # post_frames == 0 means it finalises immediately inside trigger().
    assert rec.recording is False
    finished = rec.take_finished()
    assert len(finished) == 1


def test_close_finalises_in_progress_clip(settings_factory, frame):
    settings = settings_factory()
    rec = ClipRecorder(settings.evidence_dir, "camera", pre_frames=1, post_frames=100)
    rec.push(frame)
    rec.trigger()
    assert rec.recording is True
    rec.close()
    assert rec.recording is False
    assert len(rec.take_finished()) == 1


def test_prune_evidence_keeps_newest(settings_factory):
    settings = settings_factory()
    folder = settings.evidence_dir / "2026-01-01"
    folder.mkdir(parents=True, exist_ok=True)

    now = time.time()
    for i in range(6):
        p = folder / f"img_{i}.jpg"
        p.write_bytes(b"x")
        # Stagger mtimes so ordering is deterministic (higher index = newer).
        mtime = now - (100 - i)
        os.utime(p, (mtime, mtime))

    prune_evidence(settings.evidence_dir, keep=2)

    remaining = sorted(p.name for p in folder.glob("*.jpg"))
    assert len(remaining) == 2
    # The two newest (highest index) survive.
    assert remaining == ["img_4.jpg", "img_5.jpg"]


def test_prune_evidence_ignores_other_extensions(settings_factory):
    settings = settings_factory()
    folder = settings.evidence_dir / "2026-01-01"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notes.txt").write_text("keep me")
    (folder / "img_0.jpg").write_bytes(b"x")

    prune_evidence(settings.evidence_dir, keep=0)

    # .txt is never touched; the .jpg beyond keep=0 is removed.
    assert (folder / "notes.txt").exists()
    assert not (folder / "img_0.jpg").exists()
