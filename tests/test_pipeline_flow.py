"""End-to-end pipeline: layers built, lineage recorded, idempotent, contract enforced."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_engine.config import DataConfig, PipelineConfig
from quant_engine.ml.features import FEATURE_NAMES
from quant_engine.pipeline.contracts import DataContractError
from quant_engine.pipeline.curate import load_curated
from quant_engine.pipeline.flow import run_pipeline


def _data() -> DataConfig:
    return DataConfig(source="synthetic", symbols=["AAA", "BBB", "CCC"], bars=300, seed=11)


def test_run_pipeline_builds_all_layers(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    result = run_pipeline(_data(), PipelineConfig(root=str(lake)))

    assert (lake / "raw").exists()
    assert (lake / "validated").exists()
    assert (lake / "curated").exists()
    assert (lake / "_manifest.json").exists()

    # Clean synthetic data => nothing quarantined.
    assert result.validation.n_quarantined == 0
    assert result.validation.quarantine_rate == 0.0

    curated = load_curated(lake / "curated")
    assert set(curated) == {"AAA", "BBB", "CCC"}
    for frame in curated.values():
        assert set(FEATURE_NAMES).issubset(frame.columns)
        assert "close" in frame.columns
        assert not frame.isna().any().any()  # warm-up rows were dropped


def test_manifest_records_lineage(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    result = run_pipeline(_data(), PipelineConfig(root=str(lake)))
    manifest = json.loads(Path(result.manifest_path).read_text())

    assert manifest["source"] == "synthetic"
    assert set(manifest["watermark"]) == {"AAA", "BBB", "CCC"}
    assert manifest["validation"]["n_valid"] == result.validation.n_valid


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    first = run_pipeline(_data(), PipelineConfig(root=str(lake)))
    second = run_pipeline(_data(), PipelineConfig(root=str(lake)))
    # delete-matching partition writes => re-running never duplicates rows.
    assert second.validation.n_valid == first.validation.n_valid
    assert second.curate.total_rows == first.curate.total_rows


def test_contract_gate_blocks_dirty_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_engine.pipeline.ingest as ingest_mod

    resolve = ingest_mod._resolve_frames

    def dirty(config: DataConfig) -> dict:
        frames = resolve(config)
        for frame in frames.values():
            frame.iloc[:, frame.columns.get_loc("high")] = -1.0  # break every row
        return frames

    monkeypatch.setattr(ingest_mod, "_resolve_frames", dirty)

    lake = tmp_path / "lake"
    with pytest.raises(DataContractError):
        run_pipeline(_data(), PipelineConfig(root=str(lake), max_quarantine_rate=0.02))
