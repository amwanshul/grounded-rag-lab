from app.pipeline import build_demo_pipeline


def test_pipeline_returns_citations():
    result = build_demo_pipeline().answer("what is reciprocal rank fusion?")
    assert result["status"] == "grounded_context"
    assert result["citations"]
    assert all(item["citation"].startswith("[") for item in result["evidence"])


def test_pipeline_handles_unknown_query():
    result = build_demo_pipeline().answer("quantum banana protocol")
    assert result["status"] in {"grounded_context", "abstained"}
