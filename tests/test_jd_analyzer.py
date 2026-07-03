from unittest.mock import MagicMock

from src.scoring.jd_analyzer import JDAnalyzer


def test_analyze_with_mocked_claude(mocker):
    fake_json = {
        "job_title": "Battery ML Engineer",
        "company": "VoltCell",
        "seniority": "junior",
        "required_skills": ["python", "pytorch"],
        "preferred_skills": ["pinn"],
        "responsibilities": ["train models"],
        "ats_keywords": ["python", "pytorch", "pinn"],
        "candidate_matched_skills": ["python"],
        "candidate_missing_skills": ["docker"],
        "strengths": ["strong projects"],
        "weaknesses": ["no industry experience"],
        "compatibility_score": 75,
        "ats_score": 70,
        "improvement_suggestions": ["learn docker"],
        "summary": "Good fit overall.",
    }

    client = MagicMock()
    client.complete_json.return_value = fake_json

    retriever = MagicMock()
    retriever.context_for.return_value = "Profile says: PyTorch, PINNs, MambaRUL."

    analyzer = JDAnalyzer(client=client, retriever=retriever)
    result = analyzer.analyze("We need PyTorch and PINNs.")
    assert result.job_title == "Battery ML Engineer"
    assert result.compatibility_score == 75
    assert "python" in result.candidate_matched_skills
    assert result.local_cosine_match >= 0.0
    client.complete_json.assert_called_once()
