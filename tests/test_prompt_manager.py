from src.llm.prompt_manager import PromptManager


def test_render_jd_analyzer():
    pm = PromptManager()
    out = pm.render("jd_analyzer", context="my profile", jd="some JD")
    assert "my profile" in out
    assert "some JD" in out


def test_missing_var_becomes_empty():
    pm = PromptManager()
    out = pm.render("career_qa", context="ctx")  # 'question' missing
    assert "ctx" in out
