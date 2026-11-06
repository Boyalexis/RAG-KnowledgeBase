"""无需联网的轻量测试（进入项目环境后运行）。"""
from app.rag.rag_pipeline import _section


def test_section_from_selected_evidence():
    assert _section({"section": "操作", "text": "章节：计算\n内容"}) == "操作"


def test_section_from_text():
    assert _section({"text": "菜名：皮蛋豆腐\n章节：必备原料和工具\n- 辣椒油（可选）"}) == "必备原料和工具"
