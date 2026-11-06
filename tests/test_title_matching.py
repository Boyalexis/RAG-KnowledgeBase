
from app.rag.evidence_selector import title_match_score


def test_cola_chicken():
    query = "打算用一瓶可乐烧鸡翅"

    correct = title_match_score(
        "可乐鸡翅", query
    )

    wrong = title_match_score(
        "红烧鸡翅", query
    )

    assert correct > wrong


def test_scallion_noodles():
    query = "想炸一点葱香酱油汁拌面"

    correct = title_match_score(
        "葱油拌面", query
    )

    wrong = title_match_score(
        "蒜香酱油", query
    )

    assert correct > wrong


def test_exact_title():
    assert title_match_score(
        "西红柿炒鸡蛋",
        "西红柿炒鸡蛋怎么做？"
    ) == 1.0


def test_empty_title():
    assert title_match_score(
        "", "怎么做菜？"
    ) == 0.0


def test_avoid_sweet_sour_sauce_false_match():
    query = (
        "想把黄瓜拍碎，拌成酸酸甜甜的开胃小菜，"
        "糖是和酱油醋一起放，还是要先腌？"
        "最后再放什么油？"
    )

    assert title_match_score("糖醋汁", query) == 0.0


def test_avoid_steamed_pork_false_match():
    query = (
        "我想让红薯粉丝吸足带肉末和豆瓣酱的汤汁，"
        "干粉丝要先泡多久？"
        "下锅后怎么判断差不多了？"
    )

    assert title_match_score("粉蒸肉", query) == 0.0
