"""brand_style.classify_vertical maps news text to its vertical anchor."""
from brand_style import classify_vertical, anchor_for


def test_politica_headline_classifies_politica():
    assert classify_vertical("AMLO anuncia ajuste presupuestario para Morena") == "politica"


def test_chismes_classifies_chismes():
    assert classify_vertical("Ruptura entre famoso cantante y su novia se vuelve viral") == "chismes"


def test_clima_classifies_clima():
    assert classify_vertical("Tormenta tropical Beatriz se acerca a Quintana Roo") == "clima"


def test_deportes_classifies_deportes():
    assert classify_vertical("La selección mexicana gana en el estadio Azteca") == "deportes"


def test_no_match_classifies_default():
    assert classify_vertical("Algún texto random sin keywords") == "default"


def test_political_news_picks_polibruh():
    a = anchor_for("AMLO en conferencia")
    assert a.id == "don_polibruh"


def test_chisme_news_picks_dona_chispas():
    a = anchor_for("Ruptura viral en redes sociales")
    assert a.id == "dona_chispas"


def test_unknown_news_picks_compa_caribe_as_default():
    a = anchor_for("texto sin pistas")
    assert a.id == "compa_caribe"   # the catch-all has 'default' in its verticals
