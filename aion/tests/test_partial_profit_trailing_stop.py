import numpy as np

import aion.exec.paper_loop as pl


def test_partial_profit_targets_for_long_and_short():
    long_t = pl._partial_profit_target_price(100.0, 2.0, "LONG", 1.0)
    short_t = pl._partial_profit_target_price(100.0, 2.0, "SHORT", 1.0)
    assert abs(long_t - 102.0) < 1e-12
    assert abs(short_t - 98.0) < 1e-12


def test_trailing_stop_only_active_after_partial(monkeypatch):
    monkeypatch.setattr(pl.cfg, "TRAILING_STOP_ENABLED", True)

    pos = {"side": "LONG", "stop": 99.0, "trail_stop": 101.0, "partial_taken": False}
    stop_px, trailing_active = pl._effective_stop_price(pos)
    assert trailing_active is False
    assert abs(stop_px - 99.0) < 1e-12

    pos["partial_taken"] = True
    stop_px, trailing_active = pl._effective_stop_price(pos)
    assert trailing_active is True
    assert abs(stop_px - 101.0) < 1e-12


def test_trailing_stop_candidate_uses_atr_distance():
    long_trail = pl._trailing_stop_candidate("LONG", extreme_price=110.0, atr_value=2.0, atr_multiple=2.0)
    short_trail = pl._trailing_stop_candidate("SHORT", extreme_price=90.0, atr_value=2.0, atr_multiple=2.0)
    assert abs(long_trail - 106.0) < 1e-12
    assert abs(short_trail - 94.0) < 1e-12


def test_partial_fraction_rounding_is_configurable():
    assert pl._partial_close_qty(10, 0.50) == 5
    assert pl._partial_close_qty(3, 0.50) == 1
    assert pl._partial_close_qty(10, 0.34) == 3
    assert pl._partial_close_qty(1, 0.50) == 1


def test_disabled_trailing_mode_keeps_base_stop(monkeypatch):
    monkeypatch.setattr(pl.cfg, "TRAILING_STOP_ENABLED", False)
    pos = {"side": "SHORT", "stop": 101.5, "trail_stop": 99.5, "partial_taken": True}
    stop_px, trailing_active = pl._effective_stop_price(pos)
    assert trailing_active is False
    assert abs(stop_px - 101.5) < 1e-12


def test_asymmetric_entry_stop_multipliers_long_wider_than_short(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_LONG", 2.5)
    monkeypatch.setattr(pl.cfg, "STOP_ATR_SHORT", 2.0)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", False)

    long_mult, long_expanded = pl._entry_stop_atr_multiple("LONG", np.zeros(64))
    short_mult, short_expanded = pl._entry_stop_atr_multiple("SHORT", np.zeros(64))

    assert long_expanded is False
    assert short_expanded is False
    assert abs(long_mult - 2.5) < 1e-12
    assert abs(short_mult - 2.0) < 1e-12
    assert long_mult > short_mult


def test_vol_adaptive_stop_expansion_activates_in_high_vol(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_LONG", 2.5)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", True)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_LOOKBACK", 20)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_EXPANSION_MULT", 1.3)

    calm = np.tile(np.array([-0.001, 0.001], dtype=float), 140)
    shock = np.tile(np.array([-0.05, 0.05], dtype=float), 10)
    rets = np.concatenate([calm, shock])

    mult, expanded = pl._entry_stop_atr_multiple("LONG", rets)
    assert expanded is True
    assert abs(mult - (2.5 * 1.3)) < 1e-9


def test_vol_adaptive_disabled_uses_static_stop(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_SHORT", 2.0)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", False)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_EXPANSION_MULT", 1.8)
    rets = np.tile(np.array([-0.05, 0.05], dtype=float), 180)

    mult, expanded = pl._entry_stop_atr_multiple("SHORT", rets)
    assert expanded is False
    assert abs(mult - 2.0) < 1e-12
