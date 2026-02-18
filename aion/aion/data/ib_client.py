from typing import Optional

import time
import pandas as pd
from ib_insync import Contract, IB, Stock, util

from .. import config as cfg
from ..exec.order_state import load_order_state, merge_safe_req_id


_ib: Optional[IB] = None
_contract_cache: dict[str, Optional[Contract]] = {}
_hist_cache: dict[tuple[str, str, str], tuple[float, pd.DataFrame]] = {}
_last_hist_req_ts: float = 0.0
_resolved_port: Optional[int] = None
_resolved_host: Optional[str] = None
_resolved_client_id: Optional[int] = None


def _client_req_id(client) -> int:
    try:
        getter = getattr(client, "getReqId", None)
        if callable(getter):
            return int(getter())
    except Exception:
        pass
    return 0


def _set_client_req_id(client, req_id: int) -> None:
    rid = int(max(0, int(req_id)))
    try:
        setter = getattr(client, "setReqId", None)
        if callable(setter):
            setter(rid)
            return
    except Exception:
        pass
    try:
        updater = getattr(client, "updateReqId", None)
        if callable(updater):
            updater(rid)
            return
    except Exception:
        pass
    try:
        # Fallback for older clients where internal sequence is exposed.
        setattr(client, "_reqIdSeq", rid)
    except Exception:
        return


def _candidate_ports() -> list[int]:
    ports = [int(cfg.IB_PORT)]
    for p in getattr(cfg, "IB_PORT_CANDIDATES", []):
        try:
            ports.append(int(p))
        except Exception:
            continue
    uniq = []
    seen = set()
    for p in ports:
        if p <= 0 or p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def _candidate_hosts() -> list[str]:
    hosts = [str(cfg.IB_HOST)]
    for h in getattr(cfg, "IB_HOST_CANDIDATES", []):
        hs = str(h).strip()
        if hs:
            hosts.append(hs)
    uniq = []
    seen = set()
    for h in hosts:
        k = h.lower()
        if k in seen:
            continue
        uniq.append(h)
        seen.add(k)
    return uniq


def _candidate_client_ids() -> list[int]:
    ids = [int(cfg.IB_CLIENT_ID)]
    for cid in getattr(cfg, "IB_CLIENT_ID_CANDIDATES", []):
        try:
            ids.append(int(cid))
        except Exception:
            continue
    # Safety fallback offsets if caller did not configure candidates.
    base = int(cfg.IB_CLIENT_ID)
    ids.extend([base + 1, base + 2, base + 3, base + 4, base + 5])
    uniq = []
    seen = set()
    for cid in ids:
        if cid <= 0 or cid in seen:
            continue
        uniq.append(cid)
        seen.add(cid)
    return uniq


def _connect_ib_with_retries(client: IB):
    client_ids = _candidate_client_ids()
    hosts = _candidate_hosts()
    ports = _candidate_ports()

    last_exc = None
    for host in hosts:
        for port in ports:
            for i, client_id in enumerate(client_ids):
                timeout = int(8 + min(i, 6))
                try:
                    ok = client.connect(host, int(port), clientId=client_id, timeout=timeout)
                    if ok and client.isConnected():
                        return str(host), int(port), int(client_id)
                except Exception as exc:
                    last_exc = exc
                finally:
                    if not client.isConnected():
                        try:
                            client.disconnect()
                        except Exception:
                            pass
                time.sleep(0.35)

    if last_exc is not None:
        raise last_exc
    return None


def _pace_hist_requests():
    global _last_hist_req_ts
    min_interval = max(0.0, float(getattr(cfg, "IB_REQUEST_MIN_INTERVAL_SEC", 0.0)))
    if min_interval <= 0:
        return
    now = time.time()
    delta = now - _last_hist_req_ts
    if delta < min_interval:
        time.sleep(min_interval - delta)
    _last_hist_req_ts = time.time()


def _norm_symbol(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _symbol_candidates(symbol: str) -> list[str]:
    s = symbol.strip().upper()
    out: list[str] = []
    if "." in s:
        out.append(s.replace(".", " "))
        out.append(s.replace(".", "-"))
    out.append(s)

    uniq: list[str] = []
    seen = set()
    for item in out:
        if item and item not in seen:
            uniq.append(item)
            seen.add(item)
    return uniq


def ib() -> IB:
    global _ib, _resolved_port, _resolved_host, _resolved_client_id
    if _ib and _ib.isConnected():
        return _ib
    _ib = IB()
    connected = _connect_ib_with_retries(_ib)
    if connected is None:
        raise RuntimeError(f"Unable to connect to IBKR at {cfg.IB_HOST}:{cfg.IB_PORT}")
    connected_host, connected_port, connected_client_id = connected
    _resolved_host = str(connected_host)
    _resolved_port = int(connected_port)
    _resolved_client_id = int(connected_client_id)
    if str(cfg.IB_HOST) != _resolved_host:
        cfg.IB_HOST = _resolved_host
    if int(cfg.IB_PORT) != _resolved_port:
        cfg.IB_PORT = _resolved_port
    if int(cfg.IB_CLIENT_ID) != _resolved_client_id:
        cfg.IB_CLIENT_ID = _resolved_client_id
    saved = load_order_state(cfg.STATE_DIR)
    if isinstance(saved, dict):
        saved_id = int(saved.get("next_valid_id", 0))
        live_id = _client_req_id(_ib.client)
        safe_id = merge_safe_req_id(saved_id, live_id)
        if safe_id > 0:
            _set_client_req_id(_ib.client, safe_id)
    _ib.reqMarketDataType(cfg.IB_MARKET_DATA_TYPE)
    return _ib


def disconnect() -> None:
    global _ib, _last_hist_req_ts, _resolved_port, _resolved_host, _resolved_client_id
    if _ib and _ib.isConnected():
        _ib.disconnect()
    _ib = None
    _resolved_port = None
    _resolved_host = None
    _resolved_client_id = None
    _contract_cache.clear()
    _hist_cache.clear()
    _last_hist_req_ts = 0.0


def _split_exchanges(valid_exchanges: str):
    if not valid_exchanges:
        return []
    return [e.strip().upper() for e in valid_exchanges.split(",") if e.strip()]


def _qualify_stock(symbol: str) -> Optional[Contract]:
    key = symbol.strip().upper()
    if key in _contract_cache:
        return _contract_cache[key]

    client = ib()
    exact = []
    target_norm = _norm_symbol(key)
    for candidate in _symbol_candidates(key):
        try:
            details = client.reqContractDetails(Stock(candidate, "SMART", "USD"))
        except Exception:
            details = []
        if not details:
            continue

        exact = [
            d
            for d in details
            if getattr(d, "contract", None)
            and _norm_symbol(str(d.contract.symbol)) == target_norm
            and getattr(d.contract, "secType", "") == "STK"
            and getattr(d.contract, "currency", "") == "USD"
        ]
        if exact:
            break

    if not exact:
        _contract_cache[key] = None
        return None

    preferred = ("NASDAQ", "NYSE", "ARCA", "AMEX")

    def rank(d):
        c = d.contract
        pe = (getattr(c, "primaryExchange", "") or "").upper()
        vex = _split_exchanges(getattr(d, "validExchanges", ""))
        score = 0
        if any(e in pe for e in preferred):
            score += 100
        if any(e in vex for e in preferred):
            score += 60
        return score

    exact.sort(key=rank, reverse=True)
    best = exact[0]
    c0 = best.contract

    final = Contract()
    final.conId = c0.conId
    final.symbol = c0.symbol
    final.secType = "STK"
    final.exchange = "SMART"
    final.currency = "USD"

    try:
        qualified = client.qualifyContracts(final)
        resolved = qualified[0] if qualified else None
        _contract_cache[key] = resolved
        return resolved
    except Exception:
        _contract_cache[key] = None
        return None


def hist_bars(symbol: str, duration: str = None, barSize: str = None):
    client = ib()
    duration = duration or cfg.HIST_DURATION
    barSize = barSize or cfg.HIST_BAR_SIZE

    contract = _qualify_stock(symbol)
    if not contract:
        return pd.DataFrame()

    for what in ("TRADES", "MIDPOINT"):
        try:
            _pace_hist_requests()
            bars = client.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow=what,
                useRTH=cfg.HIST_USE_RTH,
                formatDate=1,
            )
            df = util.df(bars)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


def hist_bars_cached(symbol: str, duration: str = None, barSize: str = None, ttl_seconds: int = 0):
    duration = duration or cfg.HIST_DURATION
    barSize = barSize or cfg.HIST_BAR_SIZE
    key = (symbol.strip().upper(), duration, barSize)

    now = time.time()
    if ttl_seconds > 0 and key in _hist_cache:
        ts, cached = _hist_cache[key]
        if (now - ts) <= ttl_seconds:
            return cached.copy(deep=False)

    df = hist_bars(symbol=symbol, duration=duration, barSize=barSize)
    _hist_cache[key] = (now, df.copy(deep=False))
    return df
