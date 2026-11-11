#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAC 40 Swing — V2 : rebonds après excès de baisse (achats) / replis après excès de hausse en
tendance baissière (ventes), évalués séparément et sans complaisance.

Ce qui change par rapport à la V1 (conclusions des données 1990-2026) :
  - les niveaux clés (anciens points hauts/bas) n'amélioraient pas la détection des retournements :
    ils sont affichés à titre d'information mais ne conditionnent plus le signal ;
  - le RSI en excès améliorait la détection des creux : il devient un paramètre central ;
  - les ventes étaient perdantes : elles ne sont testées qu'en tendance baissière (cours < SMA 200)
    et chaque sens a son propre walk-forward, son propre contrôle par le hasard et son propre statut ;
  - un coût par trade (en points d'indice) est intégré au choix des paramètres ;
  - les règles sont GELÉES au premier lancement de la V2 (data/frozen_v2.json) : les signaux
    envoyés viennent de ces règles figées, et les trades postérieurs au gel forment un suivi
    "à blanc" qui sera la seule vraie preuve (données jamais vues lors de la conception).

Règle d'activation (fixée à l'avance) : un sens n'émet de signal que si, hors échantillon et
frais inclus, son espérance est > 0 ET au plus 10 % des tirages aléatoires font aussi bien.

Outil d'aide à la décision, pas un conseil en investissement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import random
import smtplib
import ssl
import sys
from collections import deque
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

VERSION = 2

CFG = {
    "ticker": "^FCHI",
    "atr_n": 14,
    "rsi_n": 14,
    "sma_n": 200,
    "fractal_k": 5,          # niveaux clés (information) : extrême sur k séances de chaque côté
    "level_lookback": 500,   # ancienneté max d'un niveau clé (séances, ~2 ans)
    "ext_lookback": 15,      # fenêtre de mesure de l'excès (séances)
    "touch_window": 3,       # l'extrême / le RSI sont mesurés sur les 3 dernières séances
    "trigger_atr": 0.10,     # déclenchement = extrême de la bougie signal +/- 0,1 ATR
    "order_valid": 3,        # validité de l'ordre stop (séances)
    "stop_lookback": 5,      # stop au-delà de l'extrême des 5 dernières séances
    "max_bars": 40,          # sortie forcée (séances)
    "trail_atr": 3.0,        # stop suiveur (mode "trail")
    "train_years": 8,
    "test_years": 2,
    "min_train_trades": 15,
    "random_runs": 300,
    "zigzag_thresholds": [3.0, 5.0, 8.0],
    # Coût aller-retour supposé, en points d'indice (spread + frais + financement d'un turbo).
    # HYPOTHÈSE à remplacer par votre coût réel : il est déduit de chaque trade (R - coût / risque).
    "COST_POINTS": 3.0,
    "cost_sensitivity": [0.0, 3.0, 6.0, 10.0],
    "activation_max_random_share": 0.10,
    "frozen_file": "data/frozen_v2.json",
}

GRID = {
    "long": {
        "ext": [2.5, 3.5, 4.5],
        "rsi": [30, 35, 40, None],     # RSI(14) min des 3 dernières séances <= seuil (None = pas de filtre)
        "confirm": ["rejet", "aucun"],
        "trend": ["aucun", "avec"],    # "avec" = achat seulement au-dessus de la SMA 200
        "stopbuf": [0.25, 0.75],
        "exit": ["2R", "3R", "trail"],
    },
    "short": {
        "ext": [2.5, 3.5, 4.5],
        "rsi": [70, 65, 60, None],     # RSI(14) max des 3 dernières séances >= seuil
        "confirm": ["rejet", "aucun"],
        "trend": ["avec"],             # vente seulement sous la SMA 200 (imposé)
        "stopbuf": [0.25, 0.75],
        "exit": ["2R", "3R", "trail"],
    },
}


def combos(side):
    g = GRID[side]
    keys = list(g)
    for vals in itertools.product(*[g[k] for k in keys]):
        p = dict(zip(keys, vals))
        p["side"] = side
        yield p


def combo_label(p):
    up = p["side"] == "long"
    rsi = "non" if p["rsi"] is None else (f"≤ {p['rsi']}" if up else f"≥ {p['rsi']}")
    tr = "non" if p["trend"] == "aucun" else ("cours > SMA 200" if up else "cours < SMA 200")
    ex = "stop suiveur" if p["exit"] == "trail" else "objectif " + p["exit"]
    return (f"{'Achat' if up else 'Vente'} | excès ≥ {p['ext']} ATR | RSI {rsi} | rejet : {'oui' if p['confirm']=='rejet' else 'non'} | "
            f"tendance : {tr} | stop +{p['stopbuf']} ATR | {ex}")


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
def load_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf  # importé ici pour pouvoir tester sans yfinance
    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"Aucune donnée reçue de Yahoo pour {ticker}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def drop_unfinished_bar(df: pd.DataFrame):
    """Retire la séance du jour si la bourse de Paris n'est pas encore clôturée
    (Euronext Paris : clôture 17:30, marge jusqu'à 17:45 heure de Paris).
    Sans cela, un lancement en journée analyserait une bougie incomplète."""
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("Europe/Paris"))
    if len(df) and df.index[-1].date() == now.date() and now.time() < dt.time(17, 45):
        return df.iloc[:-1], True
    return df, False


def load_csv(path: str) -> pd.DataFrame:
    """CSV avec colonnes Date, Open, High, Low, Close (séparateur , ou ;)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in df.columns if c in ("date", "datetime", "time"))
    df.index = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce").dt.normalize()
    df = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0].copy()
    prev_close = df["close"].shift(1)
    # ouvertures manquantes / nulles (fréquent sur les vieilles séries d'indices)
    bad_open = df["open"].isna() | (df["open"] <= 0)
    df.loc[bad_open, "open"] = prev_close[bad_open].fillna(df["close"][bad_open])
    for c in ("high", "low"):
        bad = df[c].isna() | (df[c] <= 0)
        df.loc[bad, c] = df.loc[bad, "close"]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


# --------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Indicateurs et niveaux clés (sans regard vers le futur)
# --------------------------------------------------------------------------
def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = wilder(tr, CFG["atr_n"])
    d = c.diff()
    up = wilder(d.clip(lower=0), CFG["rsi_n"])
    dn = wilder((-d).clip(lower=0), CFG["rsi_n"])
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["sma200"] = c.rolling(CFG["sma_n"]).mean()
    return df


# --------------------------------------------------------------------------
# Points hauts / bas fractals et niveaux clés (sans regard vers le futur)
# --------------------------------------------------------------------------
def fractal_points(h: np.ndarray, l: np.ndarray, k: int):
    """Renvoie la liste (i_point, i_confirmation, prix, 'H'|'L')."""
    n = len(h)
    pts = []
    for i in range(k, n - k):
        wh = h[i - k:i + k + 1]
        wl = l[i - k:i + k + 1]
        if h[i] == wh.max() and np.argmax(wh) == k:
            pts.append((i, i + k, float(h[i]), "H"))
        if l[i] == wl.min() and np.argmin(wl) == k:
            pts.append((i, i + k, float(l[i]), "L"))
    pts.sort(key=lambda x: x[1])
    return pts


def level_distances(df: pd.DataFrame, pts):
    """Pour chaque séance t : distance (en ATR) entre l'extrême des 3 dernières séances
    et le niveau clé connu le plus proche, + nombre de points proches (force)."""
    n = len(df)
    h, l, atr = df["high"].values, df["low"].values, df["atr"].values
    tw, lb = CFG["touch_window"], CFG["level_lookback"]
    d_long = np.full(n, np.inf)
    d_short = np.full(n, np.inf)
    lvl_long = np.full(n, np.nan)
    lvl_short = np.full(n, np.nan)
    tols = [0.5, 1.0]  # tolérances utilisées pour la recherche descriptive
    cnt_long = {tl: np.zeros(n, dtype=int) for tl in tols}
    cnt_short = {tl: np.zeros(n, dtype=int) for tl in tols}
    p_ptr = 0
    known = deque()  # (i, prix), par ordre de confirmation
    for t in range(n):
        while p_ptr < len(pts) and pts[p_ptr][1] <= t:
            known.append((pts[p_ptr][0], pts[p_ptr][2]))
            p_ptr += 1
        while known and known[0][0] < t - lb:
            known.popleft()
        if t < tw or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        lo3 = l[t - tw + 1:t + 1].min()
        hi3 = h[t - tw + 1:t + 1].max()
        best_l, best_s = np.inf, np.inf
        for i, p in known:
            if i < t - lb or i > t - tw:  # trop vieux, ou fait partie de la zone testée
                continue
            dl = abs(lo3 - p)
            ds = abs(hi3 - p)
            for tl in tols:
                if dl <= tl * atr[t]:
                    cnt_long[tl][t] += 1
                if ds <= tl * atr[t]:
                    cnt_short[tl][t] += 1
            if dl < best_l:
                best_l, lvl_long[t] = dl, p
            if ds < best_s:
                best_s, lvl_short[t] = ds, p
        d_long[t] = best_l / atr[t]
        d_short[t] = best_s / atr[t]
    return d_long, d_short, lvl_long, lvl_short, cnt_long, cnt_short


def level_zones(df: pd.DataFrame, pts, t: int, tol_atr: float = 0.5):
    """Zones de niveaux clés connues à la séance t (regroupement des points proches)."""
    atr = float(df["atr"].values[t])
    lb = CFG["level_lookback"]
    dates = df.index
    cand = [(p, i) for (i, ic, p, _) in pts if ic <= t and i >= t - lb]
    cand.sort()
    zones = []
    for p, i in cand:
        if zones and p - zones[-1]["prices"][0] <= tol_atr * atr:
            zones[-1]["prices"].append(p)
            zones[-1]["idx"].append(i)
        else:
            zones.append({"prices": [p], "idx": [i]})
    out = []
    for z in zones:
        out.append({
            "price": round(float(np.mean(z["prices"])), 2),
            "low": round(min(z["prices"]), 2),
            "high": round(max(z["prices"]), 2),
            "touches": len(z["prices"]),
            "last": dates[max(z["idx"])].strftime("%Y-%m-%d"),
        })
    return out


# --------------------------------------------------------------------------
# Signaux
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Variables de décision
# --------------------------------------------------------------------------
class Features:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.o = df["open"].values
        self.h = df["high"].values
        self.l = df["low"].values
        self.c = df["close"].values
        self.atr = df["atr"].values
        self.rsi = df["rsi"].values
        self.sma = df["sma200"].values
        n = len(df)
        L, tw = CFG["ext_lookback"], CFG["touch_window"]
        hh = df["high"].rolling(L).max().values
        ll = df["low"].rolling(L).min().values
        lo3 = df["low"].rolling(tw).min().values
        hi3 = df["high"].rolling(tw).max().values
        self.ext_long = (hh - lo3) / self.atr      # baisse parcourue (en ATR) avant un rebond
        self.ext_short = (hi3 - ll) / self.atr     # hausse parcourue (en ATR) avant un repli
        rng = self.h - self.l
        pos = np.where(rng > 0, (self.c - self.l) / np.where(rng > 0, rng, 1), 0.5)
        pc = np.r_[np.nan, self.c[:-1]]
        self.rej_long = (pos >= 0.5) & (self.c > pc)
        self.rej_short = (pos <= 0.5) & (self.c < pc)
        self.rsi_min3 = df["rsi"].rolling(tw).min().values
        self.rsi_max3 = df["rsi"].rolling(tw).max().values
        self.up_trend = self.c > self.sma
        self.dn_trend = self.c < self.sma
        self.pts = fractal_points(self.h, self.l, CFG["fractal_k"])
        (self.d_long, self.d_short, self.lvl_long, self.lvl_short,
         self.cnt_long, self.cnt_short) = level_distances(df, self.pts)
        self.valid = np.isfinite(self.atr) & np.isfinite(self.sma) & (np.arange(n) >= CFG["sma_n"])
        sl = CFG["stop_lookback"]
        self.low_n = df["low"].rolling(sl).min().values
        self.high_n = df["high"].rolling(sl).max().values
        self._zero = np.zeros(n, dtype=bool)

    def _rsi_ok(self, side, thr):
        if thr is None:
            return np.ones(len(self.c), dtype=bool)
        return (self.rsi_min3 <= thr) if side == "long" else (self.rsi_max3 >= thr)

    def signals(self, p):
        side = p["side"]
        ext = self.ext_long if side == "long" else self.ext_short
        m = self.valid & (ext >= p["ext"]) & self._rsi_ok(side, p["rsi"])
        if p["confirm"] == "rejet":
            m &= self.rej_long if side == "long" else self.rej_short
        if p["trend"] == "avec":
            m &= self.up_trend if side == "long" else self.dn_trend
        return (m, self._zero) if side == "long" else (self._zero, m)

    def conditions(self, p, t):
        side = p["side"]
        up = side == "long"
        ext = (self.ext_long if up else self.ext_short)[t]
        r3 = (self.rsi_min3 if up else self.rsi_max3)[t]
        rej = bool((self.rej_long if up else self.rej_short)[t])
        trd = bool((self.up_trend if up else self.dn_trend)[t])
        cond = {
            "exces": {"ok": bool(ext >= p["ext"]), "valeur": round(float(ext), 2), "seuil": p["ext"]},
            "rsi": {"ok": True if p["rsi"] is None else bool(r3 <= p["rsi"] if up else r3 >= p["rsi"]),
                    "actif": p["rsi"] is not None, "valeur": round(float(r3), 1), "seuil": p["rsi"]},
            "rejet": {"ok": rej if p["confirm"] == "rejet" else True, "actif": p["confirm"] == "rejet", "constate": rej},
            "tendance": {"ok": trd if p["trend"] == "avec" else True, "actif": p["trend"] == "avec",
                         "constate": trd, "sma200": round(float(self.sma[t]), 2)},
        }
        cond["signal"] = bool(self.valid[t]) and all(v["ok"] for v in cond.values())
        lvl = (self.lvl_long if up else self.lvl_short)[t]
        dist = (self.d_long if up else self.d_short)[t]
        cond["niveau_info"] = {"niveau": None if not np.isfinite(lvl) else round(float(lvl), 2),
                               "dist_atr": None if not np.isfinite(dist) else round(float(dist), 2)}
        return cond


# --------------------------------------------------------------------------
# Simulation des trades (barres journalières, hypothèses prudentes)
# --------------------------------------------------------------------------
def manage_position(F: Features, side, d, entry, stop, target, mode, risk, entry_by_open):
    """Gère une position ouverte à la séance d (jour d'entrée).
    Renvoie (i_sortie, prix, motif) ou None si la position est toujours ouverte.
    Hypothèses prudentes : si stop et objectif sont tous deux atteignables dans la même
    séance, on considère que le stop a été touché en premier ; un gap au-delà du stop
    est exécuté à l'ouverture."""
    o, h, l, c, atr = F.o, F.h, F.l, F.c, F.atr
    n = len(c)
    sgn = 1 if side == "long" else -1
    stop0 = stop
    best_close = c[d]
    for j in range(d, n):
        if j > d:
            if sgn * (o[j] - stop) <= 0:
                return j, o[j], "stop (gap)" if stop == stop0 else "stop suiveur (gap)"
            if target is not None and sgn * (o[j] - target) >= 0:
                return j, o[j], "objectif (gap)"
        if (sgn == 1 and l[j] <= stop) or (sgn == -1 and h[j] >= stop):
            return j, stop, "stop" if stop == stop0 else "stop suiveur"
        if target is not None and ((sgn == 1 and h[j] >= target) or (sgn == -1 and l[j] <= target)):
            return j, target, "objectif"
        if j - d + 1 >= CFG["max_bars"]:
            return j, c[j], "durée max"
        if mode == "trail":
            best_close = max(best_close, c[j]) if sgn == 1 else min(best_close, c[j])
            if sgn * (best_close - entry) >= risk:
                stop = max(stop, entry) if sgn == 1 else min(stop, entry)
            trail = best_close - sgn * CFG["trail_atr"] * atr[j]
            stop = max(stop, trail) if sgn == 1 else min(stop, trail)
    return None


def simulate(F: Features, p, long_sig=None, short_sig=None, keep_open=False):
    if long_sig is None:
        long_sig, short_sig = F.signals(p)
    o, h, l, c, atr = F.o, F.h, F.l, F.c, F.atr
    n = len(c)
    trades = []
    open_pos = None
    pending = None
    t = 0
    while t < n:
        # 1) ordre en attente : déclenchement ?
        if pending is not None and t > pending["t"]:
            side, trig, stop = pending["side"], pending["trigger"], pending["stop"]
            sgn = 1 if side == "long" else -1
            if t - pending["t"] > CFG["order_valid"]:
                pending = None
            elif sgn * (o[t] - stop) <= 0 or (sgn == 1 and l[t] <= stop and h[t] < trig) or (sgn == -1 and h[t] >= stop and l[t] > trig):
                pending = None  # setup invalidé avant déclenchement
            else:
                entry, by_open = None, False
                if sgn * (o[t] - trig) >= 0:
                    entry, by_open = o[t], True
                elif (sgn == 1 and h[t] >= trig) or (sgn == -1 and l[t] <= trig):
                    entry = trig
                if entry is not None:
                    risk = sgn * (entry - stop)
                    if risk > 0:
                        target = None
                        if p["exit"] in ("2R", "3R"):
                            target = entry + sgn * float(p["exit"][0]) * risk
                        res = manage_position(F, side, t, entry, stop, target, p["exit"], risk, by_open)
                        tr = {
                            "side": side, "sig": pending["t"], "in": t, "entry": float(entry),
                            "stop": float(stop), "target": None if target is None else float(target),
                            "risk": float(risk),
                        }
                        pending = None
                        if res is None:
                            if keep_open:
                                open_pos = tr
                            break
                        j, px, why = res
                        r = sgn * (px - entry) / risk - CFG["COST_POINTS"] / risk
                        tr.update({"out": j, "exit": float(px), "why": why, "R": float(r), "bars": j - t + 1})
                        trades.append(tr)
                        t = j + 1
                        continue
                    pending = None
        # 2) nouveau signal à la clôture de t (si pas de position)
        if long_sig[t] or short_sig[t]:
            side = "long" if long_sig[t] else "short"
            sgn = 1 if side == "long" else -1
            trig = h[t] + CFG["trigger_atr"] * atr[t] if sgn == 1 else l[t] - CFG["trigger_atr"] * atr[t]
            stop = (F.low_n[t] - p["stopbuf"] * atr[t]) if sgn == 1 else (F.high_n[t] + p["stopbuf"] * atr[t])
            pending = {"side": side, "t": t, "trigger": float(trig), "stop": float(stop)}
        t += 1
    return trades, open_pos, pending


def stats(trades):
    if not trades:
        return {"n": 0}
    R = np.array([x["R"] for x in trades])
    wins, losses = R[R > 0], R[R <= 0]
    eq = np.cumsum(R)
    dd = float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()) if len(eq) else 0.0
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None
    sd = float(R.std(ddof=1)) if len(R) > 1 else None
    return {
        "n": int(len(R)),
        "n_long": int(sum(1 for x in trades if x["side"] == "long")),
        "n_short": int(sum(1 for x in trades if x["side"] == "short")),
        "win_rate": round(float((R > 0).mean()), 4),
        "avg_win": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 3) if len(losses) else None,
        "expectancy": round(float(R.mean()), 4),
        "total_R": round(float(R.sum()), 2),
        "profit_factor": None if pf is None else round(pf, 3),
        "max_dd_R": round(dd, 2),
        "t_stat": None if not sd else round(float(R.mean() / (sd / math.sqrt(len(R)))), 2),
        "avg_bars": round(float(np.mean([x["bars"] for x in trades])), 1),
    }


def score(trades):
    if len(trades) < CFG["min_train_trades"]:
        return -np.inf
    R = np.array([x["R"] for x in trades])
    return float(R.mean() * math.sqrt(len(R)))


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Walk-forward et contrôle par le hasard
# --------------------------------------------------------------------------
def walk_forward(F: Features, all_trades):
    dates = F.df.index
    first_valid = dates[np.argmax(F.valid)]
    start_year = first_valid.year + 1
    last = dates[-1]
    windows = []
    y = start_year
    while True:
        tr_start = pd.Timestamp(y, 1, 1)
        tr_end = pd.Timestamp(y + CFG["train_years"], 1, 1)
        te_end = pd.Timestamp(y + CFG["train_years"] + CFG["test_years"], 1, 1)
        if tr_end > last:
            break
        windows.append((tr_start, tr_end, te_end))
        y += CFG["test_years"]

    def in_range(trs, a, b):
        return [x for x in trs if a <= dates[x["in"]] < b]

    oos, wf = [], []
    for (a, b, e) in windows:
        best, best_s = None, -np.inf
        for ci, trs in enumerate(all_trades):
            s = score(in_range(trs, a, b))
            if s > best_s:
                best, best_s = ci, s
        row = {"train": [a.strftime("%Y-%m-%d"), (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
               "test": [b.strftime("%Y-%m-%d"), min(e - pd.Timedelta(days=1), last).strftime("%Y-%m-%d")]}
        if best is None or best_s <= 0:
            row.update({"combo": None, "train_stats": None, "test_stats": {"n": 0}, "note": "aucune configuration positive à l'apprentissage → pas de trade"})
        else:
            te = in_range(all_trades[best], b, e)
            for x in te:
                x2 = dict(x); x2["combo"] = best
                oos.append(x2)
            row.update({"combo": best, "train_stats": stats(in_range(all_trades[best], a, b)), "test_stats": stats(te)})
        wf.append(row)
    return wf, oos


def random_benchmark(F: Features, oos, wf, params, seed=42):
    """Mêmes stops / sorties, mêmes sens, mêmes fenêtres de test, mais dates d'entrée aléatoires."""
    if not oos:
        return None
    rng = random.Random(seed)
    dates = F.df.index
    o, atr = F.o, F.atr
    n = len(o)
    # indices de test par fenêtre
    win_idx = []
    for row in wf:
        a = pd.Timestamp(row["test"][0]); b = pd.Timestamp(row["test"][1])
        ids = np.where((dates >= a) & (dates <= b))[0]
        win_idx.append(ids)
    pools = [[int(i) for i in ids if CFG["stop_lookback"] < i < n - 1] for ids in win_idx]
    def window_of(i):
        for k, ids in enumerate(win_idx):
            if len(ids) and ids[0] <= i <= ids[-1]:
                return k
        return None
    for x in oos:
        x["_w"] = window_of(x["in"])
    means = []
    for _ in range(CFG["random_runs"]):
        Rs = []
        for x in oos:
            k = x["_w"]
            if not pools[k]:
                continue
            s = rng.choice(pools[k])  # séance du "signal" tirée au hasard
            t = s + 1
            p = params[x["combo"]]
            sgn = 1 if x["side"] == "long" else -1
            entry = o[t]
            stop = (F.low_n[s] - p["stopbuf"] * atr[s]) if sgn == 1 else (F.high_n[s] + p["stopbuf"] * atr[s])
            risk = sgn * (entry - stop)
            if risk <= 0:
                continue
            target = None if p["exit"] == "trail" else entry + sgn * float(p["exit"][0]) * risk
            res = manage_position(F, x["side"], t, entry, stop, target, p["exit"], risk, True)
            if res is None:
                continue
            j, px, _ = res
            Rs.append(sgn * (px - entry) / risk - CFG["COST_POINTS"] / risk)
        if Rs:
            means.append(float(np.mean(Rs)))
    actual = float(np.mean([x["R"] for x in oos]))
    means = np.array(means)
    return {
        "runs": int(len(means)),
        "random_mean": round(float(means.mean()), 4),
        "random_p05": round(float(np.percentile(means, 5)), 4),
        "random_p95": round(float(np.percentile(means, 95)), 4),
        "actual_mean": round(actual, 4),
        "share_random_better": round(float((means >= actual).mean()), 4),
    }


# --------------------------------------------------------------------------
# ZigZag descriptif (utilise le futur : description uniquement, jamais de signal)
# --------------------------------------------------------------------------
def zigzag(h, l, pct):
    """Pivots ZigZag : un sommet est validé quand le cours recule de pct % depuis ce sommet,
    un creux quand il remonte de pct %. Renvoie [(indice, 'H'|'L'), ...]."""
    th = pct / 100.0
    piv = []
    trend, hi_i, lo_i = 0, 0, 0
    for i in range(1, len(h)):
        if trend == 0:
            if h[i] > h[hi_i]:
                hi_i = i
            if l[i] < l[lo_i]:
                lo_i = i
            if lo_i < hi_i and h[hi_i] >= l[lo_i] * (1 + th):
                piv.append((lo_i, "L")); trend = 1
            elif hi_i < lo_i and l[lo_i] <= h[hi_i] * (1 - th):
                piv.append((hi_i, "H")); trend = -1
        elif trend == 1:
            if h[i] >= h[hi_i]:
                hi_i = i
            elif l[i] <= h[hi_i] * (1 - th):
                piv.append((hi_i, "H")); trend, lo_i = -1, i
        else:
            if l[i] <= l[lo_i]:
                lo_i = i
            elif h[i] >= l[lo_i] * (1 + th):
                piv.append((lo_i, "L")); trend, hi_i = 1, i
    return piv


def zigzag_research(F: Features, oos):
    h, l = F.h, F.l
    dates = F.df.index
    valid = F.valid
    out = []
    for pct in CFG["zigzag_thresholds"]:
        piv = [(i, t) for (i, t) in zigzag(h, l, pct) if valid[i]]
        amps, durs = [], []
        for a, b in zip(piv[:-1], piv[1:]):
            pa = h[a[0]] if a[1] == "H" else l[a[0]]
            pb = h[b[0]] if b[1] == "H" else l[b[0]]
            amps.append(abs(pb / pa - 1) * 100)
            durs.append(b[0] - a[0])
        nb = len(h)
        near_L = np.zeros(nb, bool); near_H = np.zeros(nb, bool)
        for i, t in piv:
            (near_L if t == "L" else near_H)[max(0, i - 2):i + 3] = True

        def cond_rows(side):
            up = side == "long"
            near = near_L if up else near_H
            ext = F.ext_long if up else F.ext_short
            cnt = F.cnt_long if up else F.cnt_short
            rej = F.rej_long if up else F.rej_short
            rs = (F.rsi_min3 <= 35) if up else (F.rsi_max3 >= 65)
            tr = F.up_trend if up else F.dn_trend
            e = valid & (ext >= 3.5)
            steps = [
                ("Toutes les séances", valid),
                ("Excès ≥ 3,5 ATR", e),
                (f"Excès + RSI {'≤ 35' if up else '≥ 65'}", e & rs),
                (f"Excès + RSI + bougie de rejet", e & rs & rej),
                (f"Excès + {'cours > SMA 200' if up else 'cours < SMA 200'}", e & tr),
                ("Excès + niveau clé (±0,5 ATR, 1 point)", e & (cnt[0.5] >= 1)),
                ("Excès + niveau clé (±0,5 ATR, 2 points)", e & (cnt[0.5] >= 2)),
                ("Excès SANS niveau clé à ±1 ATR", e & (cnt[1.0] == 0)),
            ]
            rows = []
            for name, m in steps:
                k = int(m.sum())
                rows.append({"cond": name, "days": k, "p_pivot": round(float(near[m].mean()), 3) if k else None})
            return rows

        res = {
            "threshold": pct,
            "n_pivots": len(piv),
            "per_year": round(len(piv) / max(1e-9, (dates[-1] - dates[np.argmax(valid)]).days / 365.25), 2),
            "median_amplitude_pct": round(float(np.median(amps)), 2) if amps else None,
            "median_duration_bars": float(np.median(durs)) if durs else None,
            "table_long": cond_rows("long"),
            "table_short": cond_rows("short"),
            "pivots": [[dates[i].strftime("%Y-%m-%d"), t, round(float(h[i] if t == "H" else l[i]), 2)] for i, t in piv],
        }
        if oos:
            first = min(x["sig"] for x in oos)
            piv_oos = [(i, t) for i, t in piv if i >= first]
            hit = sum(1 for x in oos if any(abs(i - x["sig"]) <= 3 and t == ("L" if x["side"] == "long" else "H") for i, t in piv_oos))
            res["signal_precision"] = round(hit / len(oos), 3)
        out.append(res)
    return out


# --------------------------------------------------------------------------
# Évaluation d'un sens (achat ou vente)
# --------------------------------------------------------------------------
def with_cost(trades, cost):
    """Recalcule les R avec un autre coût aller-retour (points d'indice)."""
    out = []
    for x in trades:
        y = dict(x)
        y["R"] = x["R"] + CFG["COST_POINTS"] / x["risk"] - cost / x["risk"]
        out.append(y)
    return out


def best_recent(F, params, all_trades):
    dates = F.df.index
    a = dates[-1] - pd.DateOffset(years=CFG["train_years"])
    best, best_s = None, -np.inf
    for ci, trs in enumerate(all_trades):
        s = score([x for x in trs if dates[x["in"]] >= a])
        if s > best_s:
            best, best_s = ci, s
    if best is None or best_s <= 0:
        return None, a, None
    return params[best], a, stats([x for x in all_trades[best] if dates[x["in"]] >= a])


def evaluate_side(F: Features, side):
    params = list(combos(side))
    all_trades = [simulate(F, p)[0] for p in params]
    wf, oos = walk_forward(F, all_trades)
    bench = random_benchmark(F, oos, wf, params) if oos else None
    st = stats(oos)
    sens = {str(c): stats(with_cost(oos, c)) for c in CFG["cost_sensitivity"]}
    active = bool(st.get("n", 0) > 0 and st["expectancy"] > 0 and bench is not None
                  and bench["share_random_better"] <= CFG["activation_max_random_share"])
    reasons = []
    if not st.get("n"):
        reasons.append("aucun trade hors échantillon")
    else:
        if st["expectancy"] <= 0:
            reasons.append(f"espérance hors échantillon ≤ 0 ({st['expectancy']:+.2f} R, coût {CFG['COST_POINTS']} pts inclus)")
        if bench and bench["share_random_better"] > CFG["activation_max_random_share"]:
            reasons.append(f"{bench['share_random_better']*100:.0f} % des entrées aléatoires font aussi bien (max {CFG['activation_max_random_share']*100:.0f} %)")
    recent, recent_from, recent_stats = best_recent(F, params, all_trades)
    return {
        "params": params, "all_trades": all_trades, "wf": wf, "oos": oos, "bench": bench, "stats": st,
        "cost_sensitivity": sens, "active": active, "inactive_reasons": reasons,
        "recent_best": recent, "recent_from": recent_from, "recent_stats": recent_stats,
    }


def side_today(F: Features, p, t):
    """État du jour pour des paramètres donnés : conditions, nouvel ordre, ordre en attente, position modèle."""
    dates = F.df.index
    cond = F.conditions(p, t)
    sgn = 1 if p["side"] == "long" else -1
    order = None
    if cond["signal"]:
        trig = F.h[t] + CFG["trigger_atr"] * F.atr[t] if sgn == 1 else F.l[t] - CFG["trigger_atr"] * F.atr[t]
        stop = (F.low_n[t] - p["stopbuf"] * F.atr[t]) if sgn == 1 else (F.high_n[t] + p["stopbuf"] * F.atr[t])
        risk = sgn * (trig - stop)
        order = {"side": p["side"], "trigger": round(float(trig), 2), "stop": round(float(stop), 2), "risk_pts": round(float(risk), 2),
                 "target_2R": round(float(trig + sgn * 2 * risk), 2), "target_3R": round(float(trig + sgn * 3 * risk), 2),
                 "exit": p["exit"], "valid_sessions": CFG["order_valid"], "signal_date": dates[t].strftime("%Y-%m-%d")}
    trades, open_pos, pending = simulate(F, p, keep_open=True)
    pend = None
    if pending is not None and order is None and 0 < t - pending["t"] < CFG["order_valid"]:
        pend = {"side": pending["side"], "trigger": round(pending["trigger"], 2), "stop": round(pending["stop"], 2),
                "signal_date": dates[pending["t"]].strftime("%Y-%m-%d"), "sessions_left": CFG["order_valid"] - (t - pending["t"])}
    op = None
    if open_pos is not None:
        op = {"side": open_pos["side"], "signal_date": dates[open_pos["sig"]].strftime("%Y-%m-%d"),
              "entry_date": dates[open_pos["in"]].strftime("%Y-%m-%d"), "entry": round(open_pos["entry"], 2),
              "stop_initial": round(open_pos["stop"], 2), "target": None if open_pos["target"] is None else round(open_pos["target"], 2)}
    return {"conditions": cond, "new_order": order, "pending_order": pend, "model_position": op}, trades, open_pos


def fmt_tr(x, dates):
    return {
        "side": x["side"], "sig": dates[x["sig"]].strftime("%Y-%m-%d"),
        "in": dates[x["in"]].strftime("%Y-%m-%d"), "entry": round(x["entry"], 2), "stop": round(x["stop"], 2),
        "target": None if x["target"] is None else round(x["target"], 2),
        "out": dates[x["out"]].strftime("%Y-%m-%d"), "exit": round(x["exit"], 2), "why": x["why"],
        "R": round(x["R"], 3), "risk": round(x["risk"], 2), "bars": x["bars"], "combo": x.get("combo"),
    }


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def load_frozen(path):
    try:
        with open(path, encoding="utf-8") as f:
            fr = json.load(f)
        if fr.get("version") == VERSION:
            return fr
    except FileNotFoundError:
        pass
    return None


def run(df: pd.DataFrame, source: str, frozen_path: str | None):
    df = add_indicators(clean(df))
    F = Features(df)
    dates = df.index
    t = len(df) - 1
    last = dates[-1]

    ev = {s: evaluate_side(F, s) for s in ("long", "short")}

    # ---- règles gelées
    frozen = load_frozen(frozen_path) if frozen_path else None
    created = False
    if frozen is None:
        frozen = {"version": VERSION, "frozen_on": last.strftime("%Y-%m-%d"),
                  "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                  "note": "Paramètres figés au premier lancement de la V2 (meilleure combinaison des 8 années précédentes). "
                          "Supprimer ce fichier pour re-geler (le suivi à blanc repart alors de zéro).",
                  "long": ev["long"]["recent_best"], "short": ev["short"]["recent_best"]}
        created = True
        if frozen_path:
            os.makedirs(os.path.dirname(os.path.abspath(frozen_path)), exist_ok=True)
            with open(frozen_path, "w", encoding="utf-8") as f:
                json.dump(sanitize(frozen), f, ensure_ascii=False, indent=2)
    fz_date = pd.Timestamp(frozen["frozen_on"])

    sides_out, today, forward_trades, forward_open = {}, {}, [], []
    for s in ("long", "short"):
        E = ev[s]
        p = frozen.get(s)
        if p is not None:
            st_today, trs, open_pos = side_today(F, p, t)
            st_today["emit"] = E["active"]
            st_today["params"] = p
            st_today["label"] = combo_label(p)
            today[s] = st_today
            forward_trades += [fmt_tr(x, dates) for x in trs if dates[x["sig"]] > fz_date]
            if open_pos is not None and dates[open_pos["sig"]] > fz_date:
                forward_open.append(st_today["model_position"])
        else:
            today[s] = None
        eq, cum = [], 0.0
        for x in sorted(E["oos"], key=lambda x: x["out"]):
            cum += x["R"]
            eq.append([dates[x["out"]].strftime("%Y-%m-%d"), round(cum, 3)])
        used = sorted({r["combo"] for r in E["wf"] if r["combo"] is not None})
        stab = {}
        for k in GRID[s]:
            vals = [E["params"][r["combo"]][k] for r in E["wf"] if r["combo"] is not None]
            stab[k] = {str(v): vals.count(v) for v in GRID[s][k]}
        sides_out[s] = {
            "grid_size": len(E["params"]),
            "active": E["active"], "inactive_reasons": E["inactive_reasons"],
            "oos_stats": E["stats"], "cost_sensitivity": E["cost_sensitivity"],
            "random_benchmark": E["bench"], "walk_forward": E["wf"],
            "combos": {str(i): {"params": E["params"][i], "label": combo_label(E["params"][i])} for i in used},
            "param_stability": stab,
            "oos_trades": [fmt_tr(x, dates) for x in sorted(E["oos"], key=lambda x: x["in"])],
            "oos_equity": eq,
            "recent_best": None if E["recent_best"] is None else {"params": E["recent_best"], "label": combo_label(E["recent_best"]),
                                                                  "from": E["recent_from"].strftime("%Y-%m-%d"), "stats": E["recent_stats"]},
        }

    fwd_sorted = sorted(forward_trades, key=lambda x: x["in"])
    zones = level_zones(df, F.pts, t, 0.5)
    close = float(df["close"].iloc[-1])
    atr_now = float(df["atr"].iloc[-1])
    sup = sorted([z for z in zones if z["price"] < close], key=lambda z: -z["price"])[:6]
    res = sorted([z for z in zones if z["price"] >= close], key=lambda z: z["price"])[:6]
    for z in sup + res:
        z["dist_pts"] = round(z["price"] - close, 2)
        z["dist_atr"] = round((z["price"] - close) / atr_now, 2)

    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else None
    return {
        "version": VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "ticker": CFG["ticker"],
        "config": CFG,
        "last": {
            "date": last.strftime("%Y-%m-%d"), "open": round(float(df["open"].iloc[-1]), 2),
            "high": round(float(df["high"].iloc[-1]), 2), "low": round(float(df["low"].iloc[-1]), 2),
            "close": round(close, 2), "prev_close": None if prev_close is None else round(prev_close, 2),
            "atr": round(atr_now, 2), "rsi": round(float(df["rsi"].iloc[-1]), 1), "sma200": round(float(df["sma200"].iloc[-1]), 2),
        },
        "frozen": {"on": frozen["frozen_on"], "created_now": created,
                   "long": None if frozen.get("long") is None else combo_label(frozen["long"]),
                   "short": None if frozen.get("short") is None else combo_label(frozen["short"])},
        "today": today,
        "sides": sides_out,
        "forward": {"trades": fwd_sorted, "stats": stats([{"R": x["R"], "bars": x["bars"], "side": x["side"]} for x in fwd_sorted]),
                    "open": forward_open},
        "levels": {"supports": sup, "resistances": res},
        "zigzag": zigzag_research(F, ev["long"]["oos"] + ev["short"]["oos"]),
        "ohlc": [[d.strftime("%Y-%m-%d"), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2), round(float(r.close), 2)]
                 for d, r in zip(df.index, df.itertuples())],
        "sma200": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(df.index, df["sma200"].values) if np.isfinite(v)],
    }


def sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def email_text(res):
    L = res["last"]
    lines = [f"CAC 40 — clôture du {L['date']} : {L['close']:.2f} (ATR14 {L['atr']:.2f}, RSI14 {L['rsi']:.1f}, SMA200 {L['sma200']:.2f})",
             f"Source : {res['source']} · règles V2 gelées le {res['frozen']['on']}", ""]
    flags = []
    for s, name in (("long", "ACHAT / CALL"), ("short", "VENTE / SHORT")):
        S, T = res["sides"][s], res["today"][s]
        st = S["oos_stats"]
        perf = (f"{st['n']} trades hors échantillon, espérance {st['expectancy']:+.2f} R (coût {res['config']['COST_POINTS']} pts inclus)"
                if st.get("n") else "aucun trade hors échantillon")
        status = "ACTIF" if S["active"] else "DÉSACTIVÉ (" + "; ".join(S["inactive_reasons"]) + ")"
        lines.append(f"[{name}] {status} — {perf}")
        if T is None:
            lines += ["  Pas de règles gelées pour ce sens.", ""]
            continue
        c = T["conditions"]
        parts = [f"excès {'✓' if c['exces']['ok'] else '✗'} ({c['exces']['valeur']}/{c['exces']['seuil']} ATR)"]
        if c["rsi"]["actif"]:
            parts.append(f"RSI {'✓' if c['rsi']['ok'] else '✗'} ({c['rsi']['valeur']} / {c['rsi']['seuil']})")
        if c["rejet"]["actif"]:
            parts.append(f"rejet {'✓' if c['rejet']['ok'] else '✗'}")
        if c["tendance"]["actif"]:
            parts.append(f"tendance {'✓' if c['tendance']['ok'] else '✗'}")
        lines.append("  Conditions : " + ", ".join(parts))
        o = T["new_order"]
        if o:
            if S["active"]:
                flags.append("SIGNAL " + ("ACHAT" if s == "long" else "VENTE"))
                lines += [f"  >>> NOUVEAU SIGNAL : ordre stop {o['trigger']:.2f} (valable {o['valid_sessions']} séances), "
                          f"stop {o['stop']:.2f} (risque {o['risk_pts']:.2f} pts), objectifs 2R {o['target_2R']:.2f} / 3R {o['target_3R']:.2f}, "
                          f"sortie du modèle : {'stop suiveur' if o['exit']=='trail' else o['exit']}"]
            else:
                flags.append("à blanc " + ("achat" if s == "long" else "vente"))
                lines.append(f"  Signal À BLANC (sens désactivé, ne pas trader) : déclenchement {o['trigger']:.2f}, stop {o['stop']:.2f}.")
        elif T["pending_order"]:
            p = T["pending_order"]
            lines.append(f"  Ordre en attente depuis le {p['signal_date']} : déclenchement {p['trigger']:.2f}, stop {p['stop']:.2f}, encore {p['sessions_left']} séance(s).")
        if T["model_position"]:
            p = T["model_position"]
            lines.append(f"  Position modèle ouverte depuis le {p['entry_date']} à {p['entry']:.2f} (stop initial {p['stop_initial']:.2f}).")
        lines.append("")
    fw = res["forward"]["stats"]
    lines.append(f"Suivi à blanc depuis le gel : {fw['n']} trade(s) clôturé(s)" + (f", {fw['total_R']:+.2f} R au total." if fw.get("n") else "."))
    lines += ["", "Supports : " + ", ".join(f"{z['price']:.0f} ({z['touches']}x)" for z in res["levels"]["supports"][:4]),
              "Résistances : " + ", ".join(f"{z['price']:.0f} ({z['touches']}x)" for z in res["levels"]["resistances"][:4]),
              "", "Outil d'aide à la décision, pas un conseil en investissement."]
    return f"CAC40 Swing {L['date']} — {' + '.join(flags) if flags else 'RAS'}", "\n".join(lines)


def send_email(subject, body):
    srv = os.environ.get("SMTP_SERVER")
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASSWORD")
    to = os.environ.get("MAIL_TO") or user
    port = int(os.environ.get("SMTP_PORT") or 465)
    if not (srv and user and pwd and to):
        print("E-mail non envoyé : secrets SMTP_SERVER / SMTP_USER / SMTP_PASSWORD / MAIL_TO absents.")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(srv, port, context=ctx) as s:
            s.login(user, pwd); s.sendmail(user, [a.strip() for a in to.split(",")], msg.as_string())
    else:
        with smtplib.SMTP(srv, port) as s:
            s.starttls(context=ctx); s.login(user, pwd); s.sendmail(user, [a.strip() for a in to.split(",")], msg.as_string())
    print("E-mail envoyé à", to)


def main():
    ap = argparse.ArgumentParser(description="CAC 40 Swing V2")
    ap.add_argument("--csv", help="fichier CSV Date,Open,High,Low,Close (sinon Yahoo Finance)")
    ap.add_argument("--out", default="data/cac40_swing.json")
    ap.add_argument("--frozen", default=CFG["frozen_file"], help="fichier des règles gelées")
    ap.add_argument("--email", action="store_true", help="envoyer le résumé par e-mail (secrets SMTP)")
    a = ap.parse_args()
    if a.csv:
        df, src = load_csv(a.csv), f"CSV {os.path.basename(a.csv)}"
    else:
        df, src = load_yahoo(CFG["ticker"]), f"Yahoo Finance {CFG['ticker']}"
        df, dropped = drop_unfinished_bar(df)
        if dropped:
            src += " (séance en cours exclue)"
            print("Séance du jour non clôturée : exclue de l'analyse.")
    res = run(df, src, a.frozen)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sanitize(res), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    subj, body = email_text(res)
    print(subj); print(body)
    if a.email:
        send_email(subj, body)


if __name__ == "__main__":
    main()
