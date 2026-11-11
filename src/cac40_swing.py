#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAC 40 Swing — moteur d'analyse des retournements sur niveaux clés.

Principe
--------
1. Historique journalier du CAC 40 (Yahoo Finance, ticker ^FCHI, ou CSV).
2. Niveaux clés = anciens points hauts/bas "fractals" (k séances de part et
   d'autre), connus uniquement après confirmation (pas de regard vers le futur).
3. Setup de retournement évalué à la clôture de chaque séance :
     - EXCÈS   : le cours a parcouru au moins E x ATR(14) depuis l'extrême des 15 dernières séances ;
     - NIVEAU  : l'extrême des 3 dernières séances touche un niveau clé (à tol x ATR près) ;
     - REJET   : (option) clôture dans la moitié favorable de la bougie et au-delà de la clôture précédente ;
     - FILTRES : (options) RSI(14) en excès, sens de la tendance de fond (SMA 200).
   Entrée : ordre stop au-delà de l'extrême de la bougie signal (+/- 0,1 ATR), valable 3 séances.
   Stop   : au-delà de l'extrême des 5 dernières séances +/- b x ATR (niveau d'invalidation).
   Sortie : objectif 2R, 3R, ou stop suiveur (break-even à +1R puis 3 ATR sous le plus haut de clôture).
            Sortie forcée après 40 séances.
4. Recherche des paramètres en walk-forward : optimisation sur 8 ans, test sur les
   2 années suivantes jamais vues, fenêtre glissante. Seuls les résultats des
   périodes de test (hors échantillon) sont présentés comme performance.
5. Contrôle : comparaison avec des entrées aléatoires (mêmes stops/sorties).
6. Description des retournements passés (ZigZag 3 %, 5 %, 8 %) — ce ZigZag
   utilise le futur et ne sert QUE de description / d'étiquetage, jamais de signal.

Les résultats sont exprimés en R (multiple du risque initial), hors frais,
spread et financement des turbos, sauf si COST_POINTS est renseigné.

Ce programme est un outil d'aide à la décision, pas un conseil en investissement.
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CFG = {
    "ticker": "^FCHI",
    "atr_n": 14,
    "rsi_n": 14,
    "sma_n": 200,
    "fractal_k": 5,          # un point haut/bas = extrême sur k séances de chaque côté
    "level_lookback": 500,   # ancienneté max d'un niveau clé (séances, ~2 ans)
    "ext_lookback": 15,      # fenêtre de mesure de l'excès (séances)
    "touch_window": 3,       # l'extrême des 3 dernières séances doit toucher le niveau
    "trigger_atr": 0.10,     # déclenchement = extrême de la bougie signal +/- 0,1 ATR
    "order_valid": 3,        # validité de l'ordre stop (séances)
    "stop_lookback": 5,      # stop au-delà de l'extrême des 5 dernières séances
    "max_bars": 40,          # sortie forcée (séances)
    "trail_atr": 3.0,        # stop suiveur (mode "trail")
    "rsi_low": 35,
    "rsi_high": 65,
    "train_years": 8,
    "test_years": 2,
    "min_train_trades": 20,
    "random_runs": 300,
    "zigzag_thresholds": [3.0, 5.0, 8.0],
    "COST_POINTS": 0.0,      # coût aller-retour estimé en points d'indice (0 = hors frais)
}

GRID = {
    "ext": [2.5, 3.5, 4.5],
    "tol": [0.5, 1.0],
    "touches": [1, 2],
    "confirm": ["rejet", "aucun"],
    "rsi": [False, True],
    "trend": ["aucun", "avec"],
    "stopbuf": [0.25, 0.75],
    "exit": ["2R", "3R", "trail"],
}


def combos():
    keys = list(GRID)
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def combo_label(p):
    return (f"excès≥{p['ext']} ATR | niveau ±{p['tol']} ATR ({p['touches']}+ point(s)) | rejet:{'oui' if p['confirm']=='rejet' else 'non'} | "
            f"RSI:{'oui' if p['rsi'] else 'non'} | tendance:{p['trend']} | stop +{p['stopbuf']} ATR | sortie:{p['exit']}")


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
    tols = sorted(set(GRID["tol"]))
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
        self.ext_long = (hh - lo3) / self.atr      # baisse parcourue avant un rebond
        self.ext_short = (hi3 - ll) / self.atr     # hausse parcourue avant un repli
        rng = self.h - self.l
        pos = np.where(rng > 0, (self.c - self.l) / np.where(rng > 0, rng, 1), 0.5)
        pc = np.r_[np.nan, self.c[:-1]]
        self.rej_long = (pos >= 0.5) & (self.c > pc)
        self.rej_short = (pos <= 0.5) & (self.c < pc)
        rsi_min3 = df["rsi"].rolling(tw).min().values
        rsi_max3 = df["rsi"].rolling(tw).max().values
        self.rsi_long = rsi_min3 <= CFG["rsi_low"]
        self.rsi_short = rsi_max3 >= CFG["rsi_high"]
        self.up_trend = self.c > self.sma
        self.dn_trend = self.c < self.sma
        self.pts = fractal_points(self.h, self.l, CFG["fractal_k"])
        (self.d_long, self.d_short, self.lvl_long, self.lvl_short,
         self.cnt_long, self.cnt_short) = level_distances(df, self.pts)
        self.valid = np.isfinite(self.atr) & np.isfinite(self.sma) & (np.arange(n) >= CFG["sma_n"])
        sl = CFG["stop_lookback"]
        self.low_n = df["low"].rolling(sl).min().values
        self.high_n = df["high"].rolling(sl).max().values

    def conditions(self, p, t):
        """Détail des conditions à la séance t (pour l'état du jour)."""
        out = {}
        for side in ("long", "short"):
            ext = self.ext_long if side == "long" else self.ext_short
            dist = self.d_long if side == "long" else self.d_short
            cnt = (self.cnt_long if side == "long" else self.cnt_short)[p["tol"]]
            lvl = self.lvl_long if side == "long" else self.lvl_short
            rej = self.rej_long if side == "long" else self.rej_short
            rs = self.rsi_long if side == "long" else self.rsi_short
            tr = self.up_trend if side == "long" else self.dn_trend
            out[side] = {
                "exces": {"ok": bool(ext[t] >= p["ext"]), "valeur": round(float(ext[t]), 2), "seuil": p["ext"]},
                "niveau": {"ok": bool(cnt[t] >= p["touches"]), "valeur": None if not np.isfinite(dist[t]) else round(float(dist[t]), 2),
                           "seuil": p["tol"], "niveau": None if not np.isfinite(lvl[t]) else round(float(lvl[t]), 2),
                           "touches": int(cnt[t]), "touches_min": p["touches"]},
                "rejet": {"ok": bool(rej[t]) if p["confirm"] == "rejet" else True, "actif": p["confirm"] == "rejet", "constate": bool(rej[t])},
                "rsi": {"ok": bool(rs[t]) if p["rsi"] else True, "actif": bool(p["rsi"]), "valeur": round(float(self.rsi[t]), 1)},
                "tendance": {"ok": bool(tr[t]) if p["trend"] == "avec" else True, "actif": p["trend"] == "avec",
                             "sma200": round(float(self.sma[t]), 2)},
            }
            out[side]["signal"] = all(v["ok"] for k, v in out[side].items())
        return out

    def signals(self, p):
        base = self.valid
        lg = base & (self.ext_long >= p["ext"]) & (self.cnt_long[p["tol"]] >= p["touches"])
        sh = base & (self.ext_short >= p["ext"]) & (self.cnt_short[p["tol"]] >= p["touches"])
        if p["confirm"] == "rejet":
            lg &= self.rej_long
            sh &= self.rej_short
        if p["rsi"]:
            lg &= self.rsi_long
            sh &= self.rsi_short
        if p["trend"] == "avec":
            lg &= self.up_trend
            sh &= self.dn_trend
        both = lg & sh
        return lg & ~both, sh & ~both


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
# ZigZag descriptif (utilise le futur : description uniquement)
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
    out = []
    valid = F.valid
    for pct in CFG["zigzag_thresholds"]:
        piv = [(i, t) for (i, t) in zigzag(h, l, pct) if valid[i]]
        amps, durs = [], []
        for a, b in zip(piv[:-1], piv[1:]):
            pa = h[a[0]] if a[1] == "H" else l[a[0]]
            pb = h[b[0]] if b[1] == "H" else l[b[0]]
            amps.append(abs(pb / pa - 1) * 100)
            durs.append(b[0] - a[0])
        # probabilité qu'un pivot ZigZag (du bon sens) se trouve à +/- 2 séances, selon les conditions réunies
        nb = len(h)
        near_L = np.zeros(nb, bool); near_H = np.zeros(nb, bool)
        for i, t in piv:
            arr = near_L if t == "L" else near_H
            arr[max(0, i - 2):i + 3] = True
        def cond_rows(side):
            near = near_L if side == "long" else near_H
            ext = F.ext_long if side == "long" else F.ext_short
            cnt = F.cnt_long if side == "long" else F.cnt_short
            rej = F.rej_long if side == "long" else F.rej_short
            rs = F.rsi_long if side == "long" else F.rsi_short
            rows = []
            steps = [
                ("Toutes les séances", valid),
                ("Excès ≥ 3,5 ATR", valid & (ext >= 3.5)),
                ("+ niveau clé (±0,5 ATR, 1 point)", valid & (ext >= 3.5) & (cnt[0.5] >= 1)),
                ("+ niveau clé (±0,5 ATR, 2 points)", valid & (ext >= 3.5) & (cnt[0.5] >= 2)),
                ("+ bougie de rejet", valid & (ext >= 3.5) & (cnt[0.5] >= 2) & rej),
                ("+ RSI en excès", valid & (ext >= 3.5) & (cnt[0.5] >= 2) & rej & rs),
                ("Excès ≥ 3,5 ATR + rejet, SANS niveau", valid & (ext >= 3.5) & (cnt[1.0] == 0) & rej),
            ]
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
        # précision / rappel des signaux hors échantillon vis-à-vis des pivots
        if oos:
            first = min(x["sig"] for x in oos)
            piv_oos = [(i, t) for i, t in piv if i >= first]
            tol = 3
            hit = 0
            for x in oos:
                want = "L" if x["side"] == "long" else "H"
                if any(abs(i - x["sig"]) <= tol and t == want for i, t in piv_oos):
                    hit += 1
            caught = 0
            for i, t in piv_oos:
                want = "long" if t == "L" else "short"
                if any(abs(i - x["sig"]) <= tol and x["side"] == want for x in oos):
                    caught += 1
            res["signal_precision"] = round(hit / len(oos), 3)
            res["pivot_recall"] = round(caught / len(piv_oos), 3) if piv_oos else None
        out.append(res)
    return out


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def run(df: pd.DataFrame, source: str):
    df = add_indicators(clean(df))
    F = Features(df)
    params = list(combos())
    all_trades = []
    for p in params:
        trs, _, _ = simulate(F, p)
        all_trades.append(trs)
    wf, oos = walk_forward(F, all_trades)
    oos_stats = stats(oos)
    bench = random_benchmark(F, oos, wf, params)

    # configuration actuelle = meilleure sur les 8 dernières années
    dates = df.index
    last = dates[-1]
    a = last - pd.DateOffset(years=CFG["train_years"])
    cur_best, cur_s = None, -np.inf
    for ci, trs in enumerate(all_trades):
        s = score([x for x in trs if dates[x["in"]] >= a])
        if s > cur_s:
            cur_best, cur_s = ci, s
    current = None
    t = len(df) - 1
    if cur_best is not None and cur_s > 0:
        p = params[cur_best]
        _, open_pos, pending = simulate(F, p, keep_open=True)
        cond = F.conditions(p, t)
        order = None
        for side in ("long", "short"):
            if cond[side]["signal"]:
                sgn = 1 if side == "long" else -1
                trig = F.h[t] + CFG["trigger_atr"] * F.atr[t] if sgn == 1 else F.l[t] - CFG["trigger_atr"] * F.atr[t]
                stop = (F.low_n[t] - p["stopbuf"] * F.atr[t]) if sgn == 1 else (F.high_n[t] + p["stopbuf"] * F.atr[t])
                risk = sgn * (trig - stop)
                order = {
                    "side": side, "trigger": round(float(trig), 2), "stop": round(float(stop), 2), "risk_pts": round(float(risk), 2),
                    "target_2R": round(float(trig + sgn * 2 * risk), 2), "target_3R": round(float(trig + sgn * 3 * risk), 2),
                    "valid_sessions": CFG["order_valid"],
                }
        pend = None
        if pending is not None and order is None and t - pending["t"] < CFG["order_valid"]:
            pend = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in pending.items()}
            pend["signal_date"] = dates[pending["t"]].strftime("%Y-%m-%d")
            pend["sessions_left"] = CFG["order_valid"] - (t - pending["t"])
            del pend["t"]
        op = None
        if open_pos is not None:
            op = {"side": open_pos["side"], "entry_date": dates[open_pos["in"]].strftime("%Y-%m-%d"),
                  "entry": round(open_pos["entry"], 2), "stop_initial": round(open_pos["stop"], 2),
                  "target": None if open_pos["target"] is None else round(open_pos["target"], 2)}
        sel = {}
        for k in GRID:
            vals = [params[r["combo"]][k] for r in wf if r["combo"] is not None]
            sel[k] = {str(v): vals.count(v) for v in GRID[k]}
        current = {
            "combo": cur_best, "params": p, "label": combo_label(p),
            "train_from": a.strftime("%Y-%m-%d"),
            "train_stats": stats([x for x in all_trades[cur_best] if dates[x["in"]] >= a]),
            "conditions": cond, "new_order": order, "pending_order": pend, "model_position": op,
            "param_stability": sel,
        }

    zones = level_zones(df, F.pts, t, 0.5)
    close = float(df["close"].iloc[-1])
    sup = sorted([z for z in zones if z["price"] < close], key=lambda z: -z["price"])[:6]
    res = sorted([z for z in zones if z["price"] >= close], key=lambda z: z["price"])[:6]
    for z in sup + res:
        z["dist_pts"] = round(z["price"] - close, 2)
        z["dist_atr"] = round((z["price"] - close) / float(df["atr"].iloc[-1]), 2)

    def fmt_tr(x):
        return {
            "side": x["side"], "sig": dates[x["sig"]].strftime("%Y-%m-%d"),
            "in": dates[x["in"]].strftime("%Y-%m-%d"), "entry": round(x["entry"], 2), "stop": round(x["stop"], 2),
            "target": None if x["target"] is None else round(x["target"], 2),
            "out": dates[x["out"]].strftime("%Y-%m-%d"), "exit": round(x["exit"], 2), "why": x["why"],
            "R": round(x["R"], 3), "bars": x["bars"], "combo": x.get("combo"),
        }

    eq, cum = [], 0.0
    for x in sorted(oos, key=lambda x: x["out"]):
        cum += x["R"]
        eq.append([dates[x["out"]].strftime("%Y-%m-%d"), round(cum, 3)])

    used = sorted({r["combo"] for r in wf if r["combo"] is not None} | ({cur_best} if cur_best is not None else set()))
    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "ticker": CFG["ticker"],
        "config": {k: v for k, v in CFG.items()},
        "grid_size": len(params),
        "last": {
            "date": last.strftime("%Y-%m-%d"), "open": round(float(df["open"].iloc[-1]), 2),
            "high": round(float(df["high"].iloc[-1]), 2), "low": round(float(df["low"].iloc[-1]), 2),
            "close": round(close, 2), "atr": round(float(df["atr"].iloc[-1]), 2),
            "rsi": round(float(df["rsi"].iloc[-1]), 1), "sma200": round(float(df["sma200"].iloc[-1]), 2),
        },
        "levels": {"supports": sup, "resistances": res},
        "current": current,
        "walk_forward": wf,
        "combos": {str(i): {"params": params[i], "label": combo_label(params[i])} for i in used},
        "oos_stats": oos_stats,
        "oos_by_side": {s: stats([x for x in oos if x["side"] == s]) for s in ("long", "short")},
        "random_benchmark": bench,
        "oos_trades": [fmt_tr(x) for x in sorted(oos, key=lambda x: x["in"])],
        "oos_equity": eq,
        "zigzag": zigzag_research(F, oos),
        "ohlc": [[d.strftime("%Y-%m-%d"), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2), round(float(r.close), 2)]
                 for d, r in zip(df.index, df.itertuples())],
        "sma200": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(df.index, df["sma200"].values) if np.isfinite(v)],
    }
    return out


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
    cur = res["current"]
    lines = [f"CAC 40 — clôture du {L['date']} : {L['close']:.2f} (ATR14 {L['atr']:.2f}, RSI14 {L['rsi']:.1f}, SMA200 {L['sma200']:.2f})", ""]
    subject_flag = "RAS"
    if cur is None:
        lines.append("Aucune configuration positive sur les 8 dernières années : pas de signal.")
    else:
        if cur["new_order"]:
            o = cur["new_order"]
            subject_flag = "SIGNAL " + ("ACHAT (call)" if o["side"] == "long" else "VENTE (short)")
            lines += [
                f"NOUVEAU SIGNAL {('ACHAT / CALL' if o['side']=='long' else 'VENTE / SHORT')}",
                f"  Ordre stop de déclenchement : {o['trigger']:.2f} (valable {o['valid_sessions']} séances)",
                f"  Stop (invalidation)         : {o['stop']:.2f}  (risque {o['risk_pts']:.2f} pts)",
                f"  Objectifs                   : 2R {o['target_2R']:.2f} | 3R {o['target_3R']:.2f}",
                "",
            ]
        elif cur["pending_order"]:
            o = cur["pending_order"]
            subject_flag = "ordre en attente"
            lines += [f"Ordre en attente ({o['side']}) depuis le {o['signal_date']} : déclenchement {o['trigger']:.2f}, stop {o['stop']:.2f}, encore {o['sessions_left']} séance(s).", ""]
        if cur["model_position"]:
            p = cur["model_position"]
            lines += [f"Position modèle ouverte : {p['side']} depuis le {p['entry_date']} à {p['entry']:.2f} (stop initial {p['stop_initial']:.2f}).", ""]
        for side, name in (("long", "Rebond haussier"), ("short", "Retournement baissier")):
            c = cur["conditions"][side]
            parts = [f"excès {'✓' if c['exces']['ok'] else '✗'} ({c['exces']['valeur']}/{c['exces']['seuil']} ATR)",
                     f"niveau {'✓' if c['niveau']['ok'] else '✗'} ({c['niveau']['valeur']} ATR de {c['niveau']['niveau']})"]
            if c["rejet"]["actif"]:
                parts.append(f"rejet {'✓' if c['rejet']['ok'] else '✗'}")
            if c["rsi"]["actif"]:
                parts.append(f"RSI {'✓' if c['rsi']['ok'] else '✗'} ({c['rsi']['valeur']})")
            if c["tendance"]["actif"]:
                parts.append(f"tendance {'✓' if c['tendance']['ok'] else '✗'}")
            lines.append(f"{name} : " + ", ".join(parts))
        lines += ["", "Configuration utilisée : " + cur["label"]]
    lines += ["", "Supports : " + ", ".join(f"{z['price']:.0f} ({z['touches']}x)" for z in res["levels"]["supports"][:4]),
              "Résistances : " + ", ".join(f"{z['price']:.0f} ({z['touches']}x)" for z in res["levels"]["resistances"][:4])]
    s = res["oos_stats"]
    if s.get("n"):
        lines += ["", f"Historique hors échantillon : {s['n']} trades, réussite {s['win_rate']*100:.0f} %, espérance {s['expectancy']:+.2f} R/trade (hors frais)."]
    lines += ["", "Outil d'aide à la décision, pas un conseil en investissement."]
    return f"CAC40 Swing {L['date']} — {subject_flag}", "\n".join(lines)


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
    ap = argparse.ArgumentParser(description="CAC 40 Swing — analyse des retournements sur niveaux clés")
    ap.add_argument("--csv", help="fichier CSV Date,Open,High,Low,Close (sinon Yahoo Finance)")
    ap.add_argument("--out", default="data/cac40_swing.json")
    ap.add_argument("--email", action="store_true", help="envoyer le résumé par e-mail (secrets SMTP)")
    a = ap.parse_args()
    if a.csv:
        df, src = load_csv(a.csv), f"CSV {os.path.basename(a.csv)}"
    else:
        df, src = load_yahoo(CFG["ticker"]), f"Yahoo Finance {CFG['ticker']}"
    res = run(df, src)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sanitize(res), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    subj, body = email_text(res)
    print(subj); print(body)
    if a.email:
        send_email(subj, body)


if __name__ == "__main__":
    main()
