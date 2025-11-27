# -*- coding: utf-8 -*-
"""
GELİŞMİŞ FUTBOL TAHMİN MOTORU — GUI + TOPLU BACKTEST + DETAY & KALİBRASYON
===========================================================================

Bu sürüm:
- Otomatik detaylı kayıt: matches.csv, league_summary.csv, sweep_ou.csv, sweep_btts.csv,
  reliability.csv, summary.json, report.txt (+ opsiyonel calibration.json).
- Klasör: backtests/run_YYYYMMDD_HHMMSS/ (ayrıca backtests/history.csv’yi günceller).
- Backtest detay penceresi + 'Klasörü Aç' ve 'CSV dışa aktar' butonları.
- Kalibrasyon önerileri: 1X2 sıcaklık (T), OU/KG push (k_over/k_btts), skor gamma (g).
- Tahminlerde diskteki kalibrasyonlar otomatik uygulanır.

Not: Tahminler olasılıktır; %100 garanti vermez. Sert modlar kararları netleştirir, kalibrasyon
LogLoss/Brier’ı iyileştirir (olasılık güvenilirliği artar).
"""

from __future__ import annotations

import os
import csv
import json
import math
import time
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from scipy.stats import poisson
from bs4 import BeautifulSoup

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ======================================================================
# AYARLAR
# ======================================================================

API_SPORTS_KEY = (os.getenv("API_SPORTS_KEY") or "47e7c72a430f8e051bcf0f3817a197f3").strip()
API_BASE = "https://v3.football.api-sports.io/"

OWM_KEY = (os.getenv("OWM_KEY") or "3925f46e8cace22f01d1afc560c4814f").strip()
OWM_GEO = "http://api.openweathermap.org/geo/1.0/direct"
OWM_FC  = "https://api.openweathermap.org/data/2.5/forecast"

RECENT_MATCHES_DEFAULT = 12
HALF_LIFE_DEFAULT = 6
HOME_FIELD_ADV = 0.12
BASE_HOME_G = 1.55
BASE_AWAY_G = 1.25
LINEUP_STRICT_WINDOW_MIN = 60
TR_TZ = timezone(timedelta(hours=3))
TZ_NAME = "Europe/Istanbul"

BANKO_P = 0.75
NORMAL_P = 0.68

MAJOR_LEAGUES = {
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "UEFA Champions League", "UEFA Europa League", "UEFA Europa Conference League",
    "Süper Lig", "Eredivisie", "Primeira Liga", "MLS",
    "Brasileirão", "Argentina Liga Profesional", "Liga Portugal"
}

FORMATION_ADJUST = {
    "4-3-3":   (1.05, 0.98),
    "4-2-3-1": (1.03, 0.99),
    "3-5-2":   (1.03, 0.98),
    "4-4-2":   (1.00, 1.00),
    "3-4-3":   (1.06, 0.95),
    "5-4-1":   (0.95, 1.06),
    "5-3-2":   (0.97, 1.03),
    "4-1-4-1": (0.99, 1.01),
    "4-5-1":   (0.98, 1.02),
}

HEADERS_TM = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8"
}

# ======================================================================
# SERTLİK MODU
# ======================================================================

@dataclass
class SharpenPreset:
    name: str
    gamma: float     # skor matrisi keskinliği (≥0.90)
    peak: float      # favori skor piklerine ekstra güç (oran)
    ou_push: float   # OU/KG'yi 0.5 merkezinden uzaklaştırma kuvveti

SHARPEN_PRESETS: Dict[str, SharpenPreset] = {
    "Standart": SharpenPreset("Standart", gamma=1.00, peak=0.00, ou_push=0.00),
    "Sert":     SharpenPreset("Sert",     gamma=1.10, peak=0.08, ou_push=0.05),
    "Maks":     SharpenPreset("Maks",     gamma=1.22, peak=0.14, ou_push=0.10),
}

# ======================================================================
# KALİBRASYON (diskte saklanır)
# ======================================================================

CALIB_PATH = "calibration.json"
DEFAULT_CALIB = {
    "oneXtwo_T": 1.00,   # 1X2 sıcaklık (T>1 → yumuşar, T<1 → sertleşir)
    "over_push": 0.00,   # OU merkez push
    "btts_push": 0.00,   # KG merkez push
    "score_gamma": 1.00  # Skor matrisi global güç
}

def _sanitize_calib(d: Dict[str, Any]) -> Dict[str, float]:
    out = DEFAULT_CALIB.copy()
    if isinstance(d, dict):
        for k in out.keys():
            try:
                v=float(d.get(k, out[k]))
                if k=="oneXtwo_T": v = float(max(0.5, min(2.0, v)))
                if k=="score_gamma": v = float(max(0.8, min(1.4, v)))
                if k in ("over_push","btts_push"): v = float(max(-0.4, min(0.4, v)))
                out[k]=v
            except Exception:
                pass
    return out

def load_calibration() -> Dict[str, float]:
    try:
        if os.path.exists(CALIB_PATH):
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                return _sanitize_calib(json.load(f))
    except Exception:
        pass
    return DEFAULT_CALIB.copy()

def save_calibration(calib: Dict[str, float]) -> None:
    try:
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(_sanitize_calib(calib), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Kalibrasyon yazma hatası:", e)

CALIB = load_calibration()

# ======================================================================
# YARDIMCILAR
# ======================================================================

def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def clamp01(x: float) -> float: return clamp(x, 0.0, 1.0)

def exp_weights(n: int, half_life: float) -> np.ndarray:
    if n <= 0: return np.array([])
    lam = math.log(2.0) / max(half_life, 1e-6)
    w = np.exp(-lam * np.arange(n))
    return w / w.sum()

def time_weighted_avg(values: List[float], half_life: float) -> float:
    if not values: return 0.0
    w = exp_weights(len(values), half_life)
    return float(np.dot(w, np.array(values)))

def parse_tr_date(s: str) -> Optional[str]:
    try:
        dt = datetime.strptime(s.strip(), "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def fmt_tr_hour(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(TR_TZ).strftime("%H:%M")
    except Exception:
        return "-"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def pct(p: float) -> str:
    return f"{100*clamp01(p):.1f}%"

def safe_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except Exception: return default

def try_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def eur_fmt(x: float) -> str:
    if x <= 0: return "-"
    return f"€{x:,.0f}".replace(",", ".")

def ensure_dir(path: str):
    try: os.makedirs(path, exist_ok=True)
    except Exception: pass

def fmt_opt(x: Optional[float], nd: int = 3, dash: str = "-") -> str:
    if x is None: return dash
    try: return f"{x:.{nd}f}"
    except Exception: return dash

# ======================================================================
# CACHE
# ======================================================================

class APICache:
    def __init__(self):
        self.store: Dict[str, Tuple[float, Any]] = {}
        self.ttl_default = 180.0  # saniye

    def _key(self, endpoint: str, params: Dict[str, Any]) -> str:
        items = sorted(params.items())
        return endpoint + "|" + "&".join(f"{k}={v}" for k,v in items)

    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        k=self._key(endpoint, params)
        if k in self.store:
            t,val=self.store[k]
            if time.time()-t < self.ttl_default: return val
            del self.store[k]
        return None

    def put(self, endpoint: str, params: Dict[str, Any], value: Any):
        self.store[self._key(endpoint, params)] = (time.time(), value)

CACHE = APICache()

# ======================================================================
# API-SPORTS
# ======================================================================

class ApiSports:
    def __init__(self, api_key: str):
        self.h = {"x-apisports-key": api_key}

    def _get(self, endpoint: str, params: Dict[str, Any], timeout: int = 25) -> Any:
        cached = CACHE.get(endpoint, params)
        if cached is not None: return cached
        try:
            r = requests.get(API_BASE + endpoint, headers=self.h, params=params, timeout=timeout)
            r.raise_for_status()
            j = r.json(); resp = j.get("response", None)
            CACHE.put(endpoint, params, resp)
            return resp
        except Exception as e:
            print("API Hatası:", e, endpoint, params)
            return None

    def fixtures_date_raw(self, date_iso: str, tz: str = TZ_NAME) -> List[Dict]:
        return self._get("fixtures", {"date": date_iso, "timezone": tz}) or []

    def fixtures_range_raw(self, frm: str, to: str, tz: str = TZ_NAME) -> List[Dict]:
        return self._get("fixtures", {"from": frm, "to": to, "timezone": tz}) or []

    def fixtures_by_date_smart(self, date_iso: str) -> List[Dict]:
        res=self.fixtures_date_raw(date_iso)
        if res: return res
        res=self.fixtures_range_raw(date_iso, date_iso)
        if res: return res
        dt=datetime.strptime(date_iso,"%Y-%m-%d")
        prev=(dt - timedelta(days=1)).strftime("%Y-%m-%d")
        nxt=(dt + timedelta(days=1)).strftime("%Y-%m-%d")
        res=self.fixtures_range_raw(prev, nxt)
        out=[]
        for m in res or []:
            try:
                dt_tr=datetime.fromisoformat(m["fixture"]["date"].replace("Z","+00:00")).astimezone(TR_TZ)
                if dt_tr.strftime("%Y-%m-%d")==dt.astimezone(TR_TZ).strftime("%Y-%m-%d"):
                    out.append(m)
            except Exception: pass
        return out

    def fixtures_league_window(self, league_id: int, season: int, days: int = 120) -> List[Dict]:
        to_dt=datetime.now(timezone.utc); frm_dt=to_dt - timedelta(days=120)
        return self._get("fixtures", {"league": league_id, "season": season,
                                      "from": frm_dt.strftime("%Y-%m-%d"), "to": to_dt.strftime("%Y-%m-%d")}) or []

    def lineups(self, fixture_id: int) -> Optional[List[Dict]]:
        return self._get("fixtures/lineups", {"fixture": fixture_id})

    def h2h_last5(self, home_id: int, away_id: int) -> List[Dict]:
        return self._get("fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 5}) or []

    def fixtures_team_between(self, team_id: int, frm: str, to: str, limit: int = 160) -> List[Dict]:
        lst=self._get("fixtures", {"team": team_id, "from": frm, "to": to}) or []
        lst.sort(key=lambda x: x["fixture"].get("timestamp",0))
        return lst[-limit:]

    def team_recent(self, team_id: int, n: int) -> List[Dict]:
        return self._get("fixtures", {"team": team_id, "last": n}) or []

    def team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        return self._get("teams/statistics", {"league": league_id, "season": season, "team": team_id}) or {}

    def standings(self, league_id: int, season: int) -> Any:
        return self._get("standings", {"league": league_id, "season": season}) or {}

    def venues(self, venue_id: int) -> Any:
        return self._get("venues", {"id": venue_id}) or {}

# ======================================================================
# VERİ SINIFLARI
# ======================================================================

@dataclass
class TeamStrength:
    attack_g: float
    defense_g: float
    attack_xg: Optional[float] = None
    defense_xg: Optional[float] = None
    shots: Optional[float] = None
    shots_on: Optional[float] = None
    corners_for: Optional[float] = None
    corners_against: Optional[float] = None
    cards_yellow: Optional[float] = None

@dataclass
class LeagueContext:
    mu_home: float
    mu_away: float
    mu_total: float
    var_total: float
    corr_HA: float
    hfa_goals: float
    first_half_share: float
    bin_01: float
    bin_23: float
    bin_45: float
    bin_6p: float
    ht_draw: float
    ft_draw: float
    btts_rate: float

@dataclass
class RecencyPreset:
    name: str
    last_n: int
    half_life: float

RECENCY_PRESETS: Dict[str, RecencyPreset] = {
    "Son 3": RecencyPreset("Son 3", 3, 2.0),
    "Son 5": RecencyPreset("Son 5", 5, 3.0),
    "Son 7": RecencyPreset("Son 7", 7, 4.0),
    "Son 10": RecencyPreset("Son 10", 10, 5.0),
    "Oto": RecencyPreset("Oto", RECENT_MATCHES_DEFAULT, HALF_LIFE_DEFAULT),
}

# ======================================================================
# TM (kadro değeri)
# ======================================================================

TM_BASE = "https://transfermarkt-api.fly.dev"

def parse_tm_value_to_eur(val: Any) -> float:
    if val is None: return 0.0
    s=str(val).lower().replace('€','').replace(',','').strip()
    if s in ('','-','—'): return 0.0
    mult=1.0
    if 'bn' in s: mult=1_000_000_000; s=s.replace('bn','')
    elif 'm' in s: mult=1_000_000; s=s.replace('m','')
    elif 'th.' in s or 'k' in s: mult=1_000; s=s.replace('th.','').replace('k','')
    try: return float(s)*mult
    except Exception: return 0.0

def tmapi_squad_value(club_id: int, season: Optional[int] = None, delay: float = 0.5) -> Tuple[float, List[Dict]]:
    params={'season': season} if season else {}
    try:
        r=requests.get(f"{TM_BASE}/clubs/{club_id}/players", params=params, timeout=30); r.raise_for_status()
        payload=r.json(); players=payload.get('players', payload)
    except Exception:
        return 0.0, []
    rows=[]; total=0.0
    for p in players:
        pid=p.get('id') or p.get('playerId') or p.get('player_id')
        name=p.get('name') or p.get('playerName') or ''
        if not pid: continue
        time.sleep(delay)
        mv=0.0
        try:
            r2=requests.get(f"{TM_BASE}/players/{pid}/market_value", timeout=30); r2.raise_for_status()
            j=r2.json(); val=j.get('marketValue') if isinstance(j,dict) else j
            mv=parse_tm_value_to_eur(val) if isinstance(val,str) else float(val or 0)
        except Exception:
            mv=0.0
        rows.append({'player_id':pid,'name':name,'eur_value':mv}); total+=mv
    return total, rows

def tm_guess_club_id_by_search(team_name: str) -> Optional[int]:
    q=team_name.strip().replace(" ","+")
    url=f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={q}"
    try:
        r=requests.get(url, headers=HEADERS_TM, timeout=30); r.raise_for_status()
        soup=BeautifulSoup(r.text,"lxml")
        for a in soup.select("a[href*='/verein/']"):
            href=a.get("href","")
            if "/verein/" in href:
                try: return int(href.split("/verein/")[-1].split("/")[0])
                except Exception: continue
    except Exception:
        return None
    return None

def tm_scrape_squad_value(club_id: int, season: int) -> Tuple[float, List[Dict]]:
    url=f"https://www.transfermarkt.com/-/startseite/verein/{club_id}/saison_id/{season}"
    try:
        r=requests.get(url, headers=HEADERS_TM, timeout=30); r.raise_for_status()
    except Exception:
        return 0.0, []
    soup=BeautifulSoup(r.text,"lxml"); total=0.0; rows=[]
    for tr in soup.select("table.items tbody tr"):
        cell=tr.select_one("td.rechts.hauptlink"); name_cell=tr.select_one("td.posrela table tr:nth-of-type(1) td:nth-of-type(2) a")
        if not cell: continue
        mv=parse_tm_value_to_eur(cell.get_text(strip=True)); name=name_cell.get_text(strip=True) if name_cell else ""
        if mv>0: rows.append({"name":name,"eur_value":mv}); total+=mv
    return total, rows

def squad_market_value(team_name: str, prefer_club_id: Optional[int] = None, season: Optional[int] = None) -> float:
    cid=prefer_club_id or tm_guess_club_id_by_search(team_name)
    if not cid: return 0.0
    total,_=tmapi_squad_value(cid, season=season or None)
    if total<=0.0 and season:
        total,_=tm_scrape_squad_value(cid, season=season)
    return float(total)

def market_value_adjustments(value_home: float, value_away: float) -> float:
    if value_home<=0 or value_away<=0: return 1.0
    ratio=math.log(value_home/value_away); w=0.10
    f=1.0 + (1.0/(1.0+math.exp(-w*ratio))-0.5)*0.08
    return clamp(f, 0.94, 1.06)

# ======================================================================
# HAVA
# ======================================================================

def owm_geocode_city(city: str, country: Optional[str]) -> Optional[Tuple[float, float]]:
    if not city: return None
    params={"q": f"{city},{country or ''}".strip(","), "limit":1, "appid": OWM_KEY}
    try:
        r=requests.get(OWM_GEO, params=params, timeout=20); r.raise_for_status()
        arr=r.json()
        if isinstance(arr,list) and arr:
            lat=try_float(arr[0].get("lat")); lon=try_float(arr[0].get("lon"))
            if lat and lon: return lat,lon
    except Exception:
        return None
    return None

def owm_forecast_nearest(lat: float, lon: float, kickoff_iso: str) -> Optional[Dict]:
    params={"lat":lat,"lon":lon,"appid":OWM_KEY,"units":"metric","lang":"tr"}
    try:
        r=requests.get(OWM_FC, params=params, timeout=25); r.raise_for_status(); j=r.json()
    except Exception:
        return None
    target=None
    try:
        kickoff_dt=datetime.fromisoformat(kickoff_iso.replace("Z","+00:00"))
        best=10**9
        for it in j.get("list",[]):
            t=it.get("dt"); 
            if not t: continue
            dt=datetime.fromtimestamp(int(t), tz=timezone.utc)
            diff=abs((dt-kickoff_dt).total_seconds())
            if diff<best: best=diff; target=it
    except Exception:
        return None
    return target

def weather_adjustments(forecast: Optional[Dict], surface: Optional[str]) -> Tuple[float, float, Dict]:
    lam_mult=1.0; card_mult=1.0; info={}
    if forecast:
        main=(forecast.get("weather") or [{}])[0].get("description","-")
        wind=try_float(forecast.get("wind",{}).get("speed"),0.0)
        rain=try_float((forecast.get("rain") or {}).get("3h") or forecast.get("rain",{}).get("1h"),0.0)
        temp=try_float(forecast.get("main",{}).get("temp"),16.0)
        hum=try_float(forecast.get("main",{}).get("humidity"),60.0)
        if wind>=8.0: lam_mult*=0.93
        elif wind>=5.0: lam_mult*=0.96
        if rain>=6.0: lam_mult*=0.90; card_mult*=1.06
        elif rain>=2.0: lam_mult*=0.95; card_mult*=1.03
        if temp>=30: lam_mult*=0.94
        elif temp<=0: lam_mult*=0.95
        if hum>=85: lam_mult*=0.97
        info={"desc":main,"wind":wind,"rain":rain,"temp":temp,"hum":hum}
    if surface:
        s=surface.strip().lower()
        if "artificial" in s or "synthetic" in s or "turf" in s: lam_mult*=1.02
        elif "clay" in s or "dirt" in s: lam_mult*=0.98
    return lam_mult, card_mult, info

# ======================================================================
# LİG & TAKIM GÜÇLERİ
# ======================================================================

def extract_goal_series(matches: List[Dict], team_id: int) -> Tuple[List[int], List[int]]:
    gf,ga=[],[]
    for m in matches or []:
        if "goals" not in m: continue
        is_home=safe_int(m["teams"]["home"]["id"])==team_id
        gh,ga_ = safe_int(m["goals"]["home"]), safe_int(m["goals"]["away"])
        gf.append(gh if is_home else ga_); ga.append(ga_ if is_home else gh)
    return gf,ga

def league_games_played(api: ApiSports, league_id: int, season: int, team_id: int) -> Optional[int]:
    try:
        st=api.standings(league_id, season)
        table=None
        if isinstance(st,list) and st:
            cand=st[0]
            if isinstance(cand,dict) and cand.get("league",{}).get("standings"):
                table=cand["league"]["standings"][0]
        elif isinstance(st,dict) and st.get("league",{}).get("standings"):
            table=st["league"]["standings"][0]
        if table:
            for r in table:
                if safe_int(r.get("team",{}).get("id"))==team_id:
                    return safe_int(r.get("all",{}).get("played"))
    except Exception:
        return None
    return None

def compute_league_context(api: ApiSports, league_id: int, season: int) -> Optional[LeagueContext]:
    try:
        lst=api.fixtures_league_window(league_id, season, days=120) or []
        H,A=[],[]; hf_tot,ft_tot=0,0
        sum_hist: Dict[int,int]={}
        ht_draws=0; ht_count=0
        ft_draws=0; btts_n=0; tot_n=0
        for m in lst:
            st=m["fixture"]["status"]["short"]
            if st not in ("FT","AET","PEN"): continue
            gh,ga=safe_int(m["goals"]["home"]), safe_int(m["goals"]["away"])
            H.append(gh); A.append(ga)
            s=gh+ga; sum_hist[s]=sum_hist.get(s,0)+1
            if gh==ga: ft_draws+=1
            if gh>0 and ga>0: btts_n+=1
            tot_n+=1
            try:
                hf_h=safe_int(m.get("score",{}).get("halftime",{}).get("home"))
                hf_a=safe_int(m.get("score",{}).get("halftime",{}).get("away"))
                if hf_h>=0 and hf_a>=0:
                    hf_tot+=(hf_h+hf_a); ft_tot+=(gh+ga); ht_count+=1
                    if hf_h==hf_a: ht_draws+=1
            except Exception: pass
        if len(H)<40: return None
        mu_h=statistics.mean(H); mu_a=statistics.mean(A)
        mu_t=statistics.mean([h+a for h,a in zip(H,A)])
        var_t=statistics.pvariance([h+a for h,a in zip(H,A)])
        try:
            corr=float(np.corrcoef(np.array(H,dtype=float), np.array(A,dtype=float))[0,1])
            if math.isnan(corr): corr=0.05
        except Exception:
            corr=0.05
        hfa=max(0.0, mu_h-mu_a)
        fh_share=0.44
        if ft_tot>0: fh_share=clamp(hf_tot/ft_tot, 0.38, 0.50)
        tot=sum(sum_hist.values()) or 1
        b01=sum(sum_hist.get(k,0) for k in (0,1)) / tot
        b23=sum(sum_hist.get(k,0) for k in (2,3)) / tot
        b45=sum(sum_hist.get(k,0) for k in (4,5)) / tot
        b6p=sum(v for k,v in sum_hist.items() if k>=6) / tot
        ht_draw = (ht_draws / ht_count) if ht_count>0 else 0.35
        ft_draw = (ft_draws / tot_n) if tot_n>0 else 0.27
        btts_rate = (btts_n / tot_n) if tot_n>0 else 0.48
        return LeagueContext(mu_home=mu_h, mu_away=mu_a, mu_total=mu_t, var_total=var_t,
                             corr_HA=corr, hfa_goals=hfa, first_half_share=fh_share,
                             bin_01=b01, bin_23=b23, bin_45=b45, bin_6p=b6p,
                             ht_draw=ht_draw, ft_draw=ft_draw, btts_rate=btts_rate)
    except Exception:
        return None

# Rakip gücüne göre normalize
def elo_to_factor(elo: float, base: float = 1500.0, scale: float = 800.0) -> float:
    return 10.0 ** ((elo - base) / scale)

def adjust_goals_by_opponent_elo(goals: List[int], opponents_elo: List[float]) -> List[float]:
    out=[]
    for g,e in zip(goals, opponents_elo):
        f = elo_to_factor(e)
        adj = float(g) * (1.0 + 0.35*(f-1.0))
        out.append(max(0.0, adj))
    return out

def opponent_elos_for_matches(api: ApiSports, matches: List[Dict], team_id: int,
                              as_of_iso: Optional[str], league_id: Optional[int], season: Optional[int]) -> List[float]:
    elos=[]
    for m in matches:
        h_id=safe_int(m["teams"]["home"]["id"]); a_id=safe_int(m["teams"]["away"]["id"])
        opp_id = (a_id if h_id==team_id else h_id)
        elo = fallback_elo_via_statistics(api, opp_id, league_id, season) if (league_id and season) else 1500.0
        elos.append(elo)
    if not elos: elos=[1500.0]*len(matches)
    return elos

# ======================================================================
# ELO
# ======================================================================

def fixtures_aggregate_ppg(fixtures: List[Dict], team_id: int) -> Tuple[float,int]:
    pts=0; n=0
    for m in fixtures or []:
        st=m["fixture"]["status"]["short"]
        if st not in ("FT","AET","PEN"): continue
        n+=1
        is_home = safe_int(m["teams"]["home"]["id"])==team_id
        gh=safe_int(m["goals"]["home"]); ga=safe_int(m["goals"]["away"])
        res = (1 if (gh>ga and is_home) or (ga>gh and not is_home) else (0 if (gh<ga and is_home) or (ga<gh and not is_home) else 0.5))
        pts += 3*res
    return (pts/max(n,1)), n

def fallback_elo_via_statistics(api: ApiSports, team_id: int, league_id: int, season: int) -> float:
    ts=api.team_statistics(team_id, league_id, season) or {}
    try:
        fx=ts.get("fixtures",{})
        w=safe_int(fx.get("wins",{}).get("total"))
        d=safe_int(fx.get("draws",{}).get("total"))
        l=safe_int(fx.get("loses",{}).get("total"))
        n=w+d+l
        if n>0:
            ppg=(3*w+1*d)/n
            elo=1500.0 + 250.0*(ppg-1.33)
            return float(clamp(elo, 1200.0, 1800.0))
    except Exception:
        pass
    return 1500.0

def compute_elo_from_history(api: ApiSports, home_id: int, away_id: int,
                             as_of_iso: Optional[str] = None,
                             league_id: Optional[int] = None,
                             season: Optional[int] = None) -> Dict[int, float]:
    HFA_ELO=60.0; K_BASE=22.0; elos: Dict[int,float]={}
    if as_of_iso:
        dt_to=datetime.fromisoformat(as_of_iso.replace("Z","+00:00"))
        frm=(dt_to - timedelta(days=365)).strftime("%Y-%m-%d")
        allm=api.fixtures_team_between(home_id, frm, as_of_iso, 160) + \
             api.fixtures_team_between(away_id, frm, as_of_iso, 160)
    else:
        allm=api.team_recent(home_id, 80) + api.team_recent(away_id, 80)
    seen=set(); uniq=[]
    for m in allm or []:
        if "goals" not in m: continue
        key=m["fixture"].get("id")
        if key in seen: continue
        seen.add(key); uniq.append(m)
    uniq.sort(key=lambda x: x["fixture"].get("timestamp",0))
    for m in uniq:
        h=safe_int(m["teams"]["home"]["id"]); a=safe_int(m["teams"]["away"]["id"])
        gh=safe_int(m["goals"]["home"]); ga=safe_int(m["goals"]["away"])
        if gh<0 or ga<0: continue
        elos.setdefault(h,1500.0); elos.setdefault(a,1500.0)
        mov = math.log(abs(gh-ga)+1.0) * (2.2/(abs(elos[h]-elos[a])*0.001+2.2))
        eh=1.0/(1.0 + 10.0**(-(elos[h]+HFA_ELO - elos[a])/400.0))
        Ah = 1.0 if gh>ga else (0.5 if gh==ga else 0.0)
        K = K_BASE * (1.0 + 0.15*abs(gh-ga))
        elos[h]+=K*mov*(Ah-eh)
        elos[a]+=K*mov*((1.0-Ah)-(1.0-eh))
    for tid in (home_id, away_id):
        if tid not in elos or abs(elos[tid]-1500.0)<1.0:
            if league_id and season:
                try:
                    st=api.standings(league_id, season)
                    table=None
                    if isinstance(st,list) and st:
                        cand=st[0]
                        if isinstance(cand,dict) and cand.get("league",{}).get("standings"):
                            table=cand["league"]["standings"][0]
                    elif isinstance(st,dict) and st.get("league",{}).get("standings"):
                        table=st["league"]["standings"][0]
                    if table:
                        played=0; pts=0
                        for r in table:
                            if safe_int(r.get("team",{}).get("id"))==tid:
                                played=safe_int(r.get("all",{}).get("played")); pts=safe_int(r.get("points")); break
                        if played>0:
                            ppg=pts/played
                            elos[tid]=1500.0 + 250.0*(ppg-1.33)
                except Exception:
                    pass
    for tid in (home_id, away_id):
        if tid not in elos or abs(elos[tid]-1500.0)<1.0:
            if league_id and season:
                elos[tid]=fallback_elo_via_statistics(api, tid, league_id, season)
            else:
                elos[tid]=1500.0
    for k in list(elos.keys()):
        elos[k]=float(clamp(elos[k], 1200.0, 1850.0))
    return elos

def elo_probs(eh: float, ea: float) -> Tuple[float, float, float]:
    diff=eh-ea; ph=1.0/(1.0+10.0**(-diff/400.0)); pa=1.0-ph; px=0.22
    s=ph+px+pa; return ph/s, px/s, pa/s

# ======================================================================
# POISSON / NB / MATRİSLER / KALİBRASYON
# ======================================================================

def expected_goals_from_strengths(h: TeamStrength, a: TeamStrength, lg: Optional[LeagueContext]) -> Tuple[float,float]:
    atk_h=max(0.1,h.attack_g); atk_a=max(0.1,a.attack_g)
    def_h=max(0.1,h.defense_g); def_a=max(0.1,a.defense_g)
    if h.attack_xg is not None: atk_h=0.5*atk_h+0.5*h.attack_xg
    if a.attack_xg is not None: atk_a=0.5*atk_a+0.5*a.attack_xg
    if h.defense_xg is not None: def_h=0.5*def_h+0.5*h.defense_xg
    if a.defense_xg is not None: def_a=0.5*def_a+0.5*a.defense_xg
    def_factor_h=1.0/(1.0+def_h); def_factor_a=1.0/(1.0+def_a)
    lam_h=BASE_HOME_G*atk_h*def_factor_a*(1.0+HOME_FIELD_ADV)
    lam_a=BASE_AWAY_G*atk_a*def_factor_h
    if lg:
        scale_h=lg.mu_home/max(0.1,BASE_HOME_G); scale_a=lg.mu_away/max(0.1,BASE_AWAY_G)
        lam_h=0.6*lam_h + 0.4*(lam_h*scale_h)
        lam_a=0.6*lam_a + 0.4*(lam_a*scale_a)
    return clamp(lam_h,0.2,5.2), clamp(lam_a,0.1,5.0)

def apply_formation_adjust(lh: float, la: float, home_form: Optional[str], away_form: Optional[str]) -> Tuple[float,float]:
    def adj_for(f: Optional[str])->Tuple[float,float]:
        if not f: return 1.0,1.0
        k=f.strip().upper().replace(" ","").replace("–","-").replace("—","-")
        return FORMATION_ADJUST.get(k,(1.0,1.0))
    ha,hd=adj_for(home_form); aa,ad=adj_for(away_form)
    lh=clamp(lh*ha*(1.0/hd),0.15,5.5); la=clamp(la*aa*(1.0/ad),0.10,5.2); return lh,la

def bvpoisson_matrix(lh: float, la: float, rho: float, max_goals: int = 8) -> np.ndarray:
    rho=max(0.0,min(rho,min(lh,la)-1e-6)); lam1=max(1e-8, lh-rho); lam2=max(1e-8, la-rho)
    n=max_goals; P=np.zeros((n+1,n+1),dtype=float)
    pois1=[poisson.pmf(k,lam1) for k in range(n+1)]
    pois2=[poisson.pmf(k,lam2) for k in range(n+1)]
    poisR=[poisson.pmf(k,rho) for k in range(n+1)]
    for a in range(n+1):
        for b in range(n+1):
            s=0.0; m=min(a,b)
            for k in range(m+1): s+=pois1[a-k]*pois2[b-k]*poisR[k]
            P[a,b]=s
    P/=P.sum(); return P

def poisson_matrix(lh: float, la: float, max_goals: int = 8) -> np.ndarray:
    i=np.arange(0,max_goals+1); j=np.arange(0,max_goals+1)
    return np.outer(poisson.pmf(i,lh), poisson.pmf(j,la))

def markets_from_matrix(P: np.ndarray) -> Dict[str,float]:
    n=P.shape[0]
    p_home=float(np.tril(P,-1).sum()); p_draw=float(np.trace(P)); p_away=float(np.triu(P,1).sum())
    over15=float(sum(P[i,j] for i in range(n) for j in range(n) if (i+j)>=2))
    over25=float(sum(P[i,j] for i in range(n) for j in range(n) if (i+j)>=3))
    over35=float(sum(P[i,j] for i in range(n) for j in range(n) if (i+j)>=4))
    btts=float(sum(P[i,j] for i in range(n) for j in range(n) if (i>0 and j > 0)))
    flat=[((i,j), float(P[i,j])) for i in range(n) for j in range(n)]
    flat.sort(key=lambda x:x[1], reverse=True); top_scores=[(f"{i}-{j}", p) for (i,j),p in flat[:10]]
    r01=float(sum(P[i,j] for i in range(n) for j in range(n) if 0<=i+j<=1))
    r23=float(sum(P[i,j] for i in range(n) for j in range(n) if 2<=i+j<=3))
    r45=float(sum(P[i,j] for i in range(n) for j in range(n) if 4<=i+j<=5))
    r6p=1.0-(r01+r23+r45)
    return {"p_home":p_home,"p_draw":p_draw,"p_away":p_away,
            "over15":over15,"over25":over25,"over35":over35,"btts":btts,"top_scores":top_scores,
            "range_0_1":r01,"range_2_3":r23,"range_4_5":r45,"range_6p":r6p}

def dirichlet_smooth_triplet(p1: float, px: float, p2: float,
                             alpha: Tuple[float,float,float]=(1.15,1.05,1.15)) -> Tuple[float,float,float]:
    s=p1+px+p2
    if s<=0: return 1/3,1/3,1/3
    p1,px,p2=p1/s, px/s, p2/s
    a1,ax,a2=alpha; tot=a1+ax+a2
    p1=(p1+a1/tot)/2.0; px=(px+ax/tot)/2.0; p2=(p2+a2/tot)/2.0
    return clamp01(p1), clamp01(px), clamp01(p2)

def beta_smooth(p: float, alpha: float = 1.1, beta_: float = 1.1) -> float:
    p=clamp01(p); prior=alpha/(alpha+beta_)
    return clamp01(0.5*p + 0.5*prior)

def nb_total_probs(mu: float, var: float, max_sum: int = 16) -> List[float]:
    if var <= mu + 1e-9 or mu <= 0:
        return [float(poisson.pmf(k, mu)) for k in range(max_sum+1)]
    r = (mu*mu) / max(var - mu, 1e-6)
    p = r / (r + mu)
    from math import lgamma, log, exp
    out=[]
    for k in range(0, max_sum+1):
        logpmf = lgamma(k + r) - lgamma(r) - lgamma(k+1) + r*log(p) + k*log(1-p)
        out.append(float(exp(logpmf)))
    s=sum(out)
    return [x/s for x in out] if s>0 else out

def calibrate_totals_with_league(P: np.ndarray, lg: Optional[LeagueContext], alpha: float = 0.25) -> np.ndarray:
    if not lg: return P
    n=P.shape[0]; max_sum=2*n
    cur=[0.0]*(max_sum+1)
    for i in range(n):
        for j in range(n):
            cur[i+j]+=P[i,j]
    def alloc(bin_sums: List[int], bin_share: float) -> Dict[int,float]:
        tot=sum(cur[s] for s in bin_sums) or 1e-9
        return {s: bin_share * (cur[s]/tot) for s in bin_sums}
    tgt = {**alloc([0,1], lg.bin_01),
           **alloc([2,3], lg.bin_23),
           **alloc([4,5], lg.bin_45)}
    s6=[s for s in range(6, max_sum+1)]
    if s6:
        tot6=sum(cur[s] for s in s6) or 1e-9
        for s in s6: tgt[s]=lg.bin_6p * (cur[s]/tot6)
    new_sum=[(1-alpha)*cur[s] + alpha*tgt.get(s, cur[s]) for s in range(max_sum+1)]
    P2=np.zeros_like(P); eps=1e-12
    for i in range(n):
        for j in range(n):
            s=i+j; scale=(new_sum[s]/(cur[s]+eps))
            P2[i,j]=P[i,j]*scale
    P2=np.maximum(P2,0); P2/=P2.sum()
    return P2

def calibrate_totals_with_nb(P: np.ndarray, mu: float, var: float, alpha: float) -> np.ndarray:
    if alpha <= 1e-6: return P
    n=P.shape[0]; max_sum=2*n
    cur=[0.0]*(max_sum+1)
    for i in range(n):
        for j in range(n):
            cur[i+j]+=P[i,j]
    tgt=nb_total_probs(mu, var, max_sum=max_sum)
    new_sum=[(1-alpha)*cur[s] + alpha*tgt[s] for s in range(max_sum+1)]
    P2=np.zeros_like(P); eps=1e-12
    for i in range(n):
        for j in range(n):
            s=i+j; scale=(new_sum[s]/(cur[s]+eps))
            P2[i,j]=P[i,j]*scale
    P2=np.maximum(P2,0); P2/=P2.sum()
    return P2

def dixon_coles_small_score_adj(P: np.ndarray, xi: float = 0.07) -> np.ndarray:
    n=P.shape[0]; P2=P.copy()
    adj={(0,0): 1.0 + 0.50*xi,(1,0): 1.0 + 0.25*xi,(0,1): 1.0 + 0.25*xi,(1,1): 1.0 + 0.30*xi,(2,0): 1.0 - 0.10*xi,(0,2): 1.0 - 0.10*xi}
    for (i,j),m in adj.items():
        if i<=n-1 and j<=n-1: P2[i,j]*=m
    P2=np.maximum(P2,0); P2/=P2.sum()
    return P2

def zero_inflation_boost(P: np.ndarray, zeta: float = 0.04) -> np.ndarray:
    P2=P.copy(); P2[0,0] *= (1.0 + zeta); P2/=P2.sum(); return P2

# ----------------------------------------------------------------------
# YARDIMCI dönüşümler (kalibrasyon)
# ----------------------------------------------------------------------

def temp_scale_triplet(p1: float, px: float, p2: float, T: float) -> Tuple[float,float,float]:
    T = float(max(0.5, min(2.0, T)))
    v = np.array([clamp01(p1), clamp01(px), clamp01(p2)], dtype=float)
    v = np.power(np.maximum(v, 1e-12), 1.0/T)
    v = v / v.sum()
    return float(v[0]), float(v[1]), float(v[2])

def center_push(p: float, k: float) -> float:
    return clamp01(0.5 + (clamp01(p) - 0.5) * (1.0 + k))

# ======================================================================
# YARI ve HT/FT
# ======================================================================

def half_markets(lh_total: float, la_total: float, rho: float, lg: Optional[LeagueContext]) -> Dict[str, Any]:
    share=lg.first_half_share if lg else 0.44
    lh_ht=clamp(lh_total * share, 0.05, 2.5); la_ht=clamp(la_total * share, 0.05, 2.3)
    lh_2h=clamp(max(0.05, lh_total - lh_ht), 0.05, 3.5); la_2h=clamp(max(0.05, la_total - la_ht), 0.05, 3.2)
    rho_ht=clamp(0.6*rho, 0.01, 0.18); rho_2h=clamp(1.2*rho, 0.02, 0.30)
    P_ht=bvpoisson_matrix(lh_ht, la_ht, rho=rho_ht, max_goals=6)
    P_2h=bvpoisson_matrix(lh_2h, la_2h, rho=rho_2h, max_goals=6)
    mk_ht=markets_from_matrix(P_ht); mk_2h=markets_from_matrix(P_2h)
    ph,px,pa=mk_ht["p_home"], mk_ht["p_draw"], mk_ht["p_away"]
    if lg:
        px=(px + lg.ht_draw)/2.0
        remain=max(1e-9, 1.0 - px); ratio=ph/(ph+pa+1e-9); ph=remain*ratio; pa=remain*(1-ratio)
    ph,px,pa=dirichlet_smooth_triplet(ph,px,pa, alpha=(1.1,1.15,1.1))
    combos={"1/1":0.0,"1/X":0.0,"1/2":0.0,"X/1":0.0,"X/X":0.0,"X/2":0.0,"2/1":0.0,"2/X":0.0,"2/2":0.0}
    n=P_ht.shape[0]
    for i in range(n):
        for j in range(n):
            p_ht=P_ht[i,j]; 
            if p_ht<=0: continue
            for a in range(n):
                for b in range(n):
                    p_2h=P_2h[a,b]; 
                    if p_2h<=0: continue
                    ih,ia=i,j; fh,fa=i+a,j+b
                    if ih>ia and fh>fa: combos["1/1"]+=p_ht*p_2h
                    elif ih>ia and fh==fa: combos["1/X"]+=p_ht*p_2h
                    elif ih>ia and fh<fa: combos["1/2"]+=p_ht*p_2h
                    elif ih==ia and fh>fa: combos["X/1"]+=p_ht*p_2h
                    elif ih==ia and fh==fa: combos["X/X"]+=p_ht*p_2h
                    elif ih==ia and fh<fa: combos["X/2"]+=p_ht*p_2h
                    elif ih<ia and fh>fa: combos["2/1"]+=p_ht*p_2h
                    elif ih<ia and fh==fa: combos["2/X"]+=p_ht*p_2h
                    elif ih<ia and fh<fa: combos["2/2"]+=p_ht*p_2h
    tot=sum(combos.values()) or 1.0
    for k in list(combos.keys()): combos[k]=combos[k]/tot
    return {
        "HT": {"oneXtwo": {"home": ph, "draw": px, "away": pa},
               "over05": beta_smooth(float(1.0 - P_ht[0,0]), 1.1,1.1),
               "over15": beta_smooth(mk_ht["over15"], 1.1,1.1),
               "top_scores": mk_ht["top_scores"][:5]},
        "2H": {"goals_over05": beta_smooth(float(1.0 - P_2h[0,0]), 1.1,1.1),
               "goals_over15": beta_smooth(mk_2h["over15"], 1.1,1.1),
               "top_scores": mk_2h["top_scores"][:5]},
        "HTFT": combos
    }

# ======================================================================
# KORNER / KART
# ======================================================================

def expected_corners(h: TeamStrength, a: TeamStrength, lg: Optional[LeagueContext]) -> Tuple[float, float]:
    cf_h=h.corners_for if h.corners_for is not None else 5.2
    ca_h=h.corners_against if h.corners_against is not None else 4.8
    cf_a=a.corners_for if a.corners_for is not None else 4.8
    ca_a=a.corners_against if a.corners_against is not None else 5.2
    base=((cf_h+ca_a)/2.0)+((cf_a+ca_h)/2.0)
    if lg: base=0.6*base+0.4*(lg.mu_total*2.1)
    base=clamp(base,2.0,16.0)
    pmf=[poisson.pmf(k,base) for k in range(0,25)]
    p_over95=1.0 - sum(pmf[:10])
    return base, clamp01(p_over95)

def expected_cards(h: TeamStrength, a: TeamStrength, referee_bias: Optional[float]=None) -> Tuple[float, float]:
    hy=h.cards_yellow if h.cards_yellow is not None else 2.1
    ay=a.cards_yellow if a.cards_yellow is not None else 2.1
    base=clamp(0.5*(hy+ay)+1.6,1.0,7.5)
    if referee_bias is not None: base*=clamp(1.0+referee_bias,0.85,1.15)
    pmf=[poisson.pmf(k,base) for k in range(0,16)]
    p_over45=1.0 - sum(pmf[:5])
    return base, clamp01(p_over45)

# ======================================================================
# İLK 11 SENKRON
# ======================================================================

def latest_lineup_and_formation(api: ApiSports, team_id: int, before_iso: Optional[str]) -> Tuple[Optional[List[Dict]], Optional[str]]:
    if before_iso:
        dt_to=datetime.fromisoformat(before_iso.replace("Z","+00:00"))
        frm=(dt_to - timedelta(days=180)).strftime("%Y-%m-%d")
        matches=api.fixtures_team_between(team_id, frm, before_iso, limit=40)
    else:
        matches=api.team_recent(team_id, 6)
    matches=[m for m in matches or [] if m["fixture"]["status"]["short"] in ("FT","AET","PEN")]
    if not matches: return None, None
    last=matches[-1]; ln=(api.lineups(safe_int(last["fixture"]["id"])) or [])
    for it in ln or []:
        if safe_int(it.get("team",{}).get("id",-1))==team_id:
            formation=it.get("formation") or None
            players=[]
            for p in (it.get("startXI") or []):
                pp=p.get("player",{}); players.append({"name":pp.get("name",""), "pos":pp.get("pos","")})
            return (players if players else None), formation
    return None, None

def lineup_sync_factor(current_xi: List[Dict], last_xi: List[Dict]) -> float:
    if not current_xi or not last_xi: return 1.0
    cur=set(p.get("name","") for p in current_xi if p.get("name"))
    prev=set(p.get("name","") for p in last_xi if p.get("name"))
    diff=len(cur.symmetric_difference(prev))
    if diff>=6: return 0.94
    if diff>=4: return 0.96
    if diff>=3: return 0.98
    return 1.0

# ======================================================================
# TAKIM GÜCÜ
# ======================================================================

def compute_team_strength(api: ApiSports, team_id: int, league_id: int, season: int,
                          as_of_iso: Optional[str],
                          preset: RecencyPreset) -> TeamStrength:
    if as_of_iso:
        dt_to=datetime.fromisoformat(as_of_iso.replace("Z","+00:00"))
        frm=(dt_to - timedelta(days=365)).strftime("%Y-%m-%d")
        recent_all=api.fixtures_team_between(team_id, frm, as_of_iso, limit=160)
    else:
        recent_all=api.team_recent(team_id, max(RECENT_MATCHES_DEFAULT, preset.last_n))
    recent=[m for m in recent_all or [] if m["fixture"]["status"]["short"] in ("FT","AET","PEN")]
    if not recent:
        return TeamStrength(attack_g=1.0, defense_g=1.0)

    last_n = min(preset.last_n, len(recent))
    last_block = recent[-last_n:]

    opp_elos = opponent_elos_for_matches(api, last_block, team_id, as_of_iso, league_id, season)
    gf,ga = extract_goal_series(last_block, team_id)
    gf_adj = adjust_goals_by_opponent_elo(gf, opp_elos)
    ga_adj = adjust_goals_by_opponent_elo(ga, opp_elos)

    played = league_games_played(api, league_id, season, team_id) or 0
    if played <= 6 and preset.name != "Oto":
        half_life = max(1.5, preset.half_life * 0.9)
    elif played <= 6 and preset.name == "Oto":
        half_life = 3.0
        last_n = min(7, len(recent))
        last_block = recent[-last_n:]
        gf,ga = extract_goal_series(last_block, team_id)
        opp_elos = opponent_elos_for_matches(api, last_block, team_id, as_of_iso, league_id, season)
        gf_adj = adjust_goals_by_opponent_elo(gf, opp_elos)
        ga_adj = adjust_goals_by_opponent_elo(ga, opp_elos)
    else:
        half_life = preset.half_life

    last3 = last_block[-3:] if len(last_block)>=3 else last_block
    trend = 0.0
    if last3:
        gd=[]
        for m in last3:
            is_home = safe_int(m["teams"]["home"]["id"])==team_id
            gh=safe_int(m["goals"]["home"]); ga_=safe_int(m["goals"]["away"])
            gd.append((gh-ga_) if is_home else (ga_-gh))
        if gd:
            trend = clamp(sum(gd)/max(1,len(gd)), -2.0, 2.0)

    atk = time_weighted_avg(gf_adj, half_life) * (1.0 + 0.04*trend)
    dfn = time_weighted_avg(ga_adj, half_life) * (1.0 - 0.03*trend)

    out=TeamStrength(attack_g=max(0.1, atk), defense_g=max(0.1, dfn))

    if not as_of_iso:
        ts=api.team_statistics(team_id, league_id, season) or {}
        try:
            shots_total=ts.get("shots",{}).get("total",{}).get("total")
            shots_on=ts.get("shots",{}).get("on",{}).get("total")
            if shots_total is not None: out.shots=float(shots_total)
            if shots_on is not None: out.shots_on=float(shots_on)
            cf=ts.get("corners",{}).get("for",{}).get("average")
            ca=ts.get("corners",{}).get("against",{}).get("average")
            if cf is not None: out.corners_for=float(cf)
            if ca is not None: out.corners_against=float(ca)
            cy=ts.get("cards",{}).get("yellow",{}).get("average")
            if cy is not None: out.cards_yellow=float(cy)
            xg_for=ts.get("goals",{}).get("for",{}).get("average",{}).get("xg")
            xg_ag =ts.get("goals",{}).get("against",{}).get("average",{}).get("xg")
            if xg_for is not None: out.attack_xg=float(xg_for)
            if xg_ag  is not None: out.defense_xg=float(xg_ag)
        except Exception: pass
    return out

# ======================================================================
# MATRİS SERTLEŞTİRME + HİZALAMA
# ======================================================================

def _align_matrix_to_triplet(Q: np.ndarray, target: Tuple[float,float,float]) -> np.ndarray:
    th, td, ta = target
    n=Q.shape[0]; eps=1e-12
    sh=float(np.tril(Q,-1).sum()); sd=float(np.trace(Q)); sa=float(np.triu(Q,1).sum())
    scale_h = (th/max(sh,eps)) if sh>0 else 1.0
    scale_d = (td/max(sd,eps)) if sd>0 else 1.0
    scale_a = (ta/max(sa,eps)) if sa>0 else 1.0
    for i in range(n):
        for j in range(n):
            if i>j:   Q[i,j]*=scale_h
            elif i==j:Q[i,j]*=scale_d
            else:     Q[i,j]*=scale_a
    Q=np.maximum(Q,0); Q/=Q.sum()
    return Q

def _boost_favorite_peaks(Q: np.ndarray, fav_side: int, peak: float, low_tempo_hint: bool) -> np.ndarray:
    if peak<=1e-9: return Q
    n=Q.shape[0]; P=Q.copy()
    def boost(i,j,w):
        if 0<=i<n and 0<=j<n: P[i,j] *= (1.0 + peak * w)
    if fav_side > 0:
        boost(1,0,1.00); boost(2,1,0.80); boost(2,0,0.65); boost(3,1,0.50)
    elif fav_side < 0:
        boost(0,1,1.00); boost(1,2,0.80); boost(0,2,0.65); boost(1,3,0.50)
    else:
        if low_tempo_hint:
            boost(0,0,0.70); boost(1,0,0.15); boost(0,1,0.15)
        else:
            boost(1,1,0.40); boost(0,0,0.20)
    P/=P.sum(); return P

def sharpen_and_align_matrix(P: np.ndarray,
                             gamma: float,
                             target_triplet: Tuple[float,float,float],
                             fav_side: int,
                             low_tempo_hint: bool,
                             peak: float) -> np.ndarray:
    g=max(0.9, float(gamma))
    Q = np.power(np.maximum(P, 1e-15), g); Q/=Q.sum()
    Q = _boost_favorite_peaks(Q, fav_side=fav_side, peak=peak, low_tempo_hint=low_tempo_hint)
    Q = _align_matrix_to_triplet(Q, target_triplet)
    return Q

# ======================================================================
# TAHMİN — ENSEMBLE + KALİBRASYON UYGULAMASI
# ======================================================================

def predict_fixture(api: ApiSports, fixture: Dict,
                    lineup_policy: str = "D", allow_previous_lineup: bool = True,
                    backtest_asof_iso: Optional[str] = None, backtest_use_lineup: bool = True,
                    preset: RecencyPreset = RECENCY_PRESETS["Oto"],
                    sharpen: SharpenPreset = SHARPEN_PRESETS["Sert"]) -> Dict:
    h=fixture["teams"]["home"]; a=fixture["teams"]["away"]
    h_id,a_id=safe_int(h["id"]), safe_int(a["id"])
    league_id,season=safe_int(fixture["league"]["id"]), safe_int(fixture["league"]["season"])
    fixture_id=safe_int(fixture["fixture"]["id"]); fixture_iso=fixture["fixture"]["date"]
    referee_name=fixture["fixture"].get("referee") or "-"
    venue=fixture["fixture"].get("venue",{}) or {}
    venue_id=safe_int(venue.get("id")); venue_name=venue.get("name") or "-"
    venue_city=venue.get("city") or "-"; venue_country=fixture["league"].get("country") or "-"

    lg_ctx=compute_league_context(api, league_id, season)

    surface=None; capacity=None
    if venue_id:
        try:
            vresp=api.venues(venue_id)
            if isinstance(vresp,list) and vresp:
                vv=vresp[0]; surface=vv.get("surface"); capacity=vv.get("capacity")
        except Exception: pass

    has_lineup=False; home_form=None; away_form=None; home_xi=[]; away_xi=[]
    if backtest_asof_iso:
        if backtest_use_lineup:
            ln=api.lineups(fixture_id) or []
            for it in ln or []:
                if safe_int(it["team"]["id"])==h_id:
                    has_lineup=True; home_form=it.get("formation")
                    home_xi=[{"name":p["player"]["name"],"pos":p["player"].get("pos","")} for p in (it.get("startXI") or [])]
                if safe_int(it["team"]["id"])==a_id:
                    has_lineup=True; away_form=it.get("formation")
                    away_xi=[{"name":p["player"]["name"],"pos":p["player"].get("pos","")} for p in (it.get("startXI") or [])]
    else:
        if lineup_policy.upper()!="I":
            ln=api.lineups(fixture_id) or []
            for it in ln or []:
                if safe_int(it["team"]["id"])==h_id:
                    has_lineup=True; home_form=it.get("formation")
                    home_xi=[{"name":p["player"]["name"],"pos":p["player"].get("pos","")} for p in (it.get("startXI") or [])]
                if safe_int(it["team"]["id"])==a_id:
                    has_lineup=True; away_form=it.get("formation")
                    away_xi=[{"name":p["player"]["name"],"pos":p["player"].get("pos","")} for p in (it.get("startXI") or [])]

    last_h_xi, last_h_form = latest_lineup_and_formation(api, h_id, before_iso=backtest_asof_iso or fixture_iso)
    last_a_xi, last_a_form = latest_lineup_and_formation(api, a_id, before_iso=backtest_asof_iso or fixture_iso)

    Hs=compute_team_strength(api, h_id, league_id, season, backtest_asof_iso, preset)
    As=compute_team_strength(api, a_id, league_id, season, backtest_asof_iso, preset)

    lh,la=expected_goals_from_strengths(Hs,As,lg_ctx)
    if not has_lineup and allow_previous_lineup:
        home_form=home_form or last_h_form
        away_form=away_form or last_a_form
    lh,la=apply_formation_adjust(lh,la,home_form,away_form)

    lam_mult, card_mult, winfo = 1.0, 1.0, {}
    geo=owm_geocode_city(venue_city, venue_country)
    if geo:
        fc=owm_forecast_nearest(geo[0], geo[1], fixture_iso)
        lam_mult, card_mult, winfo = weather_adjustments(fc, surface)
    lh*=lam_mult; la*=lam_mult

    try:
        mv_home=squad_market_value(h["name"], season=season); mv_away=squad_market_value(a["name"], season=season)
    except Exception:
        mv_home=0.0; mv_away=0.0
    mv_mult=market_value_adjustments(mv_home, mv_away)

    elos=compute_elo_from_history(api, h_id, a_id, as_of_iso=backtest_asof_iso, league_id=league_id, season=season)
    elo_h=elos.get(h_id,1500.0); elo_a=elos.get(a_id,1500.0)
    elo_gap = elo_h - elo_a

    dominance = 1.0/(1.0 + math.exp(- (elo_gap/160.0 + math.log((mv_home+1)/(mv_away+1))*0.35)))
    scale = 1.0 + 0.10*(dominance-0.5)
    lh *= (scale*mv_mult); la /= (scale*mv_mult)

    if has_lineup and last_h_xi and last_a_xi:
        sync_h = lineup_sync_factor(home_xi, last_h_xi)
        sync_a = lineup_sync_factor(away_xi, last_a_xi)
        lh *= sync_h; la *= sync_a

    rho0=0.10
    if lg_ctx: rho0=clamp(lg_ctx.corr_HA*0.20, 0.02, 0.25)
    rho0 *= (1.0 - clamp(abs(elo_gap)/800.0, 0.0, 0.35))

    xg_h = Hs.attack_xg or Hs.attack_g
    xg_a = As.attack_xg or As.attack_g
    form_var = np.var([Hs.attack_g, Hs.defense_g, As.attack_g, As.defense_g])
    p_open = clamp(0.20 + 0.22*(xg_h + xg_a - 2.3) + 0.10*form_var + 0.10*abs(elo_gap)/400.0, 0.08, 0.52)

    lh_lo, la_lo = 0.92*lh, 0.92*la
    lh_hi, la_hi = 1.15*lh, 1.15*la
    P_low  = bvpoisson_matrix(lh_lo, la_lo, rho=rho0*1.05, max_goals=8)
    P_high = bvpoisson_matrix(lh_hi, la_hi, rho=rho0*0.95, max_goals=8)
    P = (1.0-p_open)*P_low + p_open*P_high
    P/=P.sum()

    if lg_ctx:
        P=calibrate_totals_with_league(P, lg_ctx, alpha=0.25)

    mu_tot=(lh+la)*( (1.0-p_open)*0.92 + p_open*1.15 )
    if lg_ctx:
        overdisp=max(0.0, lg_ctx.var_total - lg_ctx.mu_total)
        target_var = max(mu_tot + 0.20, mu_tot + overdisp*0.85)
        alpha_nb = clamp(0.22 * max(0.0, target_var - mu_tot) / max(mu_tot,1e-6), 0.0, 0.40)
        P=calibrate_totals_with_nb(P, mu=mu_tot, var=target_var, alpha=alpha_nb)

    P=dixon_coles_small_score_adj(P, xi=0.07)
    if p_open < 0.15:
        P=zero_inflation_boost(P, zeta=clamp(0.06*(0.15-p_open)/0.15, 0.01, 0.06))

    mk_base=markets_from_matrix(P)
    Pind=poisson_matrix(lh, la, max_goals=8)
    mk_ind=markets_from_matrix(Pind)

    w_elo = clamp(0.08 + (abs(elo_gap)/500.0)*0.15, 0.08, 0.20)
    w_ind = 0.20
    w_bv  = clamp(1.0 - (w_elo + w_ind), 0.62, 0.80)
    ep1,epx,ep2=elo_probs(elo_h, elo_a)

    p1 = w_bv*mk_base["p_home"] + w_ind*(mk_ind["p_home"]) + w_elo*ep1
    px = w_bv*mk_base["p_draw"] + w_ind*(mk_ind["p_draw"]) + w_elo*epx
    p2 = w_bv*mk_base["p_away"] + w_ind*(mk_ind["p_away"]) + w_elo*ep2

    if lg_ctx:
        px=(px + lg_ctx.ft_draw)/2.0
        remain=max(1e-9, 1.0 - px); ratio=p1/(p1+p2+1e-9)
        p1=remain*ratio; p2=remain*(1-ratio)
    p1,px,p2=dirichlet_smooth_triplet(p1,px,p2)

    # ---- KALİBRASYON (diske kaydedilmişse uygula)
    if CALIB.get("oneXtwo_T",1.0)!=1.0:
        p1,px,p2 = temp_scale_triplet(p1,px,p2, CALIB["oneXtwo_T"])

    fav_side = (1 if p1>max(px,p2) else (-1 if p2>max(p1,px) else 0))
    target_triplet = (p1, px, p2)
    P_sharp = sharpen_and_align_matrix(
        P, gamma=sharpen.gamma,
        target_triplet=target_triplet,
        fav_side=fav_side,
        low_tempo_hint=(p_open<0.15 or mu_tot<2.0),
        peak=sharpen.peak
    )

    # Skor kalibrasyonu (global güç)
    if CALIB.get("score_gamma",1.0)!=1.0:
        P_sharp = np.power(np.maximum(P_sharp, 1e-15), CALIB["score_gamma"])
        P_sharp /= P_sharp.sum()

    mk = markets_from_matrix(P_sharp)

    probs=np.array([p1,px,p2]); margin=float(probs.max()-probs.min())
    entropy=float(-(probs*np.log(probs+1e-12)).sum())
    conf = clamp(0.5*margin + 0.5*(1.1 - entropy), 0.0, 1.0)

    blend_w = 0.25 if sharpen.gamma>1.0 else 0.35
    o15_raw = blend_w*mk_ind["over15"] + (1-blend_w)*mk["over15"]
    o25_raw = blend_w*mk_ind["over25"] + (1-blend_w)*mk["over25"]
    o35_raw = blend_w*mk_ind["over35"] + (1-blend_w)*mk["over35"]
    btts_raw= blend_w*mk_ind["btts"]   + (1-blend_w)*mk["btts"]

    def push_from_center(p: float, k: float) -> float:
        return clamp01(0.5 + (p - 0.5)*(1.0 + k*conf))
    if sharpen.ou_push>0:
        o15_raw = push_from_center(o15_raw, sharpen.ou_push*0.6)
        o25_raw = push_from_center(o25_raw, sharpen.ou_push*1.0)
        o35_raw = push_from_center(o35_raw, sharpen.ou_push*1.2)
        btts_raw= push_from_center(btts_raw, sharpen.ou_push*0.8)

    # Kalibrasyon push (global)
    if abs(CALIB.get("over_push",0.0))>1e-9:
        k=CALIB["over_push"]
        o15_raw = center_push(o15_raw, k*0.8)
        o25_raw = center_push(o25_raw, k*1.0)
        o35_raw = center_push(o35_raw, k*1.2)
    if abs(CALIB.get("btts_push",0.0))>1e-9:
        btts_raw = center_push(btts_raw, CALIB["btts_push"])

    def confident_smooth(p: float, conf: float, sharp_gamma: float,
                         min_w: float = 0.08, max_w: float = 0.45) -> float:
        sharp_strength = clamp((sharp_gamma - 1.0)/0.25, 0.0, 1.0)
        w = clamp(max_w - 0.30*conf - 0.25*sharp_strength, min_w, max_w)
        return clamp01((1.0 - w)*p + w*0.5)

    o15=confident_smooth(o15_raw, conf, sharp_gamma=sharpen.gamma)
    o25=confident_smooth(o25_raw, conf, sharp_gamma=sharpen.gamma)
    o35=confident_smooth(o35_raw, conf, sharp_gamma=sharpen.gamma)
    btts=confident_smooth(btts_raw, conf, sharp_gamma=sharpen.gamma)
    if lg_ctx:
        btts=(btts + lg_ctx.btts_rate)/2.0

    halves=half_markets(lh, la, rho=rho0, lg=lg_ctx)

    exp_corners,p_corners_95=expected_corners(Hs,As,lg_ctx)
    exp_cards,p_cards_45=expected_cards(Hs,As,referee_bias=0.0)

    h2h_sum={"home_w":0,"away_w":0,"draw":0,"avg_g":None}
    try:
        h2h=api.h2h_last5(h_id,a_id)
        if h2h:
            gls=[]
            for m in h2h:
                gh=safe_int(m["goals"]["home"]); ga=safe_int(m["goals"]["away"])
                if gh>ga: h2h_sum["home_w"]+=1
                elif gh<ga: h2h_sum["away_w"]+=1
                else: h2h_sum["draw"]+=1
                gls.append(gh+ga)
            if gls: h2h_sum["avg_g"]=sum(gls)/len(gls)
    except Exception: pass

    top_scores = mk["top_scores"]; top3 = top_scores[:3]

    eligible=True; used_prev_lineup=False
    if not backtest_asof_iso:
        lp=lineup_policy.upper()[:1]
        if lp=="S":
            try: mt=(datetime.fromisoformat(fixture_iso.replace("Z","+00:00")) - now_utc()).total_seconds()/60.0
            except Exception: mt=None
            if (mt is not None and mt > LINEUP_STRICT_WINDOW_MIN) or (not has_lineup and not allow_previous_lineup):
                eligible=False
        elif lp=="D" and (not has_lineup) and (not allow_previous_lineup):
            alpha=0.08; p1=0.5*alpha+(1-alpha)*p1; px=0.5*alpha+(1-alpha)*px; p2=0.5*alpha+(1-alpha)*p2
            s=p1+px+p2; p1,px,p2=p1/s,px/s,p2/s
        if (not has_lineup) and allow_previous_lineup: used_prev_lineup=True

    return {
        "inputs":{"home":h["name"],"away":a["name"],"fixture_id":fixture_id},
        "kickoff_tr": fmt_tr_hour(fixture_iso),
        "referee": referee_name or "-",
        "venue":{"name":venue_name,"city":venue_city,"country":venue_country,"surface":surface,"capacity":capacity},
        "weather":winfo,
        "lambdas":{"home":lh,"away":la,"rho":rho0},
        "oneXtwo":{"home":p1,"draw":px,"away":p2,"confidence":conf},
        "goals":{
            "over_1_5":o15, "under_1_5":1.0-o15,
            "over_2_5":o25, "under_2_5":1.0-o25,
            "over_3_5":o35, "under_3_5":1.0-o35,
            "btts_yes":btts, "btts_no":1.0-btts,
            "ranges":{"0-1":mk["range_0_1"],"2-3":mk["range_2_3"],"4-5":mk["range_4_5"],"6+":mk["range_6p"]},
            "top_scores":top_scores, "top3":top3
        },
        "halves":halves,
        "corners":{"expected_total":exp_corners,"p_over_9_5":p_corners_95},
        "cards":{"expected_total":exp_cards,"p_over_4_5":p_cards_45},
        "eligible":eligible,
        "has_lineup":has_lineup,
        "used_previous_lineup":used_prev_lineup,
        "formations":{"home":home_form,"away":away_form},
        "lineups":{"home":home_xi,"away":away_xi},
        "elo":{"home":elo_h,"away":elo_a},
        "league_ctx": (lg_ctx.__dict__ if lg_ctx else None),
        "market_values":{"home_eur":mv_home,"away_eur":mv_away},
        "h2h":h2h_sum,
        "sharp_mode": sharpen.name,
        "score_matrix": P_sharp
    }

# ======================================================================
# ÖNERİLER
# ======================================================================

def build_recommendations(pred: Dict) -> Tuple[List[str], List[str]]:
    banko, normal = [], []
    cand=[
        (f"{pred['inputs']['home']} kazanır", pred["oneXtwo"]["home"]),
        ("Beraberlik", pred["oneXtwo"]["draw"]),
        (f"{pred['inputs']['away']} kazanır", pred["oneXtwo"]["away"]),
        ("1.5 Üst", pred["goals"]["over_1_5"]),
        ("2.5 Üst", pred["goals"]["over_2_5"]),
        ("3.5 Üst", pred["goals"]["over_3_5"]),
        ("2.5 Alt", pred["goals"]["under_2_5"]),
        ("KG Var", pred["goals"]["btts_yes"]),
        ("İY 0.5 Üst", pred["halves"]["HT"]["over05"]),
        ("İY 1.5 Üst", pred["halves"]["HT"]["over15"])
    ]
    for name,p in cand:
        if p>=BANKO_P: banko.append(f"● {name}  ({pct(p)})")
        elif p>=NORMAL_P: normal.append(f"● {name}  ({pct(p)})")
    return banko, normal

# ======================================================================
# BACKTEST — Ek Metrikler & Kalibrasyon Araçları
# ======================================================================

def logloss(prob: float) -> float:
    return float(-math.log(max(prob, 1e-12)))

def brier_binary(p: float, y: int) -> float:
    p=clamp01(p); return (p-y)**2

def brier_1x2(p1: float, px: float, p2: float, outcome: str) -> float:
    y = np.array([1.0 if outcome=="1" else 0.0,
                  1.0 if outcome=="X" else 0.0,
                  1.0 if outcome=="2" else 0.0])
    p = np.array([clamp01(p1), clamp01(px), clamp01(p2)])
    return float(np.mean((p - y)**2))

def auc_binary(probs: List[float], labels: List[int]) -> Optional[float]:
    # Rank-based AUC (Mann–Whitney U)
    n=len(probs)
    if n==0: return None
    P=sum(labels); N=n-P
    if P==0 or N==0: return None
    # rank with ties (average rank)
    idx=sorted(range(n), key=lambda i: probs[i])
    ranks=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and probs[idx[j+1]]==probs[idx[i]]:
            j+=1
        avg=(i+j+2)/2.0
        for k in range(i,j+1): ranks[idx[k]]=avg
        i=j+1
    sum_pos=sum(ranks[i] for i in range(n) if labels[i]==1)
    auc=(sum_pos - P*(P+1)/2.0)/(P*N)
    return float(auc)

def reliability_bins(probs: List[float], hits: List[int],
                     edges: List[float] = [0.50,0.60,0.70,0.80,0.90,1.01]) -> List[Dict[str,Any]]:
    out=[]
    for a,b in zip(edges[:-1], edges[1:]):
        items=[(p,h) for p,h in zip(probs,hits) if a<=p<b]
        m=len(items)
        acc=(sum(h for _,h in items)/m) if m>0 else None
        avgp=(sum(p for p,_ in items)/m) if m>0 else None
        out.append({"range":f"[{a:.2f},{b:.2f})", "n":m, "avg_p":avgp, "acc":acc})
    return out

def optimize_temperature_1x2(triples: List[Tuple[float,float,float,str]]) -> Tuple[float,float,float]:
    if not triples: return 1.0, float("nan"), float("nan")
    def avg_ll(T: float) -> float:
        s=0.0
        for p1,px,p2,outcome in triples:
            q1,qx,q2 = temp_scale_triplet(p1,px,p2, T)
            s += logloss({"1":q1,"X":qx,"2":q2}[outcome])
        return s/len(triples)
    base = avg_ll(1.0)
    grid=[round(x,2) for x in np.linspace(0.70,1.50,41)]
    best=min(grid, key=avg_ll)
    return best, base, avg_ll(best)

def optimize_center_push_binary(probs: List[float], labels: List[int]) -> Tuple[float,float,float]:
    if not probs: return 0.0, float("nan"), float("nan")
    def avg_ll(k: float) -> float:
        s=0.0
        for p,y in zip(probs, labels):
            q=center_push(p, k)
            s += logloss(q if y==1 else 1.0-q)
        return s/len(probs)
    base=avg_ll(0.0)
    grid=[round(x,2) for x in np.linspace(-0.40,0.40,41)]
    best=min(grid, key=avg_ll)
    return best, base, avg_ll(best)

def optimize_gamma_for_scores(score_data: List[Tuple[np.ndarray,int,int]]) -> Tuple[float,float,float]:
    valid=[(P,gh,ga) for (P,gh,ga) in score_data if isinstance(P,np.ndarray) and 0<=gh<=8 and 0<=ga<=8]
    if not valid: return 1.0, float("nan"), float("nan")
    def avg_ll(g: float) -> float:
        s=0.0
        for P,gh,ga in valid:
            Q=np.power(np.maximum(P,1e-15), g); Q/=Q.sum()
            s += logloss(float(Q[gh,ga]))
        return s/len(valid)
    base=avg_ll(1.0)
    grid=[round(x,2) for x in np.linspace(0.90,1.30,41)]
    best=min(grid, key=avg_ll)
    return best, base, avg_ll(best)

# ======================================================================
# GUI — Detay Penceresi
# ======================================================================

class BacktestDetailWindow(tk.Toplevel):
    def __init__(self, parent, matches: List[Dict[str,Any]], league_summary: List[Dict[str,Any]],
                 sweep_ou: List[Dict[str,Any]], sweep_btts: List[Dict[str,Any]],
                 reliability: List[Dict[str,Any]], out_dir: Optional[str]):
        super().__init__(parent)
        self.title("Backtest Detayları")
        self.geometry("1320x760")
        self.out_dir = out_dir

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # ---- Maçlar sekmesi
        f1=ttk.Frame(nb); nb.add(f1, text="Maçlar (Detay)")
        cols=["Tarih","Lig","Ev","Dep","FT","p1","px","p2","Pick","Hit",
              "pO2.5","OU","OU Hit","pBTTS","KG","KG Hit",
              "Top1","Top3?","Aralık","Aralık Hit","ScoreProb","λH","λA","Güven"]
        tree=ttk.Treeview(f1, columns=cols, show="headings", height=26)
        for c,w in zip(cols,[110,200,160,160,60,60,60,60,60,50,65,70,60,65,60,60,90,60,80,90,70,60,60]):
            tree.heading(c, text=c); tree.column(c, width=w, anchor="center")
        vsb=ttk.Scrollbar(f1, orient="vertical", command=tree.yview); tree.configure(yscroll=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        for r in matches:
            tree.insert("", "end", values=[
                r.get("kickoff","")[:16].replace("T"," "), r.get("league",""),
                r.get("home",""), r.get("away",""), r.get("FT",""),
                f"{r.get('p1',0):.2f}", f"{r.get('px',0):.2f}", f"{r.get('p2',0):.2f}",
                r.get("pick_1x2",""), "✓" if r.get("hit_1x2",0)==1 else "×",
                f"{r.get('p_over25',0):.2f}", r.get("ou_pick",""), "" if r.get("ou_pick","SKIP")=="SKIP" else ("✓" if r.get("ou_hit",0)==1 else "×"),
                f"{r.get('p_btts',0):.2f}", r.get("btts_pick",""), "" if r.get("btts_pick","SKIP")=="SKIP" else ("✓" if r.get("btts_hit",0)==1 else "×"),
                r.get("top1","-"), "✓" if r.get("top3_hit",0)==1 else "×",
                r.get("range_pick","-"), "✓" if r.get("range_hit",0)==1 else "×",
                ("" if r.get("score_prob") in ("",None) else f"{float(r.get('score_prob')):.4f}"),
                f"{r.get('lam_h',0):.2f}", f"{r.get('lam_a',0):.2f}", f"{r.get('conf',0):.2f}"
            ])

        btnf=ttk.Frame(f1); btnf.pack(fill=tk.X)
        def export_matches():
            path=filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
            if not path: return
            try:
                with open(path,"w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(cols)
                    for r in matches:
                        w.writerow([
                            r.get("kickoff",""), r.get("league",""), r.get("home",""), r.get("away",""), r.get("FT",""),
                            f"{r.get('p1',0):.4f}", f"{r.get('px',0):.4f}", f"{r.get('p2',0):.4f}",
                            r.get("pick_1x2",""), r.get("hit_1x2",0),
                            f"{r.get('p_over25',0):.4f}", r.get("ou_pick",""), r.get("ou_hit",""),
                            f"{r.get('p_btts',0):.4f}", r.get("btts_pick",""), r.get("btts_hit",""),
                            r.get("top1","-"), r.get("top3_hit",0), r.get("range_pick","-"), r.get("range_hit",0),
                            r.get("score_prob",""), f"{r.get('lam_h',0):.2f}", f"{r.get('lam_a',0):.2f}", f"{r.get('conf',0):.2f}"
                        ])
                messagebox.showinfo("Dışa aktar", f"Kaydedildi:\n{path}")
            except Exception as e:
                messagebox.showerror("Dışa aktar", str(e))
        ttk.Button(btnf, text="CSV dışa aktar", command=export_matches).pack(side=tk.LEFT, padx=6, pady=6)

        def open_folder():
            if not self.out_dir: 
                messagebox.showinfo("Klasör", "Klasör bilgisi bulunamadı."); return
            try:
                if os.name == "nt":
                    os.startfile(self.out_dir)
                elif sys.platform == "darwin":
                    subprocess.call(["open", self.out_dir])
                else:
                    subprocess.call(["xdg-open", self.out_dir])
            except Exception as e:
                messagebox.showerror("Klasör", str(e))
        ttk.Button(btnf, text="Klasörü Aç", command=open_folder).pack(side=tk.LEFT, padx=6, pady=6)

        # ---- Lig özeti sekmesi
        f2=ttk.Frame(nb); nb.add(f2, text="Lig Özeti")
        cols2=["Lig","Maç","MS%","OU Kap.","OU%","KG Kap.","KG%"]
        tree2=ttk.Treeview(f2, columns=cols2, show="headings", height=26)
        for c,w in zip(cols2,[280,70,70,80,70,80,70]):
            tree2.heading(c, text=c); tree2.column(c, width=w, anchor="center")
        vsb2=ttk.Scrollbar(f2, orient="vertical", command=tree2.yview); tree2.configure(yscroll=vsb2.set)
        tree2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb2.pack(side=tk.RIGHT, fill=tk.Y)
        for s in league_summary:
            tree2.insert("", "end", values=[
                s["league"], s["n"],
                f"{100*s['ms_acc']:.1f}%",
                f"{100*s['ou_cov']:.1f}%", f"{100*s['ou_acc']:.1f}%",
                f"{100*s['btts_cov']:.1f}%", f"{100*s['btts_acc']:.1f}%"
            ])

        # ---- Eşik & Güvenilirlik sekmesi
        f3=ttk.Frame(nb); nb.add(f3, text="Eşik & Güvenilirlik")
        txt=tk.Text(f3, wrap="word", font=("Consolas",10)); txt.pack(fill=tk.BOTH, expand=True)

        def format_sweep(arr, title):
            lines=[title]
            lines.append("   Thr    OU(Kap-%)  OU(İsab-%)    |    Thr    KG(Kap-%)  KG(İsab-%)")
            return "\n".join(lines)

        txt.insert(tk.END, "[Eşik Taraması]\n")
        txt.insert(tk.END, "   Thr    OU(Kap-%)  OU(İsab-%)    |    Thr    KG(Kap-%)  KG(İsab-%)\n")
        for so, sb in zip(sweep_ou, sweep_btts):
            txt.insert(tk.END, f"  {so['thr']:.2f}     {so['cov']*100:6.1f}%    {so['acc']*100:6.1f}%   |"
                                f"   {sb['thr']:.2f}     {sb['cov']*100:6.1f}%    {sb['acc']*100:6.1f}%\n")
        txt.insert(tk.END, "\n[Güvenilirlik (MS 1X2; seçilen sınıf olasılığına göre)]\n")
        txt.insert(tk.END, "   Kov   N    Ort.P    Gerçek İsabet\n")
        for r in reliability:
            ap = "-" if r["avg_p"] is None else f"{100*r['avg_p']:.1f}%"
            ac = "-" if r["acc"] is None else f"{100*r['acc']:.1f}%"
            txt.insert(tk.END, f"  {r['range']:>9}  {r['n']:>3}   {ap:>7}   {ac:>8}\n")

# ======================================================================
# GUI
# ======================================================================

class App(tk.Tk):
    def __init__(self, api: ApiSports):
        super().__init__()
        self.api = api
        self.title("Gelişmiş Futbol Tahmin Motoru — GUI + Toplu Backtest")
        self.geometry("1320x860")
        self.minsize(1180, 760)

        style=ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=6)
        style.configure("Treeview", rowheight=24)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9))
        style.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"))

        # Üst bar
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="Tarih (GG.AA.YYYY):", style="Bold.TLabel").pack(side=tk.LEFT)
        self.date_var = tk.StringVar(master=self, value=datetime.now().strftime("%d.%m.%Y"))
        ttk.Entry(top, textvariable=self.date_var, width=12).pack(side=tk.LEFT, padx=6)

        self.backtest_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(top, text="Backtest (bitmiş maçlar)", variable=self.backtest_var).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Lig Filtresi:", style="Bold.TLabel").pack(side=tk.LEFT, padx=(10,0))
        self.filter_mode = tk.StringVar(master=self, value="Hepsi")
        ttk.Combobox(top, textvariable=self.filter_mode, values=["Hepsi","Ana Ligler"], width=12, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Ara:", style="Bold.TLabel").pack(side=tk.LEFT, padx=(10,0))
        self.search_var = tk.StringVar(master=self)
        ttk.Entry(top, textvariable=self.search_var, width=20).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Form Penceresi:", style="Bold.TLabel").pack(side=tk.LEFT, padx=(12,0))
        self.recency_var = tk.StringVar(master=self, value="Oto")
        ttk.Combobox(top, textvariable=self.recency_var,
                     values=list(RECENCY_PRESETS.keys()), width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        # Sertlik modu
        ttk.Label(top, text="Sertlik:", style="Bold.TLabel").pack(side=tk.LEFT, padx=(12,0))
        self.sharp_var = tk.StringVar(master=self, value="Sert")
        ttk.Combobox(top, textvariable=self.sharp_var,
                     values=list(SHARPEN_PRESETS.keys()), width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text="Listele", command=self.list_fixtures).pack(side=tk.LEFT, padx=8)

        polf = ttk.Frame(top); polf.pack(side=tk.RIGHT)
        ttk.Label(polf, text="Kadro Politikası:", style="Bold.TLabel").grid(row=0, column=0, padx=(0,6))
        self.policy_var = tk.StringVar(master=self, value="D")
        for i,(k,lab) in enumerate([("S","Strict (60dk)"),("D","Degrade"),("I","Ignore")]):
            ttk.Radiobutton(polf, text=lab, value=k, variable=self.policy_var).grid(row=0, column=i+1, padx=2)
        self.use_prev_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(polf, text="Son kadroyu baz al", variable=self.use_prev_var).grid(row=0, column=4, padx=(10,0))

        # Paned
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # Sol liste
        left = ttk.Frame(paned); paned.add(left, weight=1)
        ttk.Label(left, text="Fikstür Listesi", style="Header.TLabel").pack(anchor="w", pady=(0,4))
        cols=("Lig","Ev","Dep","Saat")
        self.tree=ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for c,w in zip(cols,(280,260,260,80)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, anchor="w")
        vsb=ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        btnf=ttk.Frame(left); btnf.pack(fill=tk.X, pady=6)
        ttk.Button(btnf, text="Analiz", command=self.analyze_selected).pack(side=tk.LEFT)
        ttk.Button(btnf, text="Toplu Backtest...", command=self.open_backtest_dialog).pack(side=tk.LEFT, padx=6)
        ttk.Button(btnf, text="Temizle", command=lambda: self.txt.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=6)

        # Sağ analiz
        right = ttk.Frame(paned); paned.add(right, weight=2)
        ttk.Label(right, text="Analiz / Backtest Çıktısı", style="Header.TLabel").pack(anchor="w", pady=(0,4))
        self.txt = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.txt.pack(fill=tk.BOTH, expand=True)
        self._config_text_tags()

        foot = ttk.Frame(right); foot.pack(fill=tk.X, pady=6)
        ttk.Button(foot, text="Analizi Kaydet...", command=self.save_report).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(master=self, value="Hazır.")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.raw_fixtures: List[Dict] = []
        self.list_fixtures(initial=True)

    def _config_text_tags(self):
        t=self.txt
        t.tag_configure("title", font=("Consolas", 12, "bold"))
        t.tag_configure("h", foreground="#0a84ff", font=("Consolas", 11, "bold"))
        t.tag_configure("ok", foreground="#0aa80a", font=("Consolas", 10, "bold"))
        t.tag_configure("warn", foreground="#e6a700", font=("Consolas", 10, "bold"))
        t.tag_configure("bad", foreground="#d91e18", font=("Consolas", 10, "bold"))
        t.tag_configure("bold", font=("Consolas", 10, "bold"))
        t.tag_configure("mono", font=("Consolas", 10))
        t.tag_configure("small", font=("Consolas", 9), foreground="#777777")

    def set_status(self, msg: str):
        self.status_var.set(msg); self.update_idletasks()

    # -------------------- Listeleme --------------------

    def list_fixtures(self, initial: bool=False):
        iso=parse_tr_date(self.date_var.get().strip())
        if not iso:
            messagebox.showwarning("Tarih", "Tarih formatı hatalı. GG.AA.YYYY giriniz.")
            return
        self.set_status("Fikstür çekiliyor...")
        try:
            fixtures=self.api.fixtures_by_date_smart(iso) or []
        except Exception:
            fixtures=[]
        if self.backtest_var.get():
            fixtures=[m for m in fixtures if m["fixture"]["status"]["short"] in ("FT","AET","PEN")]
        if self.filter_mode.get()=="Ana Ligler":
            fixtures=[m for m in fixtures if m["league"]["name"] in MAJOR_LEAGUES]
        q=self.search_var.get().strip().lower()
        if q:
            fx=[]; 
            for m in fixtures:
                lig=m["league"]["name"].lower(); h=m["teams"]["home"]["name"].lower(); a=m["teams"]["away"]["name"].lower()
                if q in lig or q in h or q in a: fx.append(m)
            fixtures=fx
        self.raw_fixtures=fixtures
        for i in self.tree.get_children(): self.tree.delete(i)
        for idx,m in enumerate(fixtures, start=1):
            lig=m["league"]["name"]; home=m["teams"]["home"]["name"]; away=m["teams"]["away"]["name"]; hour=fmt_tr_hour(m["fixture"]["date"])
            self.tree.insert("", "end", iid=str(idx-1), values=(lig,home,away,hour))
        self.set_status(f"{iso} — {len(fixtures)} maç listelendi.")
        if initial and len(fixtures)==0:
            self.txt.insert(tk.END, "Bu tarih için maç bulunamadı.\n")

    def get_selected_fixture(self) -> Optional[Dict]:
        sel=self.tree.selection()
        if not sel: messagebox.showinfo("Seçim", "Lütfen listeden bir maç seçin."); return None
        idx=int(sel[0]); 
        if 0<=idx<len(self.raw_fixtures): return self.raw_fixtures[idx]
        return None

    # -------------------- Analiz --------------------

    def analyze_selected(self):
        fx=self.get_selected_fixture()
        if not fx: return
        pol=self.policy_var.get()
        allow_prev=self.use_prev_var.get()
        is_back=self.backtest_var.get()
        kickoff_iso=fx["fixture"]["date"]
        preset = RECENCY_PRESETS.get(self.recency_var.get(), RECENCY_PRESETS["Oto"])
        sharp  = SHARPEN_PRESETS.get(self.sharp_var.get(), SHARPEN_PRESETS["Sert"])
        self.set_status("Analiz hesaplanıyor...")
        try:
            pred=predict_fixture(self.api, fx, lineup_policy=pol, allow_previous_lineup=allow_prev,
                                 backtest_asof_iso=(kickoff_iso if is_back else None),
                                 backtest_use_lineup=True,
                                 preset=preset,
                                 sharpen=sharp)
            self.render_prediction(pred, is_back, preset, sharp)
            self.set_status("Analiz tamamlandı.")
        except Exception as e:
            self.set_status("Hata.")
            messagebox.showerror("Hata", f"Analiz sırasında hata oluştu:\n{e}")

    def render_prediction(self, pred: Dict, backtest: bool, preset: RecencyPreset, sharp: SharpenPreset):
        t=self.txt; t.delete("1.0", tk.END)
        H=pred["inputs"]["home"]; A=pred["inputs"]["away"]
        t.insert(tk.END, f"{H} vs {A}\n", "title")
        t.insert(tk.END, "-"*96 + "\n", "mono")
        t.insert(tk.END, "Maç Saati (TR): ", "bold"); t.insert(tk.END, f"{pred.get('kickoff_tr','-')}\n")
        t.insert(tk.END, "Hakem: ", "bold"); t.insert(tk.END, f"{pred.get('referee','-')}\n")
        v=pred.get("venue",{})
        t.insert(tk.END, "Stadyum: ", "bold")
        venue_line=f"{v.get('name','-')} — {v.get('city','-')}, {v.get('country','-')}  •  {v.get('surface','-') or '-'}"
        if v.get("capacity"): venue_line+=f"  •  Kapasite: {v.get('capacity')}"
        t.insert(tk.END, venue_line+"\n")
        w=pred.get("weather",{})
        if w:
            t.insert(tk.END, "Hava (OWM): ", "bold")
            wline=f"{w.get('desc','-')}  •  {w.get('temp','?')}°C  •  Rüzgâr {w.get('wind','?')} m/s  •  Nem %{w.get('hum','?')}"
            t.insert(tk.END, wline+"\n")
        t.insert(tk.END, "Elo (Ev/Dep): ", "bold"); t.insert(tk.END, f"{int(pred['elo']['home'])} / {int(pred['elo']['away'])}\n")
        mv=pred.get("market_values",{})
        t.insert(tk.END, "Kadro Değeri (Ev/Dep): ", "bold"); t.insert(tk.END, f"{eur_fmt(mv.get('home_eur',0))}  /  {eur_fmt(mv.get('away_eur',0))}\n")
        lam=pred["lambdas"]
        t.insert(tk.END, "Beklenen Goller λ: ", "bold"); t.insert(tk.END, f"Ev={lam['home']:.2f} • Dep={lam['away']:.2f} • ρ≈{lam['rho']:.2f}\n")
        t.insert(tk.END, f"Form Penceresi: {preset.name}   •   Sertlik: {sharp.name}\n", "small")

        # MS 1X2
        p1,px,p2=pred["oneXtwo"]["home"], pred["oneXtwo"]["draw"], pred["oneXtwo"]["away"]
        pick_ft, p_ft = (("1",p1),("X",px),("2",p2))[np.argmax([p1,px,p2])]
        conf=pred["oneXtwo"].get("confidence",0.0)
        t.insert(tk.END, "\n[ MS 1X2 ]\n", "h")
        t.insert(tk.END, f"1: {pct(p1)}   X: {pct(px)}   2: {pct(p2)}   →  MS Tahmin: ", "mono")
        t.insert(tk.END, f"{pick_ft} ({pct(p_ft)})   Güven: {int(100*conf)}%\n", "bold")

        # İY 1X2
        ht=pred["halves"]["HT"]; ph,pxh,pah=ht["oneXtwo"]["home"], ht["oneXtwo"]["draw"], ht["oneXtwo"]["away"]
        pick_ht, p_ht=(("1",ph),("X",pxh),("2",pah))[np.argmax([ph,pxh,pah])]
        t.insert(tk.END, "\n[ İY 1X2 ]\n", "h")
        t.insert(tk.END, f"1: {pct(ph)}   X: {pct(pxh)}   2: {pct(pah)}   →  İY Tahmin: ", "mono")
        t.insert(tk.END, f"{pick_ht} ({pct(p_ht)})\n", "bold")

        # HT/FT kombinasyonları
        cmb=pred["halves"]["HTFT"]
        t.insert(tk.END, "\n[ İY/MS Kombinasyonları ]\n", "h")
        for k in ["1/1","1/X","1/2","X/1","X/X","X/2","2/1","2/X","2/2"]:
            t.insert(tk.END, f"{k}: {pct(cmb[k])}   ", "mono")
        t.insert(tk.END, "\n")

        # O/U & KG
        g=pred["goals"]; t.insert(tk.END, "\n[ Alt/Üst & KG (FT) — Sertleştirilmiş + Kalibre ]\n", "h")
        t.insert(tk.END, f"1.5 Üst: {pct(g['over_1_5'])}   1.5 Alt: {pct(g['under_1_5'])}\n", "mono")
        t.insert(tk.END, f"2.5 Üst: {pct(g['over_2_5'])}   2.5 Alt: {pct(g['under_2_5'])}\n", "mono")
        t.insert(tk.END, f"3.5 Üst: {pct(g['over_3_5'])}   3.5 Alt: {pct(g['under_3_5'])}\n", "mono")
        t.insert(tk.END, f"KG Var: {pct(g['btts_yes'])}   KG Yok: {pct(g['btts_no'])}\n", "mono")

        # Skorlar ve aralıklar
        t.insert(tk.END, "\n[ En Olası Skorlar (FT) — Top 10 ]\n", "h")
        for idx,(sc,pv) in enumerate(g["top_scores"], start=1):
            tag="mono"
            if idx<=3: tag="bold"
            t.insert(tk.END, f"{idx:>2}. {sc:<5}  {pct(pv)}\n", tag)
        rng=g["ranges"]
        t.insert(tk.END, "[ Gol Aralıkları ]  ", "bold")
        t.insert(tk.END, f"0–1: {pct(rng['0-1'])}   2–3: {pct(rng['2-3'])}   4–5: {pct(rng['4-5'])}   6+: {pct(rng['6+'])}\n", "mono")

        # Korner/Kart
        t.insert(tk.END, "\n[ Korner / Kart ]\n", "h")
        t.insert(tk.END, f"Korner beklenen: {pred['corners']['expected_total']:.2f}   (>9.5): {pct(pred['corners']['p_over_9_5'])}\n", "mono")
        t.insert(tk.END, f"Kart beklenen: {pred['cards']['expected_total']:.2f}     (>4.5): {pct(pred['cards']['p_over_4_5'])}\n", "mono")

        fm=pred.get("formations",{}); lineup_str = "RESMÎ İLK 11" if pred["has_lineup"] else ("SON KADRO" if pred["used_previous_lineup"] else "YOK")
        t.insert(tk.END, "\n[ Kadro / Dizilim ]\n", "h")
        t.insert(tk.END, f"Kadro: {lineup_str}    Dizilim (Ev/Dep): {(fm.get('home') or '-')} / {(fm.get('away') or '-')}\n", "mono")

        t.insert(tk.END, "\nNot: En güvenilir analiz, maçtan ~60 dk önce (kadrolar açıklandığında) yapılır. "
                 "Sertlik, OU/KG kararlarını netleştirir; kalibrasyon, olasılıkların güvenilirliğini artırır.\n", "small")

    # -------------------- Backtest Dialog --------------------

    def open_backtest_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Toplu Backtest")
        dlg.geometry("560x420")
        dlg.resizable(False, False)

        def add_row(r, label, var, width=14):
            ttk.Label(dlg, text=label, style="Bold.TLabel").grid(row=r, column=0, sticky="e", padx=(12,6), pady=6)
            e=tk.Entry(dlg, textvariable=var, width=width); e.grid(row=r, column=1, sticky="w", padx=0, pady=6)
            return e

        seed = self.date_var.get().strip()
        if not parse_tr_date(seed):
            seed = datetime.now().strftime("%d.%m.%Y")

        start_var = tk.StringVar(value=seed)
        end_var   = tk.StringVar(value=seed)
        max_var   = tk.StringVar(value="50")
        thr_var   = tk.StringVar(value="0.55")
        save_csv_var   = tk.BooleanVar(value=True)   # eski tek dosya CSV
        autosave_var   = tk.BooleanVar(value=True)   # klasör halinde detaylı kayıt
        update_calib_var = tk.BooleanVar(value=True) # kalibrasyonu kaydet

        add_row(0, "Başlangıç (GG.AA.YYYY):", start_var)
        add_row(1, "Bitiş (GG.AA.YYYY):", end_var)
        add_row(2, "Maks. Maç:", max_var)
        add_row(3, "Karar Eşiği (OU/KG):", thr_var)

        ttk.Label(dlg, text="Lig Filtresi:", style="Bold.TLabel").grid(row=4, column=0, sticky="e", padx=(12,6), pady=6)
        lf=ttk.Combobox(dlg, values=["Hepsi","Ana Ligler"], state="readonly", width=12)
        lf.set(self.filter_mode.get()); lf.grid(row=4, column=1, sticky="w", pady=6)

        ttk.Label(dlg, text="Form Penceresi:", style="Bold.TLabel").grid(row=5, column=0, sticky="e", padx=(12,6), pady=6)
        rec_cb=ttk.Combobox(dlg, values=list(RECENCY_PRESETS.keys()), state="readonly", width=12)
        rec_cb.set(self.recency_var.get()); rec_cb.grid(row=5, column=1, sticky="w", pady=6)

        ttk.Label(dlg, text="Sertlik:", style="Bold.TLabel").grid(row=6, column=0, sticky="e", padx=(12,6), pady=6)
        shp_cb=ttk.Combobox(dlg, values=list(SHARPEN_PRESETS.keys()), state="readonly", width=12)
        shp_cb.set(self.sharp_var.get()); shp_cb.grid(row=6, column=1, sticky="w", pady=6)

        ttk.Checkbutton(dlg, text="CSV kaydet (tek dosya)", variable=save_csv_var).grid(row=7, column=1, sticky="w", pady=2)
        ttk.Checkbutton(dlg, text="Otomatik kaydet (klasör)", variable=autosave_var).grid(row=8, column=1, sticky="w", pady=2)
        ttk.Checkbutton(dlg, text="Kalibrasyonu güncelle (T/k/g kaydet)", variable=update_calib_var).grid(row=9, column=1, sticky="w", pady=2)

        btnf=ttk.Frame(dlg); btnf.grid(row=10, column=0, columnspan=2, pady=(10,8))
        def run():
            start_iso=parse_tr_date(start_var.get().strip()); end_iso=parse_tr_date(end_var.get().strip())
            if not start_iso or not end_iso:
                messagebox.showwarning("Tarih", "Tarih formatı hatalı."); return
            try:
                max_matches=max(1, int(max_var.get()))
            except Exception:
                messagebox.showwarning("Maks. Maç", "Geçerli bir sayı giriniz."); return
            try:
                threshold = float(thr_var.get()); 
                if not (0.50 <= threshold <= 0.70): raise ValueError()
            except Exception:
                messagebox.showwarning("Eşik", "Eşik 0.50 ile 0.70 arasında olmalı."); return

            major_only = (lf.get()=="Ana Ligler")
            preset = RECENCY_PRESETS.get(rec_cb.get(), RECENCY_PRESETS["Oto"])
            sharp  = SHARPEN_PRESETS.get(shp_cb.get(), SHARPEN_PRESETS["Sert"])
            dlg.destroy()
            self.run_batch_backtest(start_iso, end_iso, max_matches, threshold, major_only, preset, sharp,
                                    save_csv=save_csv_var.get(), auto_save=autosave_var.get(),
                                    update_calib=update_calib_var.get())

        ttk.Button(btnf, text="Çalıştır", command=run).pack(side=tk.LEFT, padx=6)
        ttk.Button(btnf, text="Kapat", command=dlg.destroy).pack(side=tk.LEFT, padx=6)

    # -------------------- Backtest yardımcıları --------------------

    def _collect_finished_fixtures_by_day(self, start_iso: str, end_iso: str) -> List[Dict]:
        out=[]
        try:
            d0=datetime.strptime(start_iso,"%Y-%m-%d")
            d1=datetime.strptime(end_iso,"%Y-%m-%d")
        except Exception:
            return out
        step=timedelta(days=1)
        cur=d0
        while cur<=d1:
            day=cur.strftime("%Y-%m-%d")
            try:
                arr=self.api.fixtures_by_date_smart(day) or []
            except Exception:
                arr=[]
            out.extend(arr)
            cur+=step
        out=[m for m in out if m["fixture"]["status"]["short"] in ("FT","AET","PEN")]
        if out:
            return out
        try:
            arr=self.api.fixtures_range_raw(start_iso, end_iso) or []
        except Exception:
            arr=[]
        out=[m for m in arr if m["fixture"]["status"]["short"] in ("FT","AET","PEN")]
        return out

    # -------------------- Backtest ana akış --------------------

    def run_batch_backtest(self, start_iso: str, end_iso: str, max_matches: int, threshold: float,
                           major_only: bool, preset: RecencyPreset, sharp: SharpenPreset,
                           save_csv: bool, auto_save: bool, update_calib: bool):
        self.txt.delete("1.0", tk.END)
        self.set_status("Backtest: maçlar taranıyor...")

        fixtures=self._collect_finished_fixtures_by_day(start_iso, end_iso)
        if major_only:
            fixtures=[m for m in fixtures if m["league"]["name"] in MAJOR_LEAGUES]

        fixtures.sort(key=lambda x: x["fixture"].get("timestamp",0))
        uniq={}; ordered=[]
        for m in fixtures:
            fid=safe_int(m["fixture"]["id"])
            if fid in uniq: continue
            uniq[fid]=True; ordered.append(m)

        fixtures=ordered[:max_matches]

        if not fixtures:
            self.set_status("Seçili aralıkta backtest yapılacak maç bulunamadı.")
            self.txt.insert(tk.END, "Seçili aralıkta backtest yapılacak maç bulunamadı. (FT/AET/PEN yok)\n", "bad")
            self.txt.insert(tk.END, f"Aralık: {start_iso} → {end_iso}  •  Lig filtresi: {'Ana Ligler' if major_only else 'Hepsi'}\n", "small")
            return

        # Kayıt klasörleri
        ensure_dir("backtests")
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = os.path.join("backtests", run_id)
        if auto_save:
            ensure_dir(out_dir)

        # sayaç/metrikler
        n=len(fixtures)
        ms_hit=0; ms_brier_sum=0.0; ms_logloss_sum=0.0
        ou_cov=0; ou_hit=0; ou_probs=[]; ou_labels=[]
        btts_cov=0; btts_hit=0; btts_probs=[]; btts_labels=[]
        sc_top1=0; sc_top3=0
        range_hit=0
        score_logloss_sum=0.0; score_ll_count=0
        triples=[]   # (p1,px,p2,outcome)
        score_data=[]  # (P, gh, ga)
        league_map: Dict[str, Dict[str,float]] = {}

        matches_detail=[]
        csv_rows=[]  # eski tek dosya csv için zengin satır

        self.set_status(f"Backtest: {n} maç işleniyor...")

        for idx,fx in enumerate(fixtures, start=1):
            try:
                kickoff_iso=fx["fixture"]["date"]
                pred=predict_fixture(self.api, fx,
                                     lineup_policy=self.policy_var.get(),
                                     allow_previous_lineup=self.use_prev_var.get(),
                                     backtest_asof_iso=kickoff_iso,
                                     backtest_use_lineup=True,
                                     preset=preset,
                                     sharpen=sharp)

                gh=safe_int(fx["goals"]["home"]); ga=safe_int(fx["goals"]["away"])
                outcome=("1" if gh>ga else ("2" if ga>gh else "X"))
                p1,px,p2 = pred["oneXtwo"]["home"], pred["oneXtwo"]["draw"], pred["oneXtwo"]["away"]
                pick = ("1" if p1>=max(px,p2) else ("X" if px>=max(p1,p2) else "2"))
                hit_1x2 = int(pick==outcome)
                ms_hit += hit_1x2
                ms_brier_sum += brier_1x2(p1,px,p2, outcome)
                ms_logloss_sum += logloss({"1":p1,"X":px,"2":p2}[outcome])
                triples.append((p1,px,p2,outcome))

                # OU 2.5
                p_over25 = pred["goals"]["over_2_5"]; p_under25 = 1.0-p_over25
                ou_actual=(gh+ga)>=3
                ou_pick=None; ou_prob=None
                if p_over25>=threshold or p_under25>=threshold:
                    ou_cov+=1
                    if p_over25>=p_under25:
                        ou_pick="Over2.5"; ou_prob=p_over25; ou_hit+=int(ou_actual is True)
                    else:
                        ou_pick="Under2.5"; ou_prob=p_under25; ou_hit+=int(ou_actual is False)
                ou_probs.append(p_over25); ou_labels.append(1 if ou_actual else 0)

                # BTTS
                p_btts = pred["goals"]["btts_yes"]; btts_actual=(gh>0 and ga>0)
                btts_pick=None; btts_prob=None
                if p_btts>=threshold or (1.0-p_btts)>=threshold:
                    btts_cov+=1
                    if p_btts>=1.0-p_btts:
                        btts_pick="BTTS_Yes"; btts_prob=p_btts; btts_hit+=int(btts_actual is True)
                    else:
                        btts_pick="BTTS_No"; btts_prob=(1.0-p_btts); btts_hit+=int(btts_actual is False)
                btts_probs.append(p_btts); btts_labels.append(1 if btts_actual else 0)

                # Skor Top‑1/Top‑3
                top_scores = pred["goals"]["top_scores"]
                top1 = top_scores[0][0] if top_scores else "-"
                in_top1=False; in_top3=False
                if top_scores:
                    in_top1 = (top1 == f"{gh}-{ga}")
                    in_top3 = any((lab==f"{gh}-{ga}") for lab,_ in top_scores[:3])
                sc_top1 += int(in_top1); sc_top3 += int(in_top3)

                # Gol aralığı
                pred_ranges = pred["goals"]["ranges"]
                pick_range = max(pred_ranges.items(), key=lambda kv: kv[1])[0]
                if pick_range in ("0-1","2-3","4-5","6+"):
                    range_hit += int(pick_range== ("0-1" if gh+ga<=1 else ("2-3" if gh+ga<=3 else ("4-5" if gh+ga<=5 else "6+"))))

                # Skor matrisi logloss
                P = pred.get("score_matrix", None)
                score_prob=None
                if isinstance(P, np.ndarray) and 0<=gh<=8 and 0<=ga<=8:
                    score_prob = float(P[gh,ga])
                    score_logloss_sum += logloss(score_prob); score_ll_count += 1
                    score_data.append((P, gh, ga))

                # Lig özetine ekle
                lig=fx["league"]["name"]
                d=league_map.setdefault(lig, {"n":0,"ms_hit":0,"ou_cov":0,"ou_hit":0,"btts_cov":0,"btts_hit":0})
                d["n"]+=1; d["ms_hit"]+=hit_1x2
                if ou_pick is not None: d["ou_cov"]+=1; d["ou_hit"]+=int(((ou_pick=="Over2.5") and ou_actual) or ((ou_pick=="Under2.5") and (not ou_actual)))
                if btts_pick is not None: d["btts_cov"]+=1; d["btts_hit"]+=int(((btts_pick=="BTTS_Yes") and btts_actual) or ((btts_pick=="BTTS_No") and (not btts_actual)))

                row={
                    "kickoff": kickoff_iso,
                    "league": fx["league"]["name"],
                    "home": fx["teams"]["home"]["name"],
                    "away": fx["teams"]["away"]["name"],
                    "FT": f"{gh}-{ga}",
                    "pick_1x2": pick,
                    "p1": p1, "px": px, "p2": p2,
                    "hit_1x2": hit_1x2,
                    "ou_pick": ou_pick or "SKIP",
                    "p_over25": p_over25,
                    "ou_hit": (None if ou_pick is None else int(((ou_pick=="Over2.5") and ou_actual) or ((ou_pick=="Under2.5") and (not ou_actual)))),
                    "btts_pick": btts_pick or "SKIP",
                    "p_btts": p_btts,
                    "btts_hit": (None if btts_pick is None else int(((btts_pick=="BTTS_Yes") and btts_actual) or ((btts_pick=="BTTS_No") and (not btts_actual)))),
                    "top1": top1, "top3_hit": int(in_top3),
                    "range_pick": pick_range, "range_hit": int(pick_range== ("0-1" if gh+ga<=1 else ("2-3" if gh+ga<=3 else ("4-5" if gh+ga<=5 else "6+")))),
                    "score_prob": score_prob,
                    "lam_h": pred["lambdas"]["home"], "lam_a": pred["lambdas"]["away"],
                    "conf": pred["oneXtwo"]["confidence"]
                }
                matches_detail.append(row)

                # Eski tek dosya CSV satırı
                csv_rows.append({
                    "kickoff": kickoff_iso, "league": fx["league"]["name"],
                    "home": fx["teams"]["home"]["name"], "away": fx["teams"]["away"]["name"], "FT": f"{gh}-{ga}",
                    "pick_1x2": pick, "p1": f"{p1:.4f}", "px": f"{px:.4f}", "p2": f"{p2:.4f}", "hit_1x2": hit_1x2,
                    "p_over25": f"{p_over25:.4f}", "ou_pick": ou_pick or "SKIP", "ou_hit": ("" if ou_pick is None else int(((ou_pick=='Over2.5') and ou_actual) or ((ou_pick=='Under2.5') and (not ou_actual)))),
                    "p_btts": f"{p_btts:.4f}", "btts_pick": btts_pick or "SKIP", "btts_hit": ("" if btts_pick is None else int(((btts_pick=='BTTS_Yes') and btts_actual) or ((btts_pick=='BTTS_No') and (not btts_actual)))),
                    "top1": top1, "top3_hit": int(in_top3),
                    "range_pick": pick_range, "range_hit": int(pick_range== ("0-1" if gh+ga<=1 else ("2-3" if gh+ga<=3 else ("4-5" if gh+ga<=5 else "6+")))),
                    "score_prob": ("" if score_prob is None else f"{score_prob:.6f}"),
                    "lam_h": f"{pred['lambdas']['home']:.2f}",
                    "lam_a": f"{pred['lambdas']['away']:.2f}",
                    "conf": f"{pred['oneXtwo']['confidence']:.3f}"
                })

            except Exception as e:
                csv_rows.append({
                    "kickoff": fx["fixture"]["date"],
                    "league": fx["league"]["name"],
                    "home": fx["teams"]["home"]["name"],
                    "away": fx["teams"]["away"]["name"],
                    "FT": "ERR",
                    "error": str(e)[:120]
                })

            self.set_status(f"Backtest: {idx}/{n} maç işlendi...")

        # ÖZET metrikler
        ms_acc = ms_hit/max(1,n)
        ou_acc = (ou_hit/max(1,ou_cov)) if ou_cov>0 else 0.0
        btts_acc = (btts_hit/max(1,btts_cov)) if btts_cov>0 else 0.0
        top1_acc = sc_top1/max(1,n)
        top3_acc = sc_top3/max(1,n)
        range_acc = range_hit/max(1,n)
        ms_brier = ms_brier_sum/max(1,n)
        ms_ll    = ms_logloss_sum/max(1,n)
        score_ll = (score_logloss_sum/max(1,score_ll_count)) if score_ll_count>0 else None

        # AUC & Brier (OU/KG)
        auc_ou = auc_binary(ou_probs, ou_labels)
        auc_btts = auc_binary(btts_probs, btts_labels)
        brier_ou = float(np.mean([(p - y)**2 for p,y in zip(ou_probs,ou_labels)])) if ou_probs else None
        brier_btts = float(np.mean([(p - y)**2 for p,y in zip(btts_probs,btts_labels)])) if btts_probs else None

        # Lig özeti
        league_summary=[]
        for lig,d in sorted(league_map.items(), key=lambda kv: -kv[1]["n"]):
            league_summary.append({
                "league": lig,
                "n": d["n"],
                "ms_acc": d["ms_hit"]/max(1,d["n"]),
                "ou_cov": d["ou_cov"]/max(1,d["n"]),
                "ou_acc": (d["ou_hit"]/max(1,d["ou_cov"])) if d["ou_cov"]>0 else 0.0,
                "btts_cov": d["btts_cov"]/max(1,d["n"]),
                "btts_acc": (d["btts_hit"]/max(1,d["btts_cov"])) if d["btts_cov"]>0 else 0.0
            })

        # Eşik taraması
        sweep_ou=[]; sweep_btts=[]
        for thr in [round(x,2) for x in np.linspace(0.50,0.70,11)]:
            # OU
            cov=sum(1 for p in ou_probs if (p>=thr or (1-p)>=thr))/max(1,len(ou_probs))
            hit=0; covn=0
            for p,y in zip(ou_probs, ou_labels):
                if p>=thr or (1-p)>=thr:
                    covn+=1; pick=1 if p>=1-p else 0; hit+=int(pick==y)
            acc=(hit/max(1,covn)) if covn>0 else 0.0
            sweep_ou.append({"thr":thr,"cov":cov,"acc":acc})
            # BTTS
            cov=sum(1 for p in btts_probs if (p>=thr or (1-p)>=thr))/max(1,len(btts_probs))
            hit=0; covn=0
            for p,y in zip(btts_probs, btts_labels):
                if p>=thr or (1-p)>=thr:
                    covn+=1; pick=1 if p>=1-p else 0; hit+=int(pick==y)
            acc=(hit/max(1,covn)) if covn>0 else 0.0
            sweep_btts.append({"thr":thr,"cov":cov,"acc":acc})

        # Güvenilirlik (MS)
        max_probs=[]; hits=[]
        for p1,px,p2,outcome in triples:
            v=max(p1,px,p2)
            max_probs.append(v)
            pick=("1" if p1>=max(px,p2) else ("X" if px>=max(p1,p2) else "2"))
            hits.append(1 if pick==outcome else 0)
        reliab=reliability_bins(max_probs, hits)

        # Kalibrasyon önerisi
        bestT, base1x2, best1x2 = optimize_temperature_1x2(triples)
        bestK_ou, baseOU, bestOU = optimize_center_push_binary(ou_probs, ou_labels)
        bestK_btts, baseBT, bestBT = optimize_center_push_binary(btts_probs, btts_labels)
        bestG_score, baseScore, bestScore = optimize_gamma_for_scores(score_data)

        # ÇIKTI metni
        t=self.txt
        t.insert(tk.END, f"[ TOPLU BACKTEST ]  {start_iso} → {end_iso}   (Maç: {n})\n", "title")
        t.insert(tk.END, "-"*88 + "\n", "mono")
        t.insert(tk.END, f"MS 1X2  isabet : {ms_hit}/{n}  ({pct(ms_acc)})\n", "bold")
        t.insert(tk.END, f"MS 1X2  Brier  : {ms_brier:.4f}    LogLoss: {ms_ll:.4f}\n", "mono")
        t.insert(tk.END, f"OU 2.5 kapsama: {ou_cov}/{n}  ({pct(ou_cov/max(1,n))})   isabet: {ou_hit}/{max(1,ou_cov)}  ({pct(ou_acc)})   [eşik={threshold:.2f}]\n", "mono")
        t.insert(tk.END, f"KG (BTTS) kapsama: {btts_cov}/{n}  ({pct(btts_cov/max(1,n))})   isabet: {btts_hit}/{max(1,btts_cov)}  ({pct(btts_acc)})   [eşik={threshold:.2f}]\n", "mono")
        t.insert(tk.END, f"Skor Top‑1: {sc_top1}/{n}  ({pct(top1_acc)})    Top‑3: {sc_top3}/{n}  ({pct(top3_acc)})\n", "mono")
        t.insert(tk.END, f"Gol aralığı isabet: {range_hit}/{n}  ({pct(range_acc)})\n", "mono")
        if score_ll is not None:
            t.insert(tk.END, f"Skor matrisi LogLoss: {score_ll:.4f}\n", "mono")
        # AUC satırını güvenli biçimde yaz
        t.insert(tk.END, f"OU Brier: {fmt_opt(brier_ou,4)}   AUC: {fmt_opt(auc_ou,3)}   |   "
                         f"KG Brier: {fmt_opt(brier_btts,4)}   AUC: {fmt_opt(auc_btts,3)}\n", "mono")

        # Kalibrasyon önerisini yaz
        t.insert(tk.END, "\n[ Kalibrasyon Önerisi ]\n", "h")
        t.insert(tk.END, f"1X2 sıcaklık T: önerilen {bestT:.2f}  (LogLoss {fmt_opt(base1x2,4)} → {fmt_opt(best1x2,4)})\n", "mono")
        t.insert(tk.END, f"OU push k_over: önerilen {bestK_ou:+.2f}  (LL {fmt_opt(baseOU,4)} → {fmt_opt(bestOU,4)})\n", "mono")
        t.insert(tk.END, f"KG push k_btts: önerilen {bestK_btts:+.2f}  (LL {fmt_opt(baseBT,4)} → {fmt_opt(bestBT,4)})\n", "mono")
        if not math.isnan(bestScore):
            t.insert(tk.END, f"Skor gamma g: önerilen {bestG_score:.2f}  (LL {fmt_opt(baseScore,4)} → {fmt_opt(bestScore,4)})\n", "mono")

        # Eşik önerisi
        best_ou=max(sweep_ou, key=lambda d: (d["acc"], -abs(d["cov"]-0.6)))
        best_btts=max(sweep_btts, key=lambda d: (d["acc"], -abs(d["cov"]-0.6)))
        t.insert(tk.END, f"\n[ Eşik Önerisi ]  OU: {best_ou['thr']:.2f}  (Kap: {best_ou['cov']*100:.1f}%, İsab: {best_ou['acc']*100:.1f}%)   "
                         f"KG: {best_btts['thr']:.2f}  (Kap: {best_btts['cov']*100:.1f}%, İsab: {best_btts['acc']*100:.1f}%)\n", "mono")

        # --------- Kayıt (otomatik klasör + eski tek dosya) ----------
        # 1) Otomatik klasör kayıtları
        if auto_save:
            try:
                # matches.csv
                matches_csv = os.path.join(out_dir, "matches.csv")
                with open(matches_csv, "w", newline="", encoding="utf-8-sig") as f:
                    cols = ["kickoff","league","home","away","FT","pick_1x2","p1","px","p2","hit_1x2",
                            "p_over25","ou_pick","ou_hit","p_btts","btts_pick","btts_hit",
                            "top1","top3_hit","range_pick","range_hit","score_prob","lam_h","lam_a","conf"]
                    w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
                    for r in matches_detail:
                        w.writerow({
                            "kickoff": r["kickoff"], "league": r["league"], "home": r["home"], "away": r["away"], "FT": r["FT"],
                            "pick_1x2": r["pick_1x2"], "p1": f"{r['p1']:.4f}", "px": f"{r['px']:.4f}", "p2": f"{r['p2']:.4f}", "hit_1x2": r["hit_1x2"],
                            "p_over25": f"{r['p_over25']:.4f}", "ou_pick": r["ou_pick"], "ou_hit": ("" if r["ou_hit"] is None else r["ou_hit"]),
                            "p_btts": f"{r['p_btts']:.4f}", "btts_pick": r["btts_pick"], "btts_hit": ("" if r["btts_hit"] is None else r["btts_hit"]),
                            "top1": r["top1"], "top3_hit": r["top3_hit"], "range_pick": r["range_pick"], "range_hit": r["range_hit"],
                            "score_prob": ("" if r["score_prob"] is None else f"{r['score_prob']:.6f}"),
                            "lam_h": f"{r['lam_h']:.2f}", "lam_a": f"{r['lam_a']:.2f}", "conf": f"{r['conf']:.3f}"
                        })
                # league_summary.csv
                with open(os.path.join(out_dir, "league_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["league","n","ms_acc","ou_cov","ou_acc","btts_cov","btts_acc"])
                    for s in league_summary:
                        w.writerow([s["league"], s["n"], f"{s['ms_acc']:.4f}", f"{s['ou_cov']:.4f}", f"{s['ou_acc']:.4f}",
                                    f"{s['btts_cov']:.4f}", f"{s['btts_acc']:.4f}"])
                # sweep_ou.csv / sweep_btts.csv
                with open(os.path.join(out_dir, "sweep_ou.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["thr","coverage","accuracy"])
                    for d in sweep_ou: w.writerow([f"{d['thr']:.2f}", f"{d['cov']:.4f}", f"{d['acc']:.4f}"])
                with open(os.path.join(out_dir, "sweep_btts.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["thr","coverage","accuracy"])
                    for d in sweep_btts: w.writerow([f"{d['thr']:.2f}", f"{d['cov']:.4f}", f"{d['acc']:.4f}"])
                # reliability.csv
                with open(os.path.join(out_dir, "reliability.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["range","n","avg_p","acc"])
                    for r in reliab:
                        w.writerow([r["range"], r["n"], ("" if r["avg_p"] is None else f"{r['avg_p']:.4f}"),
                                    ("" if r["acc"] is None else f"{r['acc']:.4f}")])
                # summary.json
                summary = {
                    "range": {"start": start_iso, "end": end_iso, "n": n},
                    "metrics": {"ms_acc": ms_acc, "ms_brier": ms_brier, "ms_logloss": ms_ll,
                                "ou_coverage": ou_cov/max(1,n), "ou_acc": ou_acc, "ou_brier": brier_ou, "ou_auc": auc_ou,
                                "btts_coverage": btts_cov/max(1,n), "btts_acc": btts_acc, "btts_brier": brier_btts, "btts_auc": auc_btts,
                                "score_logloss": score_ll, "top1": top1_acc, "top3": top3_acc, "sum_range_acc": range_acc},
                    "threshold_sweep": {"OU": sweep_ou, "BTTS": sweep_btts},
                    "reliability": reliab,
                    "calibration_suggested": {
                        "oneXtwo_T": bestT, "over_push": bestK_ou, "btts_push": bestK_btts,
                        "score_gamma": (bestG_score if not math.isnan(bestScore) else 1.0)
                    }
                }
                with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                # report.txt (ekranda yazılan metin)
                with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
                    f.write(self.txt.get("1.0", tk.END))

                # history.csv (append)
                hist_path = os.path.join("backtests", "history.csv")
                is_new = not os.path.exists(hist_path)
                with open(hist_path, "a", newline="", encoding="utf-8-sig") as f:
                    w=csv.writer(f)
                    if is_new:
                        w.writerow(["run_id","start","end","n","ms_acc","ms_ll","ou_acc","btts_acc","score_ll","bestT","k_over","k_btts","g_score"])
                    w.writerow([run_id, start_iso, end_iso, n, f"{ms_acc:.4f}", f"{ms_ll:.4f}",
                                f"{ou_acc:.4f}", f"{btts_acc:.4f}", ("" if score_ll is None else f"{score_ll:.4f}"),
                                f"{bestT:.2f}", f"{bestK_ou:+.2f}", f"{bestK_btts:+.2f}",
                                ("" if math.isnan(bestG_score) else f"{bestG_score:.2f}")])

                self.txt.insert(tk.END, f"\nDetaylı kayıt klasörü: {out_dir}\n", "ok")
            except Exception as e:
                self.txt.insert(tk.END, f"\nKayıt hatası: {e}\n", "bad")

        # 2) Eski tek dosya CSV (isteğe bağlı)
        if save_csv and csv_rows:
            try:
                simple_csv = os.path.join("backtests", f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                with open(simple_csv, "w", newline="", encoding="utf-8-sig") as f:
                    writer=csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                    writer.writeheader()
                    for r in csv_rows: writer.writerow(r)
                self.txt.insert(tk.END, f"CSV (tek dosya) kaydedildi: {simple_csv}\n", "ok")
            except Exception as e:
                self.txt.insert(tk.END, f"CSV yazma hatası: {e}\n", "bad")

        # Detay penceresi
        try:
            win = BacktestDetailWindow(self, matches_detail, league_summary, sweep_ou, sweep_btts, reliab,
                                       out_dir if auto_save else None)
            win.grab_set()
        except Exception as e:
            self.txt.insert(tk.END, f"\nDetay penceresi açılamadı: {e}\n", "bad")

        # Kalibrasyonu diske yaz (isteğe bağlı)
        if update_calib:
            newc = {
                "oneXtwo_T": bestT,
                "over_push": bestK_ou,
                "btts_push": bestK_btts,
                "score_gamma": bestG_score if not math.isnan(bestScore) else 1.0
            }
            save_calibration(newc)
            global CALIB
            CALIB = load_calibration()
            self.txt.insert(tk.END, f"\nKalibrasyon kaydedildi → {CALIB_PATH}\n", "ok")

        self.set_status("Backtest tamamlandı.")

    # -------------------- Kaydet --------------------

    def save_report(self):
        txt=self.txt.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showinfo("Kaydet", "Kaydedilecek analiz yok."); return
        path=filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Metin Dosyası","*.txt")])
        if not path: return
        try:
            with open(path,"w",encoding="utf-8") as f: f.write(txt)
            messagebox.showinfo("Kaydet", f"Analiz kaydedildi:\n{path}")
        except Exception as e:
            messagebox.showerror("Kaydet", f"Hata: {e}")

# ======================================================================
# ÇALIŞTIR
# ======================================================================

def main():
    if not API_SPORTS_KEY:
        messagebox.showerror("API Anahtarı", "API_SPORTS_KEY bulunamadı."); return
    api=ApiSports(API_SPORTS_KEY)
    app=App(api)
    app.mainloop()

if __name__ == "__main__":
    main()
