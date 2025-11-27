"""
Geliştirici modlu komut satırı backtest aracı.

Bu modül, lig ve bahis türü (örneğin "OVER25", "BTTS") bazında
kalibrasyon/modele göre skorlamayı farklılaştırarak sonuçları
karşılaştırmalı olarak raporlar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Varsayılan tekil kalibrasyon parametreleri
DEFAULT_CALIBRATION = {"temperature": 1.0, "bias": 0.0}
DEFAULT_CALIBRATION_MAP = "calibration_profiles.json"


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def apply_calibration(prob: float, calibration: Dict[str, float]) -> float:
    """Sıcaklık (temperature) ve sapma (bias) ile olasılık düzeltmesi uygular."""
    temp = float(calibration.get("temperature", DEFAULT_CALIBRATION["temperature"]))
    bias = float(calibration.get("bias", DEFAULT_CALIBRATION["bias"]))
    temp = max(temp, 1e-3)  # sıfır bölme koruması
    return sigmoid((logit(prob) + bias) / temp)


@dataclass
class MatchRow:
    league: str
    bet_type: str
    predicted_prob: float
    outcome: int
    meta: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_csv_row(row: Dict[str, str]) -> "MatchRow":
        try:
            prob = float(row.get("predicted_prob", 0.0))
        except Exception:
            prob = 0.0
        try:
            outcome = int(row.get("outcome", 0))
        except Exception:
            outcome = 0
        meta = {k: v for k, v in row.items() if k not in {"league", "bet_type", "predicted_prob", "outcome"}}
        return MatchRow(
            league=row.get("league", "Unknown"),
            bet_type=row.get("bet_type", "GENERIC"),
            predicted_prob=max(0.0, min(1.0, prob)),
            outcome=1 if outcome else 0,
            meta=meta,
        )


@dataclass
class Evaluation:
    logloss: float
    brier: float
    accuracy: float
    samples: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "logloss": self.logloss,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "samples": self.samples,
        }


def evaluate(predictions: Iterable[Tuple[float, int]]) -> Evaluation:
    preds = list(predictions)
    if not preds:
        return Evaluation(logloss=float("nan"), brier=float("nan"), accuracy=float("nan"), samples=0)
    logloss_sum = 0.0
    brier_sum = 0.0
    correct = 0
    for prob, outcome in preds:
        prob = min(max(prob, 1e-9), 1 - 1e-9)
        logloss_sum += - (outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))
        brier_sum += (prob - outcome) ** 2
        if (prob >= 0.5 and outcome == 1) or (prob < 0.5 and outcome == 0):
            correct += 1
    n = len(preds)
    return Evaluation(
        logloss=logloss_sum / n,
        brier=brier_sum / n,
        accuracy=correct / n,
        samples=n,
    )


def load_calibration_map(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def select_calibration(
    league: str,
    bet_type: str,
    developer_mode: bool,
    calibration_map: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    if developer_mode:
        league_map = calibration_map.get(league) or calibration_map.get("*")
        if league_map:
            return league_map.get(bet_type) or league_map.get("*") or DEFAULT_CALIBRATION
    return DEFAULT_CALIBRATION


def run_backtest(
    matches: List[MatchRow],
    developer_mode: bool,
    calibration_map: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Evaluation]:
    baseline_preds = [(m.predicted_prob, m.outcome) for m in matches]
    dev_preds: List[Tuple[float, int]] = []
    per_combo: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}

    for m in matches:
        calibration = select_calibration(m.league, m.bet_type, developer_mode, calibration_map)
        adjusted = apply_calibration(m.predicted_prob, calibration)
        dev_preds.append((adjusted, m.outcome))
        key = (m.league, m.bet_type)
        per_combo.setdefault(key, []).append((adjusted, m.outcome))

    results: Dict[str, Evaluation] = {
        "baseline": evaluate(baseline_preds),
        "developer": evaluate(dev_preds),
    }

    # Detaylı kombinasyon değerlendirmeleri
    for (league, bet_type), preds in sorted(per_combo.items()):
        label = f"{league} | {bet_type}"
        results[label] = evaluate(preds)
    return results


def load_matches(path: str) -> List[MatchRow]:
    import csv

    matches: List[MatchRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append(MatchRow.from_csv_row(row))
    return matches


def ensure_demo_data() -> List[MatchRow]:
    # Küçük bir demo seti: iki lig ve iki bahis türü
    demo_rows = [
        {"league": "Süper Lig", "bet_type": "OVER25", "predicted_prob": 0.61, "outcome": 1},
        {"league": "Süper Lig", "bet_type": "BTTS", "predicted_prob": 0.48, "outcome": 1},
        {"league": "Serie A", "bet_type": "OVER25", "predicted_prob": 0.53, "outcome": 0},
        {"league": "Serie A", "bet_type": "BTTS", "predicted_prob": 0.40, "outcome": 0},
        {"league": "Serie A", "bet_type": "BTTS", "predicted_prob": 0.52, "outcome": 1},
        {"league": "Süper Lig", "bet_type": "OVER25", "predicted_prob": 0.72, "outcome": 0},
    ]
    return [MatchRow.from_csv_row(r) for r in demo_rows]


def demo_calibration_map() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "Süper Lig": {
            "OVER25": {"temperature": 0.9, "bias": 0.05},
            "BTTS": {"temperature": 1.05, "bias": 0.02},
        },
        "Serie A": {
            "OVER25": {"temperature": 1.1, "bias": -0.04},
            "BTTS": {"temperature": 0.95, "bias": -0.02},
        },
        "*": {"*": DEFAULT_CALIBRATION},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lig/bahis bazlı kalibrasyonlu backtest")
    parser.add_argument("--input", "-i", help="CSV veri yolu (league, bet_type, predicted_prob, outcome)")
    parser.add_argument("--developer-mode", action="store_true", help="Lig/bahis bazlı kalibrasyonları uygula")
    parser.add_argument("--calibration-map", default=DEFAULT_CALIBRATION_MAP, help="Lig/bahis kalibrasyonları JSON")
    parser.add_argument("--report", help="JSON rapor çıktısı")
    parser.add_argument("--demo-data", action="store_true", help="Yerleşik demo datasıyla çalıştır")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo_data and args.input:
        parser.error("--demo-data ile --input birlikte kullanılamaz")

    if args.demo_data:
        matches = ensure_demo_data()
        calibration_map = demo_calibration_map()
    else:
        if not args.input:
            parser.error("CSV girişi için --input veya --demo-data belirtin")
        matches = load_matches(args.input)
        calibration_map = load_calibration_map(args.calibration_map)
        if not calibration_map:
            calibration_map = demo_calibration_map()

    results = run_backtest(matches, developer_mode=args.developer_mode, calibration_map=calibration_map)

    print("== Backtest Özeti ==")
    print(f"Toplam örnek: {results['baseline'].samples}")
    print("\nVarsayılan akış (geliştirici modu kapalı):")
    print(f"  LogLoss: {results['baseline'].logloss:.4f}\n  Brier: {results['baseline'].brier:.4f}\n  Doğruluk: {results['baseline'].accuracy:.3f}")

    print("\nGeliştirici modu (lig/bahis kalibrasyonlu):")
    dev = results.get("developer")
    print(f"  LogLoss: {dev.logloss:.4f}\n  Brier: {dev.brier:.4f}\n  Doğruluk: {dev.accuracy:.3f}")

    if any(k not in {"baseline", "developer"} for k in results):
        print("\nLig & bahis kırılımı:")
        for key, eval_res in results.items():
            if key in {"baseline", "developer"}:
                continue
            print(f"  {key} → LogLoss {eval_res.logloss:.4f}, Brier {eval_res.brier:.4f}, Doğruluk {eval_res.accuracy:.3f}, N={eval_res.samples}")

    if args.report:
        serialized = {k: v.as_dict() for k, v in results.items()}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        print(f"\nJSON rapor yazıldı: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
