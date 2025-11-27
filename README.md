# depo

Gelişmiş futbol tahmin/backtest aracı. GUI sürümüne ek olarak, komut satırı odaklı
`backtest.py` modülü lig ve bahis türü bazlı kalibrasyonla geliştirici modu
sunar.

## Backtest CLI

```bash
# Demo veri ile hızlı deneme
python backtest.py --demo-data --developer-mode

# Kendi CSV'in ile (header: league,bet_type,predicted_prob,outcome)
python backtest.py --input backtests/history.csv --developer-mode \
  --calibration-map calibration_profiles.json --report backtest_report.json
```

- `--developer-mode`: Lig/bahis bazlı algoritma/kalibrasyonu aktifleştirir.
- `--calibration-map`: JSON sözlüğü; anahtarlar lig ve bahis türüne göre iç içe
  kalibrasyon seçimi yapar. Geliştirici mod kapalıyken varsayılan tekil
  kalibrasyon kullanılır ve karşılaştırmalı sonuçlar raporlanır.
- `--report`: Backtest özetini JSON olarak kaydeder.

Geliştirici mod açıkken çıktı, lig & bahis kırılımını da içererek hangi
kalibrasyonların kullanıldığını doğrulamanıza yardımcı olur.
