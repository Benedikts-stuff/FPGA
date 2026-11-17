## 1. Ergebnisse

  • **results/results_summary.md**: Vollständige Zusammenfassung aller PC-Spezifikationen, Trainingszeiten, Genauigkeiten, BOPS und Latenz-Werte.
  • **results/results_table.csv**: Strukturierte CSV-Tabelle für alle 4 Modelle.
  • **results/model_comparison.png**: 4-Panel-Vergleichsdiagramm (Genauigkeit, BOPS im Log-Scale, Hardware-Gewichtsspeicher, CPU-Latenz).
  • **results/kws_float32_accuracy.png**: Trainings- und Validierungskurven des FP32-Modells.
  • **results/kws_w8a8_accuracy.png**: Genauigkeitskurven des W8A8-Modells.
  • **results/kws_w4a4_accuracy.png**: Genauigkeitskurven des W4A4-Modells.
  • **results/kws_w2a2_accuracy.png**: Genauigkeitskurven des W2A2-Modells.
  ──────

## 2. Geänderten Dateien

### 1. model_float32.py

  • Klasse KWS_Float(Module):
      • **init**() (TODO 1): Implementierung eines 4-stufigen Convolutional Neural Networks für die (1, 10, 49) MFCC-Spektrogramme bestehend aus nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d(2), nn.Dropout(0.2) sowie einem abschließenden Classifier nn.
      Linear(1536 -> 256) und nn.Linear(256 -> 11).
      • forward(x) (TODO 2): Reshaping auf (-1, 1, 10, 49), Skalierung auf [-1, 1] via 2.0 * x - 1.0, Feature-Extraktion und Klassifikation.
  • Trainingsschleife & Tracking (Schritt 2):
      • Erfassung von Trainings- und Validierungsgenauigkeit je Epoche.
      • CosineAnnealingLR-Scheduler zur Optimierung der Konvergenz.
      • Speicherung der Trainingshistorie (.npy) und Plotten der Lernkurven (kws_float32_accuracy.png).
  • Export & Test:
      • Gerätekompatibler ONNX-Export via Brevitas (bo.export_qonnx).
      • Saubere Test-Set-Auswertung im Else-Zweig (--train weggelassen).

### 2. model_quant.py

  • Klasse KWS(Module):
      • **init**(weight_bit_width, act_bit_width) (TODO 1): Implementierung des quantisierten MLP-Netzwerks mit Brevitas QuantIdentity (8-Bit Input), QuantLinear (konfigurierbare Bitbreite w_bits), BatchNorm1d und QuantReLU (konfigurierbare Bitbreite
      a_bits).
      • forward(x) (TODO 2): Flattening des Inputs (490), Skalierung auf [-1, 1], 8-Bit-Quantisierung des Inputs und sequentielle Weiterleitung durch die quantisierten Schichten.
  • Flexibilität & Kompatibilität:
      • Automatische Erkennung der Quantisierungsstufe aus dem Dateinamen (kws_w8a8, kws_w4a4, kws_w2a2) oder über optionale CLI-Flags (--weight_bits, --act_bits).
      • CosineAnnealingLR-Scheduler.
      • Tracking von Trainings-/Validierungsgenauigkeit und automatisches Speichern von Plot und Historie.
      • Korrektur beim Laden des State-Dicts (model = model.to(device)), um CUDA/CPU-Device-Mismatches bei quantisierten Scale-Buffern zu verhindern.

### 3. speechcommands_dataset.py

  • Funktion **getitem**(idx):
      • Behebung eines Bugs in Zeile 14: Änderung von if transforms is not None: auf if self.transform is not None:. Verhindert TypeError: 'NoneType' object is not callable, falls ein Dataset ohne Transformation instanziiert wird.

### 4. test.py

  • Eingabe-Normalisierung (Zeile 92–95):
      • Automatisierte Prüfung der Rohdaten: Falls test_set.data als uint8 vorliegt (max > 1.0), wird automatisch durch 255 geteilt. Dadurch liefert test.py sowohl für unnormierte als auch für vorverarbeitete Datensätze exakte, verifizierte
      Genauigkeitswerte.

  ──────

## 3. Geforderte Tabellen & Werte der Aufgabenstellung

### Aufgabe 1.2: Standard FP32 Training

  • PC Specs: AMD Ryzen 9 5950X (16 Cores / 32 Threads), NVIDIA GeForce RTX 3090 (24 GB VRAM), 32 GB RAM
  • Schritt 4: Model size: 1.8 MB, Time: 34.79 s, Accuracy: 96.86% (Val)
  • Schritt 5: Accuracy: 96.30% (Test), Total BOPS: 11.969.036.288, Total Weight Bits: 14.746.624
  • Schritt 6: Avg. Time: 0.0199 s, Min. Time: 0.0171 s, Max. Time: 0.0621 s

### Aufgabe 1.3: Brevitas QAT (3 Quantisierungsstufen)

  • Schritt 4 (Training & Validierung):
      1. W8A8: Model size: 926 KB (.pt), Time: 39.54 s, Validation Accuracy: 89.84%
      2. W4A4: Model size: 926 KB (.pt), Time: 39.59 s, Validation Accuracy: 89.62%
      3. W2A2: Model size: 926 KB (.pt), Time: 38.94 s, Validation Accuracy: 85.78%
  • Schritt 5 (Inference Cost & Test Accuracy):
      1. W8A8: Accuracy: 87.66%, Total BOPS: 14.409.728, Total Weight Bits: 1.801.216
      2. W4A4: Accuracy: 87.44%, Total BOPS: 5.609.472, Total Weight Bits: 900.608
      3. W2A2: Accuracy: 84.49%, Total BOPS: 2.405.888, Total Weight Bits: 450.304
  • Schritt 6 (Latenz des besten QNN-Modells kws_w8a8):
      • Avg. Time: 0.0201 s, Min. Time: 0.0167 s, Max. Time: 0.0440 s (Test Accuracy: 87.66%)

### Aufgabe 1.4: Gesamtvergleich aller Modelle

   Modell                                   │              Test Accuracy               │          Hardware Weight Memory          │    Total BOPS                │         BOPS-Ersparnis vs. FP32         │              Avg. Latency
  ──────────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────
   FP32                                     │                  96.30%                  │  ~1.800 KB    │             11.969.036.288              │                  1.0 ×                  │   19.9 ms
   W8A8                                     │                  87.66%                  │   ~220 KB    │    14.409.728                │              830 × weniger              │   20.1 ms
   W4A4                                     │                  87.44%                  │   ~110 KB    │                5.609.472                │             2.133 × weniger             │   19.5 ms
   W2A2                                     │                  84.49%                  │    ~55 KB    │                2.405.888                │             4.975 × weniger             │   19.5 ms
