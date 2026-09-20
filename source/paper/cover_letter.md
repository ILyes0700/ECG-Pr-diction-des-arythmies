# Cover letter (a adapter)

Dear Editor,

We submit our manuscript *"Interpretable and Well-Calibrated Inter-Patient ECG Arrhythmia
Classification with Rhythm-Aware Deep Learning and Clinical Utility Assessment"* for
consideration in <JOURNAL>.

Most published beat classifiers report accuracies above 99 % on the MIT-BIH database under an
intra-patient protocol, in which beats from the same recording appear in both training and test
sets. Such results do not transfer to new patients. Our work differs in three ways:

1. **Strict inter-patient evaluation** following the AAMI EC57 DS1/DS2 division, with paced
   records excluded, patient-level bootstrap confidence intervals and McNemar significance tests.
2. **A clinical utility layer** rarely found in this literature: likelihood ratios, diagnostic
   odds ratio, prevalence-adjusted predictive values, false-alarm burden per hour, episode-level
   detection with latency, decision-curve net benefit, probability calibration, and a
   selective-prediction reject option enabling safe deferral to a cardiologist.
3. **Full reproducibility**: modular open-source code, unit tests on the evaluation logic, fixed
   seeds, cached datasets, and external validation on the INCART database without retraining.

The work has not been published elsewhere and is not under consideration by another journal.
All authors approved the submission and declare no conflict of interest.

Sincerely,
Ilyes Rejeb
