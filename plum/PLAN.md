# End-to-End Machine Learning Engineering Plan (`plum/PLAN.md`)

## 1. Philosophy & Standard ML Workflow
Instead of treating model development as a static checklist, we follow **industry-standard / Kaggle Grandmaster ML engineering practices** driven by empirical diagnostics, error analysis, and covariate-shift adaptation:
1. **Exploratory Data Analysis & Covariate Shift Audit** (`plum/feature_analysis.ipynb` — **Completed**)
2. **Baseline & Multi-View Representation Benchmarking** (`plum/model_selection.ipynb` — **Completed**)
3. **Diagnostic Error & Subgroup Analysis** (Diagnosing *where* and *why* the baseline ensemble fails across genres, lengths, and adversarial edits)
4. **Domain-Conditioned Feature Engineering & Level-1 Out-of-Fold (OOF) Stacking** (Replacing unsupervised SVD with supervised multi-view OOF log-odds projections + genre-normalized stylometrics)
5. **Covariate Shift Adaptation (Importance Weighting) & Semi-Supervised Learning** (Adapting to `test.csv`'s literary/poetry shift and leveraging `unlabeled.csv`)
6. **Full 5-Fold Cross-Validated Final Ensemble & Calibrated Submission Generation** (`plum/submission.csv`)

---

## 2. Progress & Milestones

### [x] Stage 1: Exploratory Data Analysis & Shift Discovery (`plum/feature_analysis.ipynb`)
- Audited $10,000$ `train.csv`, $3,000$ `unlabeled.csv`, and $3,000$ `test.csv` passages.
- Engineered 77 interpretable stylometric, structural, readability, syntactic, and discourse features.
- **Discovery 1 (Formatting Trap)**: $0\%$ of `train` Human texts have `\n\n` vs. $66\%$ of `train` AI texts. Relying on raw `\n\n` causes catastrophic failure when formatting changes.
- **Discovery 2 (Bimodal AI Lexical Diversity)**: AI `ttr` has $3\times$ higher standard deviation ($0.199$ vs. $0.068$) than Human text, occupying both ultra-repetitive loops and ultra-dense vocabulary lists.
- **Discovery 3 (Covariate Shift in `test.csv`)**: `unlabeled.csv` is IID with `train.csv` ($\text{Adversarial AUC} = 0.543$), whereas `test.csv` has substantial domain shift ($\text{Adversarial AUC} = 0.744$) driven by **Poetry/Verse ($0.1\% \to 14.4\%$)** and **Book/Novel Summaries ($4.0\% \to 13.9\%$)**.

### [x] Stage 2: Multi-View Representation & Baseline Model Selection (`plum/model_selection.ipynb`)
- Built a 3-view representation: Formatting-Normalized Word $(1,3)$ TF-IDF + Character `char_wb` $(3,5)$ TF-IDF + 72 Pure & Bimodal Deviation Stylometric features.
- Demonstrated that a naive LightGBM model (`Model D0`) collapses from $0.9985 \to 0.8109$ AUROC when line breaks are stripped, whereas our formatting-free **Model E (Hybrid Linear + GBDT Ensemble)** achieves **$0.9837$ Val AUROC**, **$0.9804$ Formatting-Stripped AUROC**, and **$0.9859$ Test-Weighted AUROC**.

### [x] Stage 3: Deep Error Analysis, OOF Stacking & Domain Adaptation (`plum/error_analysis_and_stacking.ipynb` — **Completed**)
Following standard ML best practices, we executed four tightly coupled steps:

1. **Step 3.1 — Granular Subgroup & Failure Mode Analysis**:
   - Evaluated our Stage 2 ensemble across **observable writing domains** (`Recipe`: $0.9958$, `First-Person/Social`: $0.9883$, `Academic/ArXiv`: $0.9852$, `General/News`: $0.9749$, `Wiki Bio`: $0.9733$, `Film/TV`: $0.9690$, `Book/Novel Summary`: $0.9670$) and **word-count bins** (`Short 75–165w`: $0.9842$, `Medium 166–245w`: $0.9788$, `Long 246–350w`: $0.9886$).
   - Inspected the **top 5 False Positives (`id=5227, 3660, 2017, 7182, 7562`) and top 5 False Negatives (`id=3062, 1188, 4519, 2674, 1962`)**, discovering that False Positives cluster in formulaic/enumerative Wikipedia stubs or non-native repetitive abstracts, while False Negatives occur in casual Reddit-style AI posts (lowercase `"i"`, `>` quotes, curly apostrophes) and financial/geopolitical wire briefs.

2. **Step 3.2 — Error-Driven Feature Engineering (Domain-Relative Z-Scores & True Character N-Grams)**:
   - Engineered **24 Genre-Conditioned Stylometric Z-Scores** ($z_{i,g} = (x_i - \mu_{g,\text{Human}}) / \sigma_{g,\text{Human}}$) plus L2/Max anomaly norms (`genre_z_l2_norm`, `genre_z_max_abs`), expanding our pure stylometric representation to **96 features**.
   - Added a **True Cross-Word Character `(3,5)` TF-IDF View (`analyzer='char'`)** that spans spaces and punctuation boundaries (`0.9511` Val AUROC on its own vs. `0.9369` for `char_wb`).

3. **Step 3.3 — True Level-1 Out-of-Fold (OOF) Stacking Architecture**:
   - Replaced unsupervised `TruncatedSVD` with **5-fold Out-of-Fold (OOF) supervised log-odds meta-features** from **7 diverse Level-1 base learners** (`L1_Word_LogReg`, `L1_Word_LinearSVC`, `L1_CharWB_LogReg`, `L1_CharCross_LogReg`, `L1_AllNGram_LinearSVC`, `L1_AllNGram_CompNB`, and `L1_PureGenreZ_Stylo_LGBM`).
   - Improved held-out Validation AUROC from **$0.9837 \to 0.9856$**, Formatting-Stripped AUROC from **$0.9804 \to 0.9829$**, Test-Weighted AUROC to **$0.9863$**, and reduced Brier calibration error by **20% ($0.0519 \to 0.0417$)**, cutting total validation classification errors from $144 \to 123$.

4. **Step 3.4 — Learning Curves, Semi-Supervised Pseudo-Labeling (`unlabeled.csv`), and Full 10,000-Row Kaggle Submission (`plum/submission.csv`)**:
   - Evaluated sample complexity across $N \in \{500, 1000, 2000, 5000, 8000\}$ training rows ($0.9554 \to 0.9675 \to 0.9746 \to 0.9803 \to 0.9834$).
   - Applied **Transductive 3-View TF-IDF Vocabulary Expansion + High-Confidence Calibrated Pseudo-Labeling** ($1,962 / 3,000$ unlabeled passages with $\hat{P} \ge 0.95$ or $\le 0.05$, weight $w=0.5$).
   ### [x] Stage 4: Leaderboard Optimization — Out-of-Domain (OOD) Diagnostics, Topic-Masked Syntax & Convex Logit Fusion (`plum/leaderboard_boost.ipynb` — **Completed**)
- Following our initial Kaggle submission (`0.93948` Public Leaderboard AUROC), we built a **Leave-Two-Domains-Out OOD Benchmark** ($1,092$ unseen `Book / Novel Summary` + `Film / TV / Review` passages) to diagnose the gap between in-distribution validation ($0.9856$) and the unseen genres in `test.csv`.
- **Three Key Leaderboard Upgrades**:
  1. **Topic-Masked Syntactic & Morphological Skeleton (`View 4`, `ngram_range=(1,4)`)**: Replaced content nouns/verbs/adjectives with capitalization + morphological suffix tokens (`_T_ing`, `_w_tion`, `_w_ly`, `_NUM_`) while preserving all function words, pronouns, prepositions, AI transitions, and punctuation.
  2. **Regularized Damped-Structural + Stylometric LightGBM (`colsample_bytree=0.45`)**: Restored `log1p`-damped paragraph/stanza structure (`double_newline_count`, `starts_with_quote`, `has_bullet_list`, `double_space_rate`) alongside our 96 Pure + Bimodal + Genre-Z stylometric features with strong feature subsampling so trees combine structural and linguistic cues without overfitting to a single feature.
  3. **Strictly Positive Convex Logit Fusion (`0.70 Formatting-Free Multi-View + 0.30 Regularized Structural+Stylo GBDT` + Transductive SSL)**: Eliminated negative collinear weight cancellation across linear models, boosting **Unseen-Domain OOD AUROC from $0.9517 \to 0.9906$** (cutting OOD Brier error from $0.0886 \to 0.0410$) and achieving **$0.9977$ 10,000-row 5-fold OOF AUROC**.
- Generated the upgraded Kaggle submission in **`plum/submission.csv`** (also saved to **`plum/submission_v2_ood_boosted.csv`**).
