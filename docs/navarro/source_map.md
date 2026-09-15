# Source map — protocol 1.0

These are adaptations to neural training/inference, not reproductions of Navarro neural-network experiments. Download date: 2026-09-13. PDFs retained locally under `experiments/navarro/sources/`, excluded from git.

| ID | Source and fixed PDF SHA-256 | Applied section | Own implementation / difference |
|---|---|---|---|
| N1 | [González, Grabowski, Mäkinen, Navarro (2005)](https://users.dcc.uchile.cl/~gnavarro/ps/wea05.pdf), author-hosted 11-page PDF, `aba794614bba1d45701ce948ffd70acfce9f1ee5255a17a4c3949ece799d1e8d` | §1.3, pp. 4–5 | `bitmap.PackedPositive.rank`: exclusive rank, 32-bit words, sampled counters every 256 bits. Positive activation payload and segmented VJP are our adaptation. |
| N2 | [Navarro, Providel (SEA 2012)](https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf), author-hosted 12-page PDF, `a48c4313adb19ff80ad135e1b180dc3372cde7974b24a2c4d5055f01174c1ccc` | Introduction, rank/select space tradeoffs | Operation selection only: access and rank; no select. |
| N3 | [Ferragina et al. (2022)](https://arxiv.org/pdf/2203.14540), arXiv:2203.14540v2, 30 March 2022, 26 pages, `1b2021f512e66ec1a935451fee9ebd7597b83abf516766e05d54f01bf9a8846e` | §2, §3.1 Thm 3.4, §3.2 Thm 3.10, §4 | `grammar_matrix.encode/decode_exact/matvec`, `native.cpp`: CSRV, delimiter-preserving RePair, physical re_32/re_iv, direct DAG evaluation. FP32 bit identity, row blocks and adaptive RAW are study adaptations. |
| N4 | [Navarro (2025)](https://users.dcc.uchile.cl/~gnavarro/ps/spe25.pdf), author-hosted 24-page manuscript with production placeholders, `571e8cf527e54e17d94a9d23ee009206ee752b3644366a1cb75b2c2d8c8c18b1` | Abstract, §1, §6.4 | Explicit exclusion of dynamic tree: saved activations are short-lived immutable bitmaps. |

Original v1 campaign (historical record): N3 upstream code metadata accessed at GitLab API, revision `edd85fa3193dd1e6e60e1d4886f1ad8ded2a138f`. License inspected: Apache License 2.0 (`LICENSE.md`, retained with sources). Current implementation is independently written from the publication; no author code copied and no claim of validation against the author binary or reproduction of published benchmarks. The compiled compressor uses repeated frequency scans rather than the authors' linear-time implementation; this limitation must accompany performance conclusions.

Repository SHA equals the audited `b54b9fc139f455a9a5e60dc9a688ca9dbdb22944`. Native `gpt.py`, `optim.py`, `train.py`, `dataloader.py`, `engine.py`, `eval.py` and `common.py` remain the mathematical reference. Experimental schedules reproduce batch LR and weight-decay scaling in `train.py`. No changes to the scientific acceptance thresholds.

## Follow-up source review, 2026-09-14

The deeper [MM RePair assessment](mm-repair-assessment.md) found per-file license notices that the initial root-license inspection did not cover: the repository root declares Apache 2.0, while `brepair/irepair0.c` and supporting files declare GPL version 2 or later. The original v1 codec remains independently written. The follow-up compiled external author executables under ignored `sources/` and ran a separate bounded CPU diagnostic; it did not replace the v1 implementation or its evidence. This distinction corrects the scope of the earlier license review and the earlier statement that no author-binary comparison had yet been made.

## Hybrid implementation follow-up

The subsequent [hybrid implementation](hybrid-repair.md) independently implements incremental pair maintenance and an exact dictionary-and-singleton size rejection bound while keeping the existing packed format, deterministic pair order and FP32 products. It incorporates no upstream source. The earlier frequency-scan description applies to the archived v1 campaign; new comparisons identify that baseline as `7787ae2` and preserve snapshots of both implementations. The lazy heap implementation does not claim the authors' linear complexity.
