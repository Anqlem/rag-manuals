# Retrieval results

_Run 2026-11-20 21:44 UTC · embeddings `local:BAAI/bge-small-en-v1.5` · reranker `local` · chunk 500/50 tokens · candidates 40_

28 answerable questions. A hit means a retrieved chunk overlaps a gold page.

| Chunking | Retrieval | hit@1 | hit@3 | hit@5 (95% CI) | MRR@10 | page recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fixed | vector | 0.46 | 0.68 | 0.75 (0.57–0.89) | 0.59 | 0.68 | 23 | 24 |
| fixed | hybrid | 0.39 | 0.54 | 0.61 (0.43–0.79) | 0.50 | 0.58 | 37 | 45 |
| fixed | hybrid_rerank | 0.43 | 0.71 | 0.75 (0.61–0.89) | 0.59 | 0.67 | 2581 | 2751 |
| section | vector | 0.57 | 0.68 | 0.68 (0.50–0.82) | 0.63 | 0.61 | 25 | 27 |
| section | hybrid | 0.36 | 0.46 | 0.57 (0.39–0.75) | 0.45 | 0.55 | 39 | 47 |
| section | hybrid_rerank **(best)** | 0.50 | 0.79 | 0.89 (0.79–1.00) | 0.66 | 0.84 | 2421 | 2624 |

## Baseline (fixed/vector) vs best (section/hybrid_rerank)

- Fixed at @5: 5 ['s7_11', 's7_12', 's7_22', 's7_23', 's7_24']
- Broken at @5: 1 ['s7_10']

## Misses for the best config (3)

Read these before tuning anything. Each one is a failure-analysis candidate.

- **s7_10** How much 24 V current can a 1214C provide for sensors?  
  gold: s71200_system_manual_en-US_en-US.pdf p.[857] · got: s71200_system_manual_en-US_en-US.pdf p.882, s71200_system_manual_en-US_en-US.pdf p.986-989, s71200_system_manual_en-US_en-US.pdf p.864-865
- **s7_13** We shut the line down for three weeks and afterwards the PLC clock was wrong. Why, and how do I stop it happening?  
  gold: s71200_system_manual_en-US_en-US.pdf p.[104, 858, 962] · got: s71200_system_manual_en-US_en-US.pdf p.81-83, s71200_system_manual_en-US_en-US.pdf p.125-126, s71200_system_manual_en-US_en-US.pdf p.135-136
- **s7_21** I need pulse outputs for a stepper on a 1214C AC/DC/Relay. What do I have to add?  
  gold: s71200_system_manual_en-US_en-US.pdf p.[858] · got: s71200_system_manual_en-US_en-US.pdf p.872, s71200_system_manual_en-US_en-US.pdf p.986-989, s71200_system_manual_en-US_en-US.pdf p.442-443
