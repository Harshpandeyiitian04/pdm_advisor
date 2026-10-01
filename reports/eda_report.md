# C-MAPSS training data exploratory analysis

This report uses the training split only. Sensor trends are per-engine Spearman correlations with cycle, then summarized across engines; they are exploratory, not causal.

## Engine lifetime

- Engines: 100
- Cycles: minimum 128, median 199.0, mean 206.3, maximum 362
- Population standard deviation: 46.1 cycles

## Constant sensors/settings removed by preprocessing

op3, s1, s5, s10, s16, s18, s19

## Strongest sensor trends with cycle

| Sensor | Median within-engine Spearman correlation | Engines with valid correlation | Engines trending up |
| --- | ---: | ---: | ---: |
| s11 | 0.819 | 100 | 100.0% |
| s12 | -0.808 | 100 | 0.0% |
| s4 | 0.784 | 100 | 100.0% |
| s7 | -0.774 | 100 | 0.0% |
| s8 | 0.768 | 100 | 100.0% |
| s13 | 0.762 | 100 | 100.0% |
| s9 | 0.739 | 100 | 71.0% |
| s15 | 0.721 | 100 | 100.0% |
| s21 | -0.708 | 100 | 0.0% |
| s20 | -0.706 | 100 | 0.0% |
