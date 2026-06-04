import json

with open(r'C:\truthlens\TruthLens-Backend\benchmark_compare_result.json', encoding='utf-8') as f:
    data = json.load(f)

records = data['records']
W = 72

print()
print("=" * W)
print("  Threshold 탐색  —  V1 vs V2")
print("=" * W)
print(f"  {'thr':>6} | {'V1 Acc':>7} {'V1 F1':>7} | {'V2 Acc':>7} {'V2 F1':>7} {'V2 Rec':>7} {'V2 Prec':>8}")
print("-" * W)

best_v2_f1, best_v2_thr = 0, 0.5

for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    res = {}
    for model in ['v1', 'v2']:
        key = f'{model}_prob'
        tp = sum(1 for r in records if r['label'] == 'FAKE' and r[key] >= thr)
        fp = sum(1 for r in records if r['label'] == 'REAL' and r[key] >= thr)
        fn = sum(1 for r in records if r['label'] == 'FAKE' and r[key] <  thr)
        tn = sum(1 for r in records if r['label'] == 'REAL' and r[key] <  thr)
        acc  = (tp + tn) / len(records)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        res[model] = dict(acc=acc, prec=prec, rec=rec, f1=f1)

    if res['v2']['f1'] > best_v2_f1:
        best_v2_f1  = res['v2']['f1']
        best_v2_thr = thr

    marker = "  <- 현재" if abs(thr - 0.5) < 0.001 else ""
    if res['v2']['f1'] == best_v2_f1 and abs(thr - best_v2_thr) < 0.001:
        marker = "  <- V2 최적"

    print(
        f"  {thr:>6.2f} | "
        f"{res['v1']['acc']*100:>6.1f}% {res['v1']['f1']*100:>6.1f}% | "
        f"{res['v2']['acc']*100:>6.1f}% {res['v2']['f1']*100:>6.1f}% "
        f"{res['v2']['rec']*100:>6.1f}% {res['v2']['prec']*100:>7.1f}%"
        f"{marker}"
    )

print("=" * W)

# V2 확률 분포
v2_fake_probs = [r['v2_prob'] for r in records if r['label'] == 'FAKE']
v2_real_probs = [r['v2_prob'] for r in records if r['label'] == 'REAL']

print()
print("  V2 확률 분포 (히스토그램)")
print(f"  {'구간':>12}  {'FAKE':>8}  {'REAL':>8}")
print("-" * 40)
buckets = [(i/10, (i+1)/10) for i in range(10)]
for lo, hi in buckets:
    f_cnt = sum(1 for p in v2_fake_probs if lo <= p < hi)
    r_cnt = sum(1 for p in v2_real_probs if lo <= p < hi)
    bar_f = "█" * (f_cnt // 5)
    bar_r = "█" * (r_cnt // 5)
    print(f"  {lo:.1f}~{hi:.1f}  FAKE:{f_cnt:>4}  {bar_f}")
    print(f"         REAL:{r_cnt:>4}  {bar_r}")

# V1 확률 분포
v1_fake_probs = [r['v1_prob'] for r in records if r['label'] == 'FAKE']
v1_real_probs = [r['v1_prob'] for r in records if r['label'] == 'REAL']

print()
print("  V1 확률 분포 (히스토그램)")
print(f"  {'구간':>12}  {'FAKE':>8}  {'REAL':>8}")
print("-" * 40)
for lo, hi in buckets:
    f_cnt = sum(1 for p in v1_fake_probs if lo <= p < hi)
    r_cnt = sum(1 for p in v1_real_probs if lo <= p < hi)
    bar_f = "█" * (f_cnt // 5)
    bar_r = "█" * (r_cnt // 5)
    print(f"  {lo:.1f}~{hi:.1f}  FAKE:{f_cnt:>4}  {bar_f}")
    print(f"         REAL:{r_cnt:>4}  {bar_r}")
