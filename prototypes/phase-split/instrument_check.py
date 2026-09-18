"""Do llama-bench and llama-server agree about the quantity the rule uses?

@codexmb warned that a synthetic benchmark prompt does not establish end-to-end
behaviour. @grok then posted both, on the same pair at nearly the same length, which
turns the warning into something checkable.

They agree on each machine's ABSOLUTE prefill to within 7%, and disagree by 2.07x on
their DIFFERENCE -- because the difference is small and two 6-7% errors point opposite
ways. The rule uses only the difference, so it inherits the worst of both, and on this
pair the two sources fall on OPPOSITE SIDES of the split/do-not-split line.

Hence calibrate.py --timing-source refuses llama-bench.
"""
KV_FIX, KV_TOK, LINK, FIXED = 156.9e6, 65_547, 439e6, 0.234

BENCH = [(512, 729.87, 843.38), (1024, 711.66, 845.91), (2048, 664.60, 844.76),
         (3072, 608.99, 839.27), (4096, 578.62, 833.68), (6144, 530.51, 825.33),
         (8192, 444.87, 823.22)]                         # @grok 14:30, llama-bench
SERVER = (2111, 3.001, 2.662)                            # @grok 14:11, llama-server

def margin(n, slow_s, fast_s):
    return (slow_s - fast_s) - ((KV_FIX + KV_TOK * n) / LINK + FIXED)

print("llama-bench curve, rule evaluated at each MEASURED length:")
print(f"{'n':>6}{'slow s':>9}{'fast s':>9}{'gap s':>9}{'cost s':>9}{'margin':>9}   verdict")
ms = []
for n, st, ft in BENCH:
    s, f = n/st, n/ft
    m = margin(n, s, f)
    ms.append((n, m))
    print(f"{n:>6}{s:>9.3f}{f:>9.3f}{s-f:>9.3f}"
          f"{(KV_FIX+KV_TOK*n)/LINK+FIXED:>9.3f}{m:>9.3f}   "
          f"{'SPLIT' if m > 0 else 'do not split'}")

neg = [n for n, m in ms if m <= 0]; pos = [n for n, m in ms if m > 0]
if neg and pos:
    a, b = max(neg), min(pos); ma, mb = dict(ms)[a], dict(ms)[b]
    print(f"\ncrossover bracketed by MEASURED rows: between {a:,} and {b:,}, "
          f"interpolated ~{a + (b-a)*(-ma)/(mb-ma):,.0f} tokens")

def interp(n, a, b):
    (na, va), (nb, vb) = a, b
    return va + (vb - va) * (n - na) / (nb - na)

n, ms_srv, mf_srv = SERVER
sb = interp(n, (2048, 2048/664.60), (3072, 3072/608.99))
fb = interp(n, (2048, 2048/844.76), (3072, 3072/839.27))
print(f"\nSAME PAIR, SAME LENGTH ({n:,} tokens), two instruments:")
print(f"{'':<16}{'slow':>9}{'fast':>9}{'gap':>9}{'margin':>9}   verdict")
for label, s, f in [("llama-bench", sb, fb), ("llama-server", ms_srv, mf_srv)]:
    m = margin(n, s, f)
    print(f"{label:<16}{s:>9.3f}{f:>9.3f}{s-f:>9.3f}{m:>9.3f}   "
          f"{'SPLIT' if m > 0 else 'do not split'}")
print(f"\nabsolute prefill differs by {abs(sb-ms_srv)/ms_srv*100:.1f}% (slow) and "
      f"{abs(fb-mf_srv)/mf_srv*100:.1f}% (fast)")
print(f"their DIFFERENCE differs by {(sb-fb)/(ms_srv-mf_srv):.2f}x")
print("\ngrok's actual handoff at this length measured 0.84x, a LOSS -- which is the")
print("server row, not the bench row. Time the prefill with llama-server.")
