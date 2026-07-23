#!/usr/bin/env python3
"""
bench_uninavid.py — Misura il tempo di inferenza di Uni-NaVid.

Uni-NaVid e' una video-policy STATEFUL: ogni act() appende un frame alla
video-memory interna (nav_feat_cache) e predice un chunk di ~4 azioni. La
latenza NON e' un numero singolo: cresce con la profondita' della memoria.
Questo script quindi non misura solo la media, ma la CURVA latenza-vs-step.

Cosa fa, in ordine:
  1. carica l'agent (stesso costruttore usato dal nodo)
  2. warmup: reset() + qualche act() per compilare i kernel CUDA (scartati)
  3. reset() -> memoria vuota
  4. loop cronometrato: N act() con frame sintetici, misura ogni chiamata;
     la profondita' della memoria = indice dello step (cresce di 1 per act)
  5. stampa media/percentili + drift (primo terzo vs ultimo terzo) e salva CSV

Timing corretto: torch.cuda.Event + synchronize (time.time() e' sbagliato per
CUDA async). L'agent gia' usa torch.inference_mode() dentro predict_inference.

USO (dentro il container/env del nodo, col modello gia' scaricato):
    python bench_uninavid.py --steps 60 --warmup 8 --out uninavid_pcpilot.csv

NOTE:
  * Il contenuto dei frame non influenza la latenza, solo la shape: uso rumore.
  * act() usa do_sample=True (temperature=0.5): la lunghezza dell'output varia,
    quindi la latenza ha una componente di rumore reale -> la coprono i percentili.
  * Per misurare act() servono UNINAVID_MODEL_PATH / UNINAVID_REPO_DIR come in deploy.
"""

import argparse
import os
import statistics
import time

import numpy as np

# Stesso import del nodo: posiziona questo file dove sta uninavid_node.py.
from third_party.uni_navid_agent import UniNaVid_Agent

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _CUDA = False


# --------------------------------------------------------------------------- #
# Timing di una singola chiamata (ms)                                          #
# --------------------------------------------------------------------------- #
def time_call(fn) -> float:
    if _CUDA:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)          # gia' in ms
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def summarize(samples):
    s = sorted(samples)
    n = len(s)

    def pct(p):
        if n == 1:
            return s[0]
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    mean = statistics.fmean(s)
    return {
        "n": n,
        "mean_ms": round(mean, 2),
        "std_ms": round(statistics.pstdev(s), 2) if n > 1 else 0.0,
        "median_ms": round(statistics.median(s), 2),
        "p90_ms": round(pct(0.90), 2),
        "p99_ms": round(pct(0.99), 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
        "hz": round(1000.0 / mean, 2) if mean > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Risoluzione del checkpoint (config.json puo' stare in una sottocartella)     #
# --------------------------------------------------------------------------- #
def resolve_ckpt(model_path: str) -> str:
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path
    subs = [os.path.join(model_path, d) for d in sorted(os.listdir(model_path))
            if os.path.isfile(os.path.join(model_path, d, "config.json"))]
    if not subs:
        raise FileNotFoundError(f"Nessun config.json in {model_path} o sottocartelle")
    return subs[0]


def main():
    ap = argparse.ArgumentParser(description="Benchmark tempo di inferenza Uni-NaVid.")
    ap.add_argument("--model-path", default=os.path.join(
        os.environ.get("UNINAVID_MODEL_PATH", "/models"), "uni_navid_model"))
    ap.add_argument("--repo-dir", default=os.environ.get("UNINAVID_REPO_DIR", ""),
                    help="se presente, chdir in <repo-dir>/UniNaVid come fa il nodo")
    ap.add_argument("--steps", type=int, default=60, help="act() cronometrati (= profondita' max)")
    ap.add_argument("--warmup", type=int, default=8, help="act() di warmup scartati")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--instruction", default="reach the blue chair")
    ap.add_argument("--out", default="", help="salva la curva (step,depth,latency_ms) in CSV")
    args = ap.parse_args()

    if args.repo_dir:
        os.chdir(os.path.join(args.repo_dir, "UniNaVid"))

    print("=" * 64)
    print("Uni-NaVid inference benchmark")
    if _CUDA:
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
        print(f"  torch / CUDA      : {torch.__version__} / {torch.version.cuda}")
    else:
        print("  ATTENZIONE: CUDA non disponibile -> timing su CPU (perf_counter)")
    print(f"  steps / warmup    : {args.steps} / {args.warmup}")
    print(f"  frame             : {args.height}x{args.width} (RGB sintetico)")
    print("=" * 64)

    ckpt = resolve_ckpt(args.model_path)
    print(f"Carico l'agent da: {ckpt}")
    agent = UniNaVid_Agent(ckpt)

    def synth_frame():
        # RGB uint8: e' cio' che act() si aspetta (il nodo converte BGR->RGB prima)
        return np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    def one_act():
        agent.act({"instruction": args.instruction, "observations": synth_frame()})

    # --- warmup: compila i kernel, poi si scarta tutto con reset() ---
    print("Warmup...")
    agent.reset()
    for _ in range(args.warmup):
        one_act()
    agent.reset()                                  # memoria di nuovo vuota
    if _CUDA:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- loop cronometrato: la profondita' cresce di 1 a ogni act ---
    print("Misura in corso...\n")
    print(f"{'step':>5} | {'depth':>5} | {'latency_ms':>11}")
    print("-" * 28)
    rows = []
    for i in range(args.steps):
        lat = time_call(one_act)
        depth = i + 1                              # frame accumulati nella memoria
        rows.append((i, depth, lat))
        print(f"{i:>5} | {depth:>5} | {lat:>11.2f}")

    lat_all = [r[2] for r in rows]
    third = max(1, len(lat_all) // 3)
    first_third = summarize(lat_all[:third])
    last_third = summarize(lat_all[-third:])
    overall = summarize(lat_all)

    print("\n" + "=" * 64)
    print("RISULTATO (Uni-NaVid — latenza cresce con la profondita')")
    print(f"  media globale     : {overall['mean_ms']} ms  ({overall['hz']} Hz)")
    print(f"  std / mediana     : {overall['std_ms']} / {overall['median_ms']} ms")
    print(f"  p90 / p99         : {overall['p90_ms']} / {overall['p99_ms']} ms")
    print(f"  min / max         : {overall['min_ms']} / {overall['max_ms']} ms")
    print(f"  media primo terzo : {first_third['mean_ms']} ms  (depth ~1..{third})")
    print(f"  media ultimo terzo: {last_third['mean_ms']} ms  (depth ~{args.steps-third}..{args.steps})")
    drift = last_third['mean_ms'] - first_third['mean_ms']
    print(f"  DRIFT             : {drift:+.2f} ms dal primo all'ultimo terzo")
    if _CUDA:
        print(f"  picco memoria     : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("=" * 64)

    if args.out:
        with open(args.out, "w") as f:
            f.write("step,depth,latency_ms\n")
            for i, depth, lat in rows:
                f.write(f"{i},{depth},{lat:.3f}\n")
        print(f"Curva salvata in {args.out} (colonne: step,depth,latency_ms)")


if __name__ == "__main__":
    main()