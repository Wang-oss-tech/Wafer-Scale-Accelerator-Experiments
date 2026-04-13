import math

# Formula-based MeshGEMM model derived from meshgemm.csl.
#
# Structure:
# 1. Initial X skew startup cost
# 2. P iterations of compute + exposed communication
#
# The compute term is intentionally kept the same as the SUMMA model.
#
# Communication is derived from comm_pe.csl:
# - startup uses repeated X-only @mov16 shifts
# - steady state launches X and Y @mov16 transfers concurrently
# - measured runtime suggests a substantial exposed communication cost remains
#   visible on top of compute, so we model it additively per step
# - there is no per-step route reconfiguration


# ---------------------- Tunable Constants ---------------------- #

# Exposed per-step communication term:
# T_comm_exposed = COMM_EXPOSED_FIXED + COMM_EXPOSED_ALPHA * (X_elems + W_elems)
#
# These defaults are a simple first-order approximation from the MeshGEMM
# measurements. Keep them explicit so they are easy to retune.
COMM_EXPOSED_FIXED = 170.0
COMM_EXPOSED_ALPHA = 0.5

# Initial X-shift cost model.
SHIFT_FIXED = 0.0
SHIFT_HOP_LATENCY = 1.0

# Startup can be modeled for the slowest tile or an average tile.
STARTUP_MODE = "worst"   # "worst" or "average"

DEFAULT_BENCHMARK_CASES = [
      (180, 12, 12, 12),
      (360, 6, 6, 6),
      (540, 4, 4, 4),
      (720, 3, 3, 3),
      (360, 12, 12, 12),
      (540, 8, 8, 8),
      (720, 6, 6, 6),
      (360, 24, 24, 24),
      (540, 16, 16, 16),
      (720, 12, 12, 12),
]


# ---------------------- Compute Model ---------------------- #

def fmach_issue_period(Mt):
      # Effective FMACH initiation spacing for this kernel.
      #
      # This is not a hard 5-cycle hardware floor. The lower bound captures
      # the observed spacing of the current wave-scheduled implementation for
      # small Mt, where leftover SIMD waves still introduce issue spacing.
      #
      # The structural term 2 * ceil(Mt / 8) - 1 approximates the number of
      # 8-lane SIMD waves needed for one FMACH step. The max preserves the
      # empirical lower bound used by the current implementation.
      return max(5, 2 * math.ceil(Mt / 8) - 1)

def compute_iter(Mt, Kt, Nt):
      fmach = fmach_issue_period(Mt)
      outer_overhead = 40
      return Kt * (Nt - 1) * fmach + (Kt - 1) * outer_overhead


# ---------------------- Communication Helpers ---------------------- #

def even_fp16_count(size):
      # comm_pe.csl uses:
      # const _Mt_Kt = ((Mt * Kt) / 2) * 2
      # const _Kt_Nt = ((Kt * Nt) / 2) * 2
      # because transfers are issued with @mov16 over an even fp16 extent.
      return (size // 2) * 2


def x_elems(Mt, Kt):
      return even_fp16_count(Mt * Kt)


def w_elems(Kt, Nt):
      return even_fp16_count(Kt * Nt)


def x_shift_steps_for_row(py):
      # Matches meshgemm.csl:
      # even py -> py/2, odd py -> (py+1)/2
      return (py + 1) // 2


def startup_shift_steps(P, mode):
      if P <= 0:
            return 0.0
      if mode == "average":
            total = 0
            for py in range(P):
                  total += x_shift_steps_for_row(py)
            return total / P
      return x_shift_steps_for_row(P - 1)


def startup_shift_cycles(P, Mt, Kt, mode=STARTUP_MODE):
      # Each startup shift is one X-only @mov16 send/recv pair joined by x_shift_finish.
      # Since send and recv are symmetric and issued asynchronously, model one shift
      # as the critical-path X transfer rather than a sum of send and recv.
      return startup_shift_steps(P, mode) * (SHIFT_FIXED + x_elems(Mt, Kt) + SHIFT_HOP_LATENCY)


def comm_exposed_iter(Mt, Kt, Nt):
      return COMM_EXPOSED_FIXED + COMM_EXPOSED_ALPHA * (x_elems(Mt, Kt) + w_elems(Kt, Nt))


# ---------------------- Total Runtime ---------------------- #

def kernel_cycles(P, Mt, Kt, Nt):
      if P <= 0:
            return 0.0

      startup = startup_shift_cycles(P, Mt, Kt)
      compute = compute_iter(Mt, Kt, Nt)
      comm_exposed = comm_exposed_iter(Mt, Kt, Nt)

      return startup + P * (compute + comm_exposed)


def case_row(P, Mt, Kt, Nt):
      return {
            "P": P,
            "Mt": Mt,
            "Kt": Kt,
            "Nt": Nt,
            "x_elems": x_elems(Mt, Kt),
            "w_elems": w_elems(Kt, Nt),
            "startup_shift_steps": startup_shift_steps(P, STARTUP_MODE),
            "startup_shift_cycles": startup_shift_cycles(P, Mt, Kt),
            "compute_per_iter": compute_iter(Mt, Kt, Nt),
            "comm_exposed_per_iter": comm_exposed_iter(Mt, Kt, Nt),
            "kernel_cycles": kernel_cycles(P, Mt, Kt, Nt),
      }


def write_csv(path, rows):
      import csv

      fieldnames = [
            "P",
            "Mt",
            "Kt",
            "Nt",
            "x_elems",
            "w_elems",
            "startup_shift_steps",
            "startup_shift_cycles",
            "compute_per_iter",
            "comm_exposed_per_iter",
            "kernel_cycles",
      ]

      with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
      import argparse

      parser = argparse.ArgumentParser(description="MeshGEMM performance model")
      parser.add_argument("--P", type=int, default=4)
      parser.add_argument("--Mt", type=int, default=14)
      parser.add_argument("--Kt", type=int, default=14)
      parser.add_argument("--Nt", type=int, default=14)
      parser.add_argument("--dump-default-csv", type=str, default="")
      args = parser.parse_args()

      if args.dump_default_csv:
            rows = [case_row(P, Mt, Kt, Nt) for (P, Mt, Kt, Nt) in DEFAULT_BENCHMARK_CASES]
            write_csv(args.dump_default_csv, rows)
            print(f"Wrote CSV: {args.dump_default_csv}")
            raise SystemExit(0)

      P, Mt, Kt, Nt = args.P, args.Mt, args.Kt, args.Nt

      # print(f"Configuration: P={P}, Mt={Mt}, Kt={Kt}, Nt={Nt}")
      # print(f"Compute per iter: {compute_iter(Mt, Kt, Nt)}")
      # print(f"X fp16 elems per iter: {x_elems(Mt, Kt)}")
      # print(f"W fp16 elems per iter: {w_elems(Kt, Nt)}")
      # print(f"Startup shift steps ({STARTUP_MODE}): {startup_shift_steps(P, STARTUP_MODE)}")
      # print(f"Startup shift cycles: {startup_shift_cycles(P, Mt, Kt)}")
      # print(f"Exposed comm per iter: {comm_exposed_iter(Mt, Kt, Nt)}")
      print(f"{kernel_cycles(P, Mt, Kt, Nt)}")
