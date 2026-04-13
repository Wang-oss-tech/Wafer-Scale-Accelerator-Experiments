import math


def fmach_issue_period(Mt):
      # Effective FMACH initiation spacing for this kernel.
      #
      # This is not meant to represent a hard 5-cycle hardware floor.
      # Instead, the lower bound captures the observed issue spacing of the
      # current wave-scheduled kernel for small Mt, where leftover SIMD waves
      # still introduce nontrivial spacing even when Mt <= 8.
      #
      # The structural term 2 * ceil(Mt / 8) - 1 reflects the number of 8-lane
      # SIMD waves needed for one FMACH step. The max keeps the empirical
      # lower bound that matched the current implementation.
      return max(5, 2 * math.ceil(Mt / 8) - 1)

# Communication constants for the WaferLLM SUMMA collective kernel.
# This summa.csl launches one X broadcast and one Y broadcast, waits for both,
# then computes. Use an architecture-only communication model:
#
#   T_comm_step(k) = COMM_TASK_SYNC_FIXED
#                  + max(T_x_collective(k), T_y_collective(k))
#
#   T_x_collective(k) = X_COLLECTIVE_STARTUP
#                     + COMM_WORD_COST * X_words
#                     + X_HOP_COST * distance_from_root(P, k)
#
#   T_y_collective(k) = Y_COLLECTIVE_STARTUP
#                     + COMM_WORD_COST * Y_words
#                     + Y_HOP_COST * distance_from_root(P, k)
#
# X and Y broadcasts are launched together, so the step waits on the slower
# collective plus the local task synchronization cost.
X_COLLECTIVE_STARTUP = 640.0
Y_COLLECTIVE_STARTUP = 640.0
COMM_TASK_SYNC_FIXED = 26.0
COMM_WORD_COST = 1.0
X_HOP_COST = 0.0
Y_HOP_COST = 0.0


def compute_iter(Mt, Kt, Nt):
      fmach = fmach_issue_period(Mt)
      outer_overhead = 40  # per-kk cost: 3x load_to_dsr + 2x increment_dsd_offset
      return Kt * (Nt - 1) * fmach + (Kt - 1) * outer_overhead


def a_words(Mt, Kt):
      # Match summa.csl broadcast length: Mt * Kt / 2 with integer division.
      return (Mt * Kt) // 2


def b_words(Kt, Nt):
      # Match summa.csl broadcast length: Kt * Nt / 2 with integer division.
      return (Kt * Nt) // 2


def distance_from_root(P, k):
      return max(k, P - 1 - k)


def x_collective_cycles(P, Mt, Kt, k):
      return X_COLLECTIVE_STARTUP + COMM_WORD_COST * a_words(Mt, Kt) + X_HOP_COST * distance_from_root(P, k)


def y_collective_cycles(P, Kt, Nt, k):
      return Y_COLLECTIVE_STARTUP + COMM_WORD_COST * b_words(Kt, Nt) + Y_HOP_COST * distance_from_root(P, k)


def comm_iter(P, Mt, Kt, Nt, k):
      x_path = x_collective_cycles(P, Mt, Kt, k)
      y_path = y_collective_cycles(P, Kt, Nt, k)
      return COMM_TASK_SYNC_FIXED + max(x_path, y_path)


def kernel_cycles(P, Mt, Kt, Nt):
      compute = compute_iter(Mt, Kt, Nt)

      if P <= 0:
            return 0
      if P == 1:
            return comm_iter(P, Mt, Kt, Nt, 0) + compute

      # For this WaferLLM SUMMA kernel:
      # broadcast X/Y of step k, wait for both, then compute step k.
      total = 0.0
      for k in range(P):
            total += comm_iter(P, Mt, Kt, Nt, k) + compute
      return total


if __name__ == "__main__":
      import argparse

      parser = argparse.ArgumentParser(description="SUMMA fp16 optimized performance model (non-overlapping)")
      parser.add_argument("--P", type=int, default=4)
      parser.add_argument("--Mt", type=int, default=14)
      parser.add_argument("--Kt", type=int, default=14)
      parser.add_argument("--Nt", type=int, default=14)
      args = parser.parse_args()
      P, Mt, Kt, Nt = args.P, args.Mt, args.Kt, args.Nt

      # print(f"Configuration: P={P}, Mt={Mt}, Kt={Kt}, Nt={Nt}")
      # print(f"Compute per iter: {compute_iter(Mt, Kt, Nt)}")
      # print(f"X collective words per iter: {a_words(Mt, Kt)}")
      # print(f"Y collective words per iter: {b_words(Kt, Nt)}")
      # print(f"Comm step 0: {comm_iter(P, Mt, Kt, Nt, 0)}")
      # if P > 1:
      #       print(f"Comm step 1: {comm_iter(P, Mt, Kt, Nt, 1)}")
      print(f"{kernel_cycles(P, Mt, Kt, Nt)}")
