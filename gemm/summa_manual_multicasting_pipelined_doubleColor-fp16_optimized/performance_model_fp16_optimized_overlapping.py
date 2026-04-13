import math


def fmach_issue_period(Mt):
      # Effective FMACH initiation spacing for this kernel.
      #
      # This is not a hard hardware floor. The lower bound captures the
      # observed issue spacing of the current wave-scheduled kernel for small
      # Mt, where leftover SIMD waves still create nontrivial spacing even when
      # a full 8-lane wave is not required.
      #
      # The structural term 2 * ceil(Mt / 8) - 1 approximates the number of
      # 8-lane SIMD waves needed for one FMACH step. The max preserves the
      # empirical lower bound used by the current implementation.
      return max(5, 2 * math.ceil(Mt / 8) - 1)

# Communication constants for the revised model.
# fixed + A_words + B_words cannot be overlapped.
# alpha * distance can be overlapped with compute.
COMM_FIXED = 230.0
COMM_WORD_ALPHA = 0.4
COMM_HOP_ALPHA = 0.25


def compute_iter(Mt, Kt, Nt):
      fmach = fmach_issue_period(Mt)
      outer_overhead = 40  # per-kk cost: 3x load_to_dsr + 2x increment_dsd_offset
      return Kt * (Nt - 1) * fmach + (Kt - 1) * outer_overhead


def a_words(Mt, Kt):
      return math.ceil(Mt * Kt / 2)


def b_words(Kt, Nt):
      return math.ceil(Kt * Nt / 2)


def distance_from_source(P, k):
      return max(k, P - 1 - k)


def nonoverlap_comm_iter(Mt, Kt, Nt):
      words = a_words(Mt, Kt) + b_words(Kt, Nt)
      return COMM_FIXED + COMM_WORD_ALPHA * words


def overlap_comm_iter(P, k):
      return COMM_HOP_ALPHA * distance_from_source(P, k)


def broadcast_iter(P, Mt, Kt, Nt, k):
      return nonoverlap_comm_iter(Mt, Kt, Nt) + overlap_comm_iter(P, k)


def kernel_cycles(P, Mt, Kt, Nt):
      compute = compute_iter(Mt, Kt, Nt)
      nonoverlap_comm = nonoverlap_comm_iter(Mt, Kt, Nt)

      if P <= 0:
            return 0
      if P == 1:
            return broadcast_iter(P, Mt, Kt, Nt, 0) + compute

      # Split async pipeline:
      # broadcast 0 = nonoverlap + overlap_tail(0)
      # then for k = 1..P-1:
      #   nonoverlap + max(overlap_tail(k), compute k-1)
      # then final standalone compute
      total = broadcast_iter(P, Mt, Kt, Nt, 0)
      for k in range(1, P):
            total += nonoverlap_comm + max(overlap_comm_iter(P, k), compute)
      total += compute
      return total


if __name__ == "__main__":
      import argparse

      parser = argparse.ArgumentParser(description="SUMMA fp16 optimized performance model (overlapping)")
      parser.add_argument("--P", type=int, default=4)
      parser.add_argument("--Mt", type=int, default=14)
      parser.add_argument("--Kt", type=int, default=14)
      parser.add_argument("--Nt", type=int, default=14)
      args = parser.parse_args()
      P, Mt, Kt, Nt = args.P, args.Mt, args.Kt, args.Nt

      # print(f"Configuration: P={P}, Mt={Mt}, Kt={Kt}, Nt={Nt}")
      # print(f"Compute per iter: {compute_iter(Mt, Kt, Nt)}")
      # print(f"Non-overlap comm per iter: {nonoverlap_comm_iter(Mt, Kt, Nt)}")
      # print(f"Overlap comm step 0: {overlap_comm_iter(P, 0)}")
      # if P > 1:
      #       print(f"Overlap comm step 1: {overlap_comm_iter(P, 1)}")
      # print(f"Broadcast step 0: {broadcast_iter(P, Mt, Kt, Nt, 0)}")
      print(f"{kernel_cycles(P, Mt, Kt, Nt)}")
