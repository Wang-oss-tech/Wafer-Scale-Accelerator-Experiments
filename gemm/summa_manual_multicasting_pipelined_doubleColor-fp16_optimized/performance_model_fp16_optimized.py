import math

def compute_iter(Mt, Kt, Nt):
      fmach = max(5, 2 * math.ceil(Mt / 8) - 1)  # IS-to-IS period: 8 SIMD FP16 units, observed 5 cycles for Mt=14
      outer_overhead = 40  # per-kk cost: 3x load_to_dsr + 2x increment_dsd_offset, observed from sim log
      # IS span = IS OPs within each kk + gaps at kk boundaries
      return Kt * (Nt - 1) * fmach + (Kt - 1) * outer_overhead

def broadcast_iter(P, Mt, Nt):
      configure_broadcast = 62
      wavelet_broadcasting = 2 * (Mt * Nt)
      callback_task = 13 + 13 # A_done + B_done
      receive_hops = P * (P - 1)  # Total: 2*[(P-1) + (P-2) + ... + 1 + 0] = P*(P-1)
      return configure_broadcast + wavelet_broadcasting + callback_task + receive_hops

def kernel_cycles(P, Mt, Kt, Nt):
      compute = compute_iter(Mt, Kt, Nt)
      broadcast = broadcast_iter(P, Mt, Nt)
      # Pipelined: broadcasts overlap with compute; only 1 broadcast exposed (the first, before any compute)
      return (P * compute) + (1 * broadcast)

if __name__ == "__main__":
      import argparse
      parser = argparse.ArgumentParser(description="SUMMA fp16 optimized performance model")
      parser.add_argument("--P",  type=int, default=4)
      parser.add_argument("--Mt", type=int, default=14)
      parser.add_argument("--Kt", type=int, default=14)
      parser.add_argument("--Nt", type=int, default=14)
      args = parser.parse_args()
      P, Mt, Kt, Nt = args.P, args.Mt, args.Kt, args.Nt

      print(f"Configuration: P={P}, Mt={Mt}, Kt={Kt}, Nt={Nt}") 
      print(f"Compute per iter: {compute_iter(Mt, Kt, Nt)}")
      print(f"Broadcast per iter: {broadcast_iter(P, Mt, Nt)}")
      print(f"Kernel cycles: {kernel_cycles(P, Mt, Kt, Nt)}")





