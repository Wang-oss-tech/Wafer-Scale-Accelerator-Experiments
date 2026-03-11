import math

def compute_iter(Mt, Kt, Nt):
      d = bin((3 * Kt >> 1) ^ (Kt >> 1)).count('1')
      b_load = 2 * d - 1 - (Kt % 2) + 8
      fmacs = 4 + math.ceil(Mt / 8) # latency between issue and executing, 8 FMAC units
      # print("fmacs count", fmacs, "math.ceil", math.ceil(Mt/8))
      dsd_increment = 14 + 3   # 12 ST16 + 2 ADD16 + 3 NOP
      loop_control = 3  # SUB16 + MOV16 + JNC
      return Kt * Nt * (b_load + fmacs + dsd_increment + loop_control)

def broadcast_iter(P, Mt, Nt):
      configure_broadcast = 62
      wavelet_broadcasting = 2 * (Mt * Nt)
      callback_task = 13 + 13 # A_done + B_done
      receive_hops = P * (P - 1)  # Total: 2*[(P-1) + (P-2) + ... + 1 + 0] = P*(P-1)
      return configure_broadcast + wavelet_broadcasting + callback_task + receive_hops

def h2d_memcpy(P, Mt, Kt, Nt):
      # travel_bottom_right = (P - 1) * 2 + (P - 1)
      # travel_bottom_right = 0
      startup_time = 1300 # fixed startup time
      a_wavelet_transfer = Mt*Kt*P*P
      a_to_b_switch = 1100
      b_wavelet_transfer = Kt*Nt*P*P
      finish_memcpy_launch_kernel = 330
      return startup_time + a_wavelet_transfer + a_to_b_switch + \
              b_wavelet_transfer + finish_memcpy_launch_kernel

def d2h_memcpy(P, Mt, Nt):
      wavelets = Mt*Nt
      time_between_wavelet_transfers = 52
      return wavelets * time_between_wavelet_transfers

def total_cycles(P, Mt, Kt, Nt):
      compute = compute_iter(Mt, Kt, Nt)
      broadcast = broadcast_iter(P, Mt, Nt)
      h2d = h2d_memcpy(P, Mt, Kt, Nt)
      d2h = d2h_memcpy(P, Mt, Nt)
      
      kernel_cycles = (P * (compute)) + ((P-1) * broadcast)
      io_cycles = h2d + d2h
      travel_bottom_right = (P - 1) * 2 + (P - 1)
      
      return travel_bottom_right + kernel_cycles + io_cycles, kernel_cycles, io_cycles

# Test with measured configuration
if __name__ == "__main__":
      P, Mt, Kt, Nt = 180, 12, 12, 12
      
      print(f"Configuration: P={P}, Mt={Mt}, Kt={Kt}, Nt={Nt}")
      print(f"H2D cycles: {h2d_memcpy(P, Mt, Kt, Nt)}")
      print(f"D2H cycles: {d2h_memcpy(P, Mt, Nt)}")
      print(f"Compute per iter: {compute_iter(Mt, Kt, Nt)}")
      print(f"Broadcast per iter: {broadcast_iter(P, Mt, Nt)}")
      total, kernel, io = total_cycles(P, Mt, Kt, Nt)
      print(f"Kernel cycles: {kernel}")
      print(f"IO cycles: {io}")
      print(f"Total cycles: {total}")





