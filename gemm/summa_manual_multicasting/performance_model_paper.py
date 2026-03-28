import argparse

# I represents num of elements
def t_fmac(I):
    return 2 * I + 10

def t_bcast(I):
    return 1.6 * I + 250

def total_cycle_count(P, Mt, Kt, Nt):
    n_loop = Kt * Nt      # local loop overhead, not global SUMMA steps

    t_comp = t_fmac(Mt * Nt) + 0.5 * n_loop + 10
    t_comm = t_bcast(Mt * Kt) + t_bcast(Kt * Nt)
    t_cntl = 100

    return (t_comp + t_comm + t_cntl) * P

def computation_performance(M, result):
    ops = 2 * (M ** 3) * 800 * 1000000
    return ops / result

def calculate_mape(actual, predicted):
    return abs((actual - predicted) / actual) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total cycle count.")
    parser.add_argument("-P", type=int, required=True, help="Parameter P")
    parser.add_argument("-Mt", type=int, required=True, help="Parameter Mt")
    parser.add_argument("-Kt", type=int, required=True, help="Parameter Kt")
    parser.add_argument("-Nt", type=int, required=True, help="Parameter Nt")
    parser.add_argument("-actual_cycle_count", type=float, required=False, help="Actual cycle count for comparison")

    args = parser.parse_args()

    result = total_cycle_count(P=args.P, Mt=args.Mt, Kt=args.Kt, Nt=args.Nt)
    comp_perf = computation_performance(M=args.P * args.Mt, result=result)

    print(f"Total cycle count: {result}")
    print(f"Estimated computation performance: {comp_perf / 1e12:.2f} TFLOPS")

    # If actual_cycle_count is provided, calculate and display MAPE
    if args.actual_cycle_count:
        mape_cycle_count = calculate_mape(args.actual_cycle_count, result)
        actual_comp_perf = computation_performance(M=args.P * args.Mt, result=args.actual_cycle_count)
        mape_comp_perf = calculate_mape(actual_comp_perf, comp_perf)

        print(f"Mean Absolute Percentage Error (MAPE) for Total Cycle Count: {mape_cycle_count:.2f}%")
        print(f"Mean Absolute Percentage Error (MAPE) for Estimated Computation Performance: {mape_comp_perf:.2f}%")