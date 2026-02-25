Streaming Experiment: Overlapping Communication and Computation in GEMV
=======================================================================

This experiment modifies the ``gemv-09-streaming`` tutorial to demonstrate
**overlapping communication and computation** by streaming matrix A and vector x
concurrently, then computing Ax while b streams in.

Motivation
----------

The original ``gemv-09-streaming`` example uses a hybrid approach:

1. Matrix ``A`` is copied to all PEs using copy mode (``streaming=False``) - BLOCKING
2. Vectors ``x`` and ``b`` are streamed to the device (``streaming=True``) - SEQUENTIAL
3. Result ``y`` is streamed back from the device

This is **fully sequential** with NO overlap between communication and computation.

This experiment achieves **TRUE overlap**:

1. Stream A and x **concurrently** (both nonblock=True)
2. Device **computes Ax as x arrives** (computation starts immediately)
3. Stream b **while device computes** (b transfer overlaps with Ax computation)
4. Final reduction and streaming y back

**Key Performance Win:** Ax computation overlaps with b streaming!

Changes from Original
---------------------

**layout.csl**
  - Added ``MEMCPYH2D_DATA_3_ID`` parameter for streaming matrix A
  - Added ``MEMCPYH2D_DATA_3`` color definition
  - Updated memcpy module import to include ``MEMCPYH2D_3``

**pe_program.csl**
  - Added ``h2d_a_iq`` input queue for receiving A elements
  - Added ``memcpy_recv_a_task_id`` data task ID
  - Added ``num_recv_a`` counter and ``a_ready`` flag to track A reception
  - Added ``memcpy_recv_a`` task to receive and buffer A elements
  - Modified ``recv_x`` task to only compute when ``a_ready == true``
  - Bound ``memcpy_recv_a`` task and initialized its queue (WSE-3)

**run.py**
  - Added ``MEMCPYH2D_DATA_3`` color ID extraction
  - Removed non-streaming ``memcpy_h2d`` for matrix A
  - Added streaming ``memcpy_h2d`` for matrix A with **nonblock=True**
  - Added streaming ``memcpy_h2d`` for vector x with **nonblock=True**
  - **A and x stream concurrently** (both non-blocking)
  - **b streams with nonblock=False** to ensure completion before reading y
  - This creates overlap: Ax computation happens while b is streaming!

**commands_wse2.sh**
  - Added ``MEMCPYH2D_DATA_3_ID:3`` compile parameter

Execution Flow with Overlap
---------------------------

**Timeline visualization:**

.. code-block::

   Host:   [Stream A (nonblock)] ──┐
           [Stream x (nonblock)] ──┤ ALL Concurrent
           [Stream b (nonblock)] ──┤ (maximum host-side parallelism)
                                   └──→ [Read y (SYNC)] ← Blocks here
                                            ↑
   Device:        [Receive A] → [Compute Ax as x arrives] ──┘
                                    ↑
                                    └── OVERLAP! Computation happens while all 3 stream!

**Detailed steps:**

1. **Host streams A** (``nonblock=True``) → All PEs receive their chunks concurrently

   - Each PE's ``memcpy_recv_a`` task stores incoming elements into the ``A`` array
   - When all elements are received, ``a_ready`` is set to ``true``

2. **Host streams x** (``nonblock=True``) → Top row receives x, broadcasts down

   - ``memcpy_recv_x`` receives x from host, forwards via ``x_color``
   - ``recv_x`` task triggers on all PEs as each x element arrives
   - **KEY:** If ``a_ready == true``, immediately computes ``y += A_col * x_val``
   - **Computation starts while x is still streaming!**

3. **Host streams b** (``nonblock=True``) → Left column receives b

   - Host issues command and **continues immediately** (doesn't wait)
   - **All three inputs (A, x, b) now streaming concurrently!**
   - Device is computing Ax while b streams (overlap achieved)
   - ``memcpy_recv_b`` receives b, forwards to itself via ``recv_west_color``

4. **Device finishes Ax, adds b** → ``reduce`` task activated when all x received

   - Partial results (Ax) propagate EAST across rows
   - Each PE adds its local b component: ``y + b``
   - Final column PEs have complete result: ``y = Ax + b``

5. **Host receives y** (``nonblock=False``) → **SYNCHRONIZATION POINT**

   - This call **blocks** and waits for all previous operations to complete:

     - All streaming (A, x, b) must finish
     - Device computation (Ax + b) must finish
     - Result must be ready

   - Only then does host receive y via ``MEMCPYD2H_DATA_1``
   - This ensures correctness while maximizing concurrency

Key Design Decisions
--------------------

**Concurrent A and x streaming** (``nonblock=True``):

  - Host issues both stream commands without waiting
  - Device receives A and x **simultaneously**
  - Once ``a_ready == true``, computation begins immediately
  - No blocking = maximum overlap potential

**Computation starts as x arrives**:

  - ``recv_x`` task checks ``if (a_ready)`` then computes
  - As each x element broadcasts down columns, all PEs compute ``y += A_col * x_val``
  - This is **incremental computation** - no waiting for all x to arrive

**Maximum host-side concurrency** (all inputs use ``nonblock=True``):

  - Host issues A, x, and b streaming commands without waiting
  - All three transfers happen concurrently from host perspective
  - Device receives data and computes as it arrives
  - **This is the key overlap**: communication (all inputs) and computation (Ax) happen simultaneously

**Synchronization at output** (``memcpy_d2h`` with ``nonblock=False``):

  - Instead of blocking on input streaming, we block on output reading
  - This maximizes concurrency: host doesn't wait during input phase
  - The ``memcpy_d2h`` call implicitly waits for all previous operations
  - Safe: ensures computation completes before reading result

**Synchronization**: The ``a_ready`` flag ensures:

  - PEs don't attempt GEMV computation with incomplete A data
  - Thread-safe coordination between data arrival and computation

**Memory**: A is still buffered in device memory (48 kB per PE), so this doesn't
reduce memory usage - it only changes the data transfer mechanism.

Running the Example
-------------------

From this directory, execute for WSE-2::

  ./commands_wse2.sh

Or for WSE-3::

  ./commands_wse3.sh

Or manually for WSE-2::

  cslc --arch=wse2 ./layout.csl --fabric-dims=11,5 \\
    --fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \\
    --params=MEMCPYH2D_DATA_1_ID:0 \\
    --params=MEMCPYH2D_DATA_2_ID:1 \\
    --params=MEMCPYH2D_DATA_3_ID:6 \\
    --params=MEMCPYD2H_DATA_1_ID:2 \\
    -o out --memcpy --channels 1
  cs_python run.py --name out

Or manually for WSE-3::

  cslc --arch=wse3 ./layout.csl --fabric-dims=11,5 \\
    --fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \\
    --params=MEMCPYH2D_DATA_1_ID:0 \\
    --params=MEMCPYH2D_DATA_2_ID:1 \\
    --params=MEMCPYH2D_DATA_3_ID:6 \\
    --params=MEMCPYD2H_DATA_1_ID:2 \\
    -o out --memcpy --channels 1
  cs_python run.py --name out

Expected output::

  SUCCESS!

Potential Applications
----------------------

This streaming approach could be useful for:

- **Dynamic matrix generation**: Where A is computed on-the-fly and streamed
- **Memory-bandwidth optimization**: Overlapping data transfer with computation
- **Pipelined workloads**: Where successive GEMVs use different matrices
- **Large-scale problems**: Where A doesn't fit in host memory and must be generated/streamed

Further Experiments
-------------------

Possible extensions to explore:

1. **Concurrent A and x streaming**: Use ``nonblock=True`` to overlap A and x transfers
2. **Partial computation**: Start computing with partial A columns as they arrive
3. **Performance profiling**: Compare streaming vs. copy mode for different matrix sizes
4. **Error handling**: Add checks for malformed or incomplete data streams
