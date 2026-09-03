# oneTBB flow graph transform pipeline

You are given `/app/flow_graph.cpp`, a serial C++17 program that transforms deterministic integer inputs and sums the transformed values.

Create a oneTBB implementation that:

1. Processes integer inputs `0..n-1`.
2. Applies the same transform as the serial reference: `y = (x * 17 + 5) % 1009`, then `y*y + 3*y + 7`.
3. Uses `oneapi::tbb::flow::graph` or `tbb::flow::graph` with `function_node` stages to perform the work.
4. Prints a line containing `VALID` and the same sum as the serial reference.
5. Writes an executable binary at `/app/flow_graph_tbb`.

Requirements:

- Use actual oneTBB flow graph nodes; do not solve the task with only `parallel_for` or a serial loop.
- Accept optional CLI argument: `<count>`.
- Reject non-positive counts with a non-zero exit code.

A typical compile command is:

```bash
g++ -O3 -std=c++17 /app/flow_graph_tbb.cpp -ltbb -o /app/flow_graph_tbb
```
