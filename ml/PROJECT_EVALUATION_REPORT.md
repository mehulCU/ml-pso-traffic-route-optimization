
# ML–PSO Hybrid Framework for Dynamic Traffic Route Optimization

## 1. Project Overview

This project combines machine learning-based traffic probe-count forecasting with graph-based route optimization. The objective is to forecast future road activity, assign experimental costs to road segments, and identify routes that minimize the total estimated cost under baseline and simulated disruption conditions.

**Dataset:** New Delhi Traffic Probe Count & Analytics Dataset (2024)
**Forecasting model:** Long Short-Term Memory (LSTM) neural network
**Routing baseline:** Dijkstra's shortest-path algorithm
**Optimization experiment:** Custom discrete PSO-inspired route search
**Implementation:** Python, pandas, NetworkX, TensorFlow/Keras, Matplotlib

## 2. Dataset and Preprocessing

The dataset contains hourly probe-count observations across road segments in New Delhi from August 11–30, 2024.

The processed dataset contains approximately 11.97 million records, with fields including date, hour, segment ID, street name, speed limit, functional road class, distance, and probe count.

The regional routing experiment uses 988 road segments connected through 620 graph nodes. A simplified undirected graph is constructed for route optimization.

Preprocessing and graph construction include:
- Processing hourly road-segment observations.
- Creating chronological forecasting splits.
- Generating lagged observations and time-based features.
- Building a road network from segment endpoints.
- Assigning experimental costs to graph edges.

## 3. Traffic Forecasting Evaluation

### 3.1 Initial forecasting experiment

An initial LSTM experiment was evaluated on a subset of 100 roads using historical observations to predict the next hour's probe count.

| Model | MAE | RMSE |
|---|---:|---:|
| LSTM | 104.6969 | 140.0334 |
| Naive baseline | 170.5157 | 229.7965 |

The LSTM reduced MAE by approximately 38.6% and RMSE by approximately 38.9% relative to the naive baseline.

### 3.2 Regional forecasting experiment

The regional dataset includes 988 road segments, each with 480 hourly observations.

| Model | MAE | RMSE |
|---|---:|---:|
| Regional LSTM | 16.8236 | 31.0863 |
| Naive baseline | 22.4003 | 41.0383 |

The regional LSTM reduced MAE by approximately 24.9% and RMSE by approximately 24.2% relative to its naive baseline.

**Interpretation:** Both experiments show improved probe-count forecasting compared with their respective naive baselines. The two experiments use different evaluation subsets, so their absolute error values should not be compared directly.

## 4. Routing Cost Model

Each road segment is assigned an experimental cost using its distance, speed limit, historical mean probe count, and predicted probe count.

The free-flow cost estimate is:

`free_flow_minutes = distance_m * 0.06 / speed_limit_kmh`

A traffic-related penalty is then calculated using the ratio of predicted probe count to historical mean probe count:

`traffic_ratio = predicted_probe_count / historical_mean_probe_count`

`traffic_penalty_factor = 1 + 0.25 * max(traffic_ratio - 1, 0)`

`traffic_aware_cost_minutes = free_flow_minutes * traffic_penalty_factor`

These costs provide an experimental way to connect forecasting and routing.

**Limitation:** Probe counts measure observed probe activity, not vehicle speed or congestion directly. Consequently, these costs are not validated travel-time estimates. The penalty formula is a modeling assumption and requires calibration against actual travel-time or speed data.

## 5. Dijkstra and PSO Evaluation

### 5.1 Static routing experiment

Dijkstra's algorithm provides the shortest path for a graph with non-negative edge costs.

For source node 11388 and destination node 14979, the baseline route contains 74 edges and has an experimental cost of approximately 3.495936 cost-minutes.

A custom discrete PSO-inspired optimizer was evaluated against Dijkstra across several source–destination pairs. The optimizer represents particles as candidate paths and updates them through route-edit operations inspired by personal-best and global-best search.

### 5.2 Multi-pair benchmark

Five source–destination pairs were tested with five PSO trials per pair. The experiments used randomized route initialization without seeding the known Dijkstra-optimal route.

| Source → destination | Mean PSO runtime |
|---|---:|
| 316 → 336 | 280.43 ms |
| 316 → 346 | 311.27 ms |
| 316 → 360 | 369.81 ms |
| 316 → 361 | 346.83 ms |
| 316 → 363 | 355.44 ms |

All 25 trials found the Dijkstra reference cost.

### 5.3 Challenging-route benchmark

For route 11388 → 14979:
- Dijkstra reference cost: 3.49593612.
- PSO initial-optimal rate: 55% across 20 trials.
- PSO final-optimal rate: 100%.
- Mean PSO runtime: approximately 3.93 seconds.
- Randomized-search optimal rate: 100%.
- Mean randomized-search runtime: approximately 2.75 seconds.

Both methods found the Dijkstra reference optimum in these trials. Randomized search was faster in this experiment, so the results do not establish that PSO is superior.

For static routing with non-negative edge costs, Dijkstra remains the appropriate exact baseline. The PSO-inspired method is an experimental metaheuristic rather than a replacement proven to outperform Dijkstra.

## 6. Dynamic Disruption Experiments

Simulated disruptions were introduced by multiplying the costs of selected graph edges.

For the initial graph-level experiment, 15% of graph edges were selected and their costs doubled.

| Metric | Result |
|---|---:|
| Baseline route cost | 3.495936 min |
| New optimal route cost | 3.800835 min |
| Original route cost under disruption | 3.986688 min |
| Penalty for retaining original route | 0.185853 min |
| Original-route penalty relative to new optimum | 4.8898% |
| Baseline-route edge overlap | 37.74% |

The optimal route changed in response to the simulated cost increases.

### 6.1 Robustness across random disruption patterns

Four scenarios were evaluated over ten random seeds each.

| Scenario | Mean optimal cost | Standard deviation | Route-change rate |
|---|---:|---:|---:|
| 5% edges × 2 cost | 3.6153 | 0.0505 | 50% |
| 15% edges × 2 cost | 3.9321 | 0.1675 | 80% |
| 30% edges × 2 cost | 4.3998 | 0.1838 | 70% |
| 15% edges × 3 cost | 4.3305 | 0.3389 | 90% |

The results indicate that disruption severity and the specific edges affected influence route selection. The route-change rate is not necessarily monotonic because different edge selections can affect different parts of the network.

These experiments use simulated cost multipliers and should not be interpreted as real-world incident simulations calibrated to measured traffic conditions.

## 7. Dynamic PSO Comparison

In the 15% × 2 graph disruption scenario, Dijkstra produced an optimal cost of 3.80083491.

Across 20 PSO trials:
- Initial-optimal rate: 70%.
- Final-optimal rate: 100%.
- Mean runtime: approximately 3.87 seconds.

Randomized search also reached the Dijkstra reference optimum in all 20 trials, with a mean runtime of approximately 2.74 seconds.

Therefore, these experiments demonstrate that the PSO-inspired search can find good routes in the tested scenarios, but do not demonstrate a performance advantage over randomized search or Dijkstra.

## 8. Tools and Technologies

- Python
- pandas and NumPy
- TensorFlow/Keras for LSTM forecasting
- NetworkX for graph construction and shortest-path algorithms
- Matplotlib for route visualization
- CSV-based data processing and evaluation

## 9. Limitations

1. Probe counts are not direct measurements of traffic speed, travel time, or congestion.
2. The routing cost formula is experimental and has not been calibrated against ground-truth travel times.
3. Disruptions are simulated through edge-cost multipliers rather than observed incidents.
4. The graph is simplified and undirected; one-way restrictions, turn penalties, and some intersection details are not modeled.
5. Forecasting and routing results are based on historical 2024 data, not live traffic feeds.
6. The PSO implementation is a custom discrete PSO-inspired method; further comparison is needed before claiming an advantage over established algorithms.
7. Robustness experiments use a limited number of seeds and one principal source–destination pair for disruption analysis.

## 10. Future Improvements

- Validate predicted probe counts and routing costs using independent test data.
- Incorporate measured speed, travel-time, and congestion data.
- Model road directionality, turn restrictions, and realistic intersections.
- Evaluate more source–destination pairs and additional network regions.
- Compare PSO, randomized search, and Dijkstra using consistent computational budgets.
- Repeat experiments across more random seeds and report confidence intervals.
- Investigate graph neural networks or spatiotemporal models for road-network forecasting.
- Integrate live traffic feeds only if a reliable data source becomes available.

## 11. Conclusion

The project establishes a working pipeline that connects historical traffic probe-count forecasting with graph-based route optimization and simulated disruption testing.

The LSTM experiments outperform their respective naive forecasting baselines. Dijkstra provides an exact reference for the static non-negative-cost routing problem, while the custom PSO-inspired optimizer successfully finds reference-optimal routes in the tested experiments. Simulated disruptions also demonstrate that changing road-edge costs can lead to alternative route selections.

The main contribution is the integration and experimental evaluation of forecasting, cost-based routing, and disruption-aware route selection. Further validation with real traffic-speed or travel-time data is necessary before treating the estimated costs as real-world travel times.
