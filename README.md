python3 src/main.py  --dataset cora --model_name GCN --data_format sbert --main_seed_num 5 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 140 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 150 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.15 --beta 0.65


# Theoretical Analysis of Multi-Hop Homophily Propagation

## Theorem: Multi-Scale Homophily Conservation

Consider a directed graph <i>G = (V,E)</i> with pseudo-labels <i>Y</i> generated through LLM-based annotation. For any node <i>v ∈ V</i>, assume:

1. **Conditional Independence**: Neighborhood labels {y<sub>u</sub>}<sub>u∈N(v)</sub> are conditionally independent given y<sub>v</sub>
2. **Symmetric Label Transition**:
   $$
   P(y_u = y_v | y_v) = α,\quad 
   P(y_u = y | y_v) = \frac{1-α}{|Y|-1} := β\ (∀ y ≠ y_v)
   $$

Then the following conservation properties hold:

### First-Order Threshold
The immediate neighborhood N<sub>1</sub>(v) exhibits homophily dominance in expectation _iff_:
$$
α ≥ α^{(1)}_c = \frac{1}{|Y|}
$$

### Multi-Hop Stability
For any h ≥ 2, the h-hop neighborhood N<sub>h</sub>(v) maintains homophily dominance in expectation _if_:
$$
α > α^{(h)}_c = \frac{1-α}{|Y|-1}
$$

## Proof Sketch

### First-Order Analysis
The label transition matrix Q can be decomposed as:
$$
Q = (α - β)I + βJ
$$
where I is identity matrix and J the all-ones matrix.

The expected homophily ratio:
$$
\mathbb{E}[R_1(v)] = \frac{α|Y|}{1} 
$$
Homophily dominance requires:
$$
α ≥ \frac{1}{|Y|}
$$

### Multi-Hop Propagation
Through spectral decomposition:
$$
Q^h = UΛ^hU^{-1}
$$
where Λ contains eigenvalues:
$$
λ_1 = α + (|Y|-1)β,\quad λ_2 = \cdots = λ_{|Y|} = α - β
$$

The diagonal dominance condition reduces to:
$$
(α - β)^h > \frac{1 - (α - β)^h}{|Y|-1}
$$
solved by:
$$
α > \frac{1}{|Y|}
$$

### Phase Transition
Critical threshold exhibits asymptotic behavior:
$$
\lim_{h→∞} α^{(h)}_c = \frac{1}{|Y|}
$$

## Corollary: Binary Classification
For |Y| = 2:
- First-order threshold: α ≥ 0.5
- Multi-hop condition: α > 1/3
- Critical convergence: lim<sub>h→∞</sub> α<sub>c</sub><sup>(h)</sup> = 0.5
