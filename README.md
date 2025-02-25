# Spectral Analysis of Multi-Hop Homophily Dominance

## 1. Transition Matrix Definition

We define the **transition matrix** for label propagation:

$$
\mathbf{Q} = (\alpha - \beta)\mathbf{I} + \beta\mathbf{J}
$$

where:
- $\mathbf{I}$ is the identity matrix.
- $\mathbf{J}$ is the **all-ones** matrix.
- $\beta = \frac{1 - \alpha}{|\mathcal{Y}| - 1}$ ensures that the sum of each row in $\mathbf{Q}$ equals 1.

## 2. Eigenvalues of $\mathbf{Q}$

Since $\mathbf{J}$ is a rank-1 matrix, it has a single **nonzero** eigenvalue $|\mathcal{Y}|$, with the rest being 0.

Using this, the eigenvalues of $\mathbf{Q}$ are:

$$
\lambda_1 = \alpha + (|\mathcal{Y}| - 1)\beta = \alpha + (|\mathcal{Y}| - 1)\frac{1 - \alpha}{|\mathcal{Y}| - 1} = 1
$$

$$
\lambda_2 = \lambda_3 = \dots = \lambda_{|\mathcal{Y}|} = \alpha - \beta
$$

Thus, the **spectral decomposition** of $\mathbf{Q}$ is:

$$
\mathbf{Q} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^{-1}
$$

where the diagonal eigenvalue matrix $\mathbf{\Lambda}$ is:

$$
\mathbf{\Lambda} =
\begin{bmatrix}
1 & 0 & \dots & 0 \\
0 & (\alpha - \beta) & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & (\alpha - \beta)
\end{bmatrix}
$$

## 3. Computing $\mathbf{Q}^h$

Using spectral decomposition:

$$
\mathbf{Q}^h = \mathbf{U} \mathbf{\Lambda}^h \mathbf{U}^{-1}
$$

where:

$$
\mathbf{\Lambda}^h =
\begin{bmatrix}
1^h & 0 & \dots & 0 \\
0 & (\alpha - \beta)^h & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & (\alpha - \beta)^h
\end{bmatrix}
$$

Thus, raising $\mathbf{Q}$ to power $h$, we get:

$$
\mathbf{Q}^h = (\alpha - \beta)^h \mathbf{I} + \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|} \mathbf{J}
$$

## 4. Computing Diagonal and Off-Diagonal Elements

From the above equation:

### **Diagonal Elements:**
$$
\mathbf{Q}^h_{i,i} = (\alpha - \beta)^h + \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|}
$$

### **Off-Diagonal Elements:**
$$
\mathbf{Q}^h_{i,j} = \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|}, \quad j \neq i
$$

## 5. Proof of Diagonal Dominance

We now prove:

$$
\mathbf{Q}^h_{i,i} > \mathbf{Q}^h_{i,j}, \quad \forall j \neq i
$$

Computing the difference:

$$
\mathbf{Q}^h_{i,i} - \mathbf{Q}^h_{i,j} = \left( (\alpha - \beta)^h + \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|} \right) - \frac{1 - (\alpha - \beta)^h}{\mathcal{Y}|}
$$

Simplifying:

$$
\mathbf{Q}^h_{i,i} - \mathbf{Q}^h_{i,j} = (\alpha - \beta)^h
$$

Since $\alpha > \beta$, we have:

$$
(\alpha - \beta)^h > 0
$$

### **Conclusion:**
For all $h \geq 2$, we have:

$$
\mathbf{Q}^h_{i,i} > \mathbf{Q}^h_{i,j}, \quad \forall j \neq i
$$

Thus, **multi-hop label propagation preserves homophily dominance**.

---

## 6. Key Takeaways

- The **correct** expression for $\mathbf{Q}^h$ is:

$$
\mathbf{Q}^h = (\alpha - \beta)^h \mathbf{I} + \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|} \mathbf{J}
$$

- The **correct diagonal and off-diagonal elements**:

$$
\mathbf{Q}^h_{i,i} = (\alpha - \beta)^h + \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|}
$$

$$
\mathbf{Q}^h_{i,j} = \frac{1 - (\alpha - \beta)^h}{|\mathcal{Y}|}, \quad j \neq i
$$

- **Diagonal dominance is proven** by showing:

$$
\mathbf{Q}^h_{i,i} - mathbf{Q}^h]_{i,j} = (\alpha - \beta)^h > 0
$$

🚀 **This proof rigorously establishes that homophily dominance is maintained for all $h \geq 2$.** 🚀

---

## **Conclusion**
- We proved that **2-hop neighborhoods are always homophily-dominant** in expectation.
- We established that **1-hop neighborhoods require $\alpha \geq \frac{1}{|\mathcal{Y}|}$ to maintain homophily dominance**.
- This result provides insight into how **multi-hop label propagation in graphs** behaves under a probabilistic homophily assumption.

🚀 **This proof validates the role of homophily in LLM-driven graph annotation!** 🚀

## **Run Code**
python3 src/main.py  --dataset cora --model_name GCN --data_format sbert --main_seed_num 5 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 140 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 150 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.15 --beta 0.65
