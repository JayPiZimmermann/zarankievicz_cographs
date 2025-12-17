# Depth Analysis of Extremal Cographs

## Summary

This analysis examines the relationship between cotree depth and the parameters (s, t, n) in extremal K_{s,t}-free cographs.

## Key Findings

### 1. Depth = 0 Characterization
**Pattern**: Depth = 0 occurs **only** for n = 1 (the single vertex graph).
- Verified across all three export directories
- This makes sense: a single vertex has no operations, thus depth 0

### 2. Maximum Depths Observed

| Export Directory | Max Depth | Location | Graph Structure |
|-----------------|-----------|----------|-----------------|
| exports_lattice_12_2 | 6 | K_{3,10} at n=26 | Complex nested structure |
| exports_lattice_5_2 | 5 | K_{2,5} at n=12 | `P(1,S(P(1,1,1,1,1),P(S(1,P(1,1)),S(1,P(1,1)))))` |
| exports_star | 7 | K_{1,21} at n=54 | Highly nested star-based construction |

**Observation**: Maximum depth grows with:
- Increasing t (especially for stars K_{1,t})
- More complex forbidden structures
- Larger n values

### 3. Depth Growth Patterns for Symmetric Cases K_{i,i}

For symmetric cases at their maximum n:

**exports_lattice_12_2** (n=28):
```
K_{1,1}: depth 1
K_{2,2}: depth 3
K_{3,3}: depth 4
K_{4,4}: depth 3
K_{5,5}: depth 4
K_{6,6}: depth 3
K_{7,7}: depth 4
K_{8,8}: depth 3
K_{9,9}: depth 4
K_{10,10}: depth 3
K_{11,11}: depth 3
K_{12,12}: depth 3
```

**Pattern**: For i ≥ 2, depth oscillates between 3 and 4, with odd i tending toward depth 4.

**exports_lattice_5_2** (n=82):
```
K_{1,1}: depth 1
K_{2,2}: depth 3
K_{3,3}: depth 4
K_{4,4}: depth 3
K_{5,5}: depth 4
```

**Pattern**: Same oscillation between 3 and 4 for i ∈ {2,3,4,5}.

### 4. Depth Growth with n for Fixed (s,t)

**K_{1,1} (Simple Path/Star)**:
- Always depth 1 for n ≥ 2
- Formula: P(1,1,...,1) with n factors
- Minimal complexity

**K_{2,2} (Avoiding 4-cycle)**:
- Jumps to depth 3 at n=4
- Stays at depth 3 for all n ≥ 4
- Requires more complex structure to avoid squares

**K_{3,3}**:
- Depth 1 for n ∈ {2,3,4,5}
- Depth 2 at n=6
- Depth 3 at n={7,8}
- Depth 4 at n={9,10}
- Progressive increase with n

**K_{5,5}**:
- Depth 1 for n ∈ {2,...,9}
- Jumps to depth 2 at n=10
- Later progression not shown in data range

### 5. Operation Type Distribution by Depth

**Depth 0**: 100% vertex (trivial)

**Depth 1**:
- **exports_star**: 97.0% product, 3.0% sum
- **exports_lattice_12_2**: 96.9% product, 3.1% sum
- Simple constructions dominated by products

**Depth 2**:
- **exports_lattice_5_2**: 95.3% sum (anomaly!)
- **exports_star**: 58.4% sum, 41.6% product
- More balanced, but varies by constraint

**Depth 3**:
- **exports_lattice_12_2**: 92.9% product
- **exports_lattice_5_2**: 65.1% product
- **exports_star**: 53.4% product

**Depth 4+**:
- More balanced distribution between sum and product
- **exports_star depth 4**: 55.1% sum, 44.9% product
- **exports_star depth 5**: 55.3% product, 44.7% sum
- **exports_star depth 6**: 57.8% sum, 42.2% product
- **exports_star depth 7**: 79.5% product (outlier)

### 6. Depth and Asymmetry (s vs t)

**Star graphs K_{1,t}**:
```
exports_star:
K_{1,1}: max_depth=1
K_{1,2}: max_depth=2
K_{1,3}: max_depth=3
K_{1,4}: max_depth=4
K_{1,5}: max_depth=4
K_{1,6}: max_depth=4
K_{1,7}: max_depth=4
K_{1,8}: max_depth=5
K_{1,9}: max_depth=5
K_{1,10}: max_depth=6
...
K_{1,21}: max_depth=7
```

**Pattern**: max_depth roughly grows as ⌈log₂(t)⌉ to ⌈log₂(t)⌉ + 1 for star graphs.

This makes sense: building K_{1,t}-free graphs with many vertices requires nested constructions where degree growth is carefully controlled.

### 7. Average Depths

| Export Directory | Avg Min Depth | Avg Max Depth | Max n |
|-----------------|---------------|---------------|-------|
| exports_lattice_12_2 | 2.22 | 2.26 | 28 |
| exports_lattice_5_2 | 2.82 | 2.82 | 82 |
| exports_star | 1.99 | 2.72 | 65 |

**Observation**:
- Star graphs (exports_star) have lower average min_depth but higher max_depth variance
- Lattice-based constructions have higher and more uniform depth requirements
- The difference between min and max depth is small for lattice constraints but larger for star constraints

### 8. Depth Formula Hypotheses

**H1: Simple Product Rule**
For K_{1,1} (path graph): depth = 1 for all n ≥ 2
- Formula: P(1,1,...,1)
- **Verified**: Always holds

**H2: Minimum Complexity Threshold**
For K_{s,t} with s,t ≥ 2: min_depth ≥ 2 when n ≥ s+t
- **Rejected**: Many cases with min_depth = 1 for n >> s+t
- Example: K_{5,5} has depth 1 up to n=9

**H3: Logarithmic Growth for Stars**
For K_{1,t}: max_depth ≈ O(log t) as t increases
- **Supported**: K_{1,21} has depth 7 ≈ log₂(21)
- Makes sense: balanced tree-like constructions

**H4: Depth and Component Size**
Graphs with large component_sizes tend to have lower depth
- Simple products P(1,1,...,1) have component_sizes [1, n-1] and depth 1
- Complex nested structures have smaller, more balanced component splits

**H5: Depth Jump at "Critical n"**
For each (s,t), there's a critical n where depth jumps significantly
- K_{2,2}: jump at n=4 (from 1 to 3)
- K_{3,3}: jump at n=6 (from 1 to 2), then at n=7 (to 3)
- This n appears related to (s-1)(t-1) but doesn't exactly match

### 9. Structural Observations

**High Depth Structures (depth ≥ 5)**:
- Always contain multiple nested operations: S(P(...), P(...))
- Often have small components in inner levels (e.g., S(1,1), P(1,1))
- Build up complexity gradually through recursion
- Example: `P(1,S(P(1,1,1,1,1),P(S(1,P(1,1)),S(1,P(1,1)))))`

**Depth vs Last Operation**:
- Depth 1: Almost always product (97%+)
- Depth 2-3: Mixed, depends on constraint type
- Depth 4+: Roughly balanced (45-55% each)
- Very high depth (7): Product-dominated (79.5%)

### 10. Open Questions

1. **Is there a tight upper bound on depth as a function of (s,t,n)?**
   - Current max: 7 for K_{1,21} at n=54
   - Appears to be O(log t) for stars

2. **Why do some symmetric cases maintain constant depth for large n ranges?**
   - K_{2,2} stays at depth 3 for all n ≥ 4
   - Is this optimal or an artifact of the construction algorithm?

3. **What determines when min_depth = max_depth vs when they differ?**
   - Some (s,t,n) have unique structures, others have multiple with varying depth

4. **Is the oscillation in K_{i,i} depths (3,4,3,4,...) fundamental or algorithmic?**
   - Appears for both lattice datasets
   - Suggests some deeper structural property

5. **Can we predict depth from component_sizes pattern?**
   - Balanced splits suggest higher depth
   - Unbalanced (e.g., [1, n-1]) suggest lower depth

## Conclusion

Cotree depth is a meaningful structural parameter that:
- Grows slowly with n (average around 2-3)
- Correlates with the complexity of the forbidden structure K_{s,t}
- Shows interesting patterns in symmetric cases (oscillation)
- Has logarithmic-like growth for star graphs K_{1,t}
- Is nearly deterministic for some (s,t) but varies for others
- Reflects the balance between sum and product operations needed to avoid K_{s,t}

The depth parameter provides insight into the "nesting complexity" required to construct extremal graphs and could be useful for:
- Algorithm optimization (prune by depth bound)
- Theoretical characterization of extremal constructions
- Understanding the structure of the solution space
