# Final Summary: Pointer Mechanism Impact Differential

## Key Metrics Comparison

| metric                       | DIY    | GeoLife   | interpretation               |
|:-----------------------------|:-------|:----------|:-----------------------------|
| Target-in-History Rate       | 84.12% | 83.81%    | Similar copy opportunity     |
| Pointer Head Acc@1           | 56.53% | 51.63%    | Similar pointer performance  |
| Generation Head Acc@1        | 5.64%  | 12.19%    | GeoLife gen head 2x better   |
| Combined Model Acc@1         | 56.58% | 51.40%    | Similar final performance    |
| Mean Gate Value              | 0.787  | 0.627     | DIY relies more on pointer   |
| Unique Target Locations      | 1713   | 315       | DIY has 5.4x more targets    |
| Top-10 Target Coverage       | 41.75% | 67.13%    | GeoLife more concentrated    |
| Ablation Impact (from study) | 8.3%   | 46.7%     | GeoLife hurt more by removal |

## Root Cause Explanation


### Why does removing the pointer mechanism cause 46.7% drop on GeoLife but only 8.3% on DIY?

**The answer lies in the RELATIVE importance of the pointer vs generation heads:**

1. **DIY Dataset:**
   - Generation head accuracy: 5.64% (very weak)
   - Pointer head accuracy: 56.53%
   - The model learns to rely almost entirely on pointer (gate ≈ 0.79)
   - When pointer is removed, performance drops to generation baseline
   - But this appears as small *relative* drop because baseline was already heavily pointer-dependent

2. **GeoLife Dataset:**
   - Generation head accuracy: 12.19% (reasonable backup)
   - Pointer head accuracy: 51.63%
   - The model uses both heads more balanced (gate ≈ 0.63)
   - When pointer is removed, the model loses its primary prediction mechanism
   - This appears as large *relative* drop because the combined model was performing well with both components

### The Vocabulary Size Effect

The root cause of the generation head performance difference is **vocabulary size**:
- DIY: 1,713 unique target locations
- GeoLife: 315 unique target locations

With 5.4x more target locations, DIY's generation head must predict over a much larger space,
making accurate generation much harder. This forces the model to rely more heavily on the pointer mechanism.

### Conclusion

The differential ablation impact is not due to the pointer mechanism being "more important" for GeoLife.
Rather, it's because GeoLife's generation head provides a viable alternative, making the *relative*
impact of removing the pointer larger. In DIY, the model was already maximally pointer-dependent.
