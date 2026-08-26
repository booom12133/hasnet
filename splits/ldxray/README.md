# LDXray image-level multi-label manifests

[LDXray](https://github.com/rstao-bjtu/LDXray) was originally released for
paired dual-view object detection. The original images are not redistributed
in this repository.

The split unit is one synchronized dual-view pair: View A and View B always
belong to the same subset and are never separated.

For the image-level multi-label task, bounding-box coordinates are discarded
as supervision. Detection annotations are temporarily converted to a 12-class
pair-level multi-hot vector according to whether each prohibited-item category
appears anywhere in the pair. Multiple bounding boxes of the same class still
contribute one positive label; bounding-box instance counts are not used for
stratification or as supervision for HASNet.

The official LDXray split contains 110,148 training pairs and 36,849 test
pairs. The official test split is preserved unchanged and never mixed into the
training data. Only the official training split is divided into training and
validation subsets using pair-level iterative multi-label stratification with
a 9:1 ratio and split seed 42. The validation size is fixed as
`ceil(110,148 × 0.10) = 11,015` pairs.

The stratification procedure prioritizes categories with fewer remaining
positive pairs and assigns their pairs to the subset with the larger current
class-distribution deficit. Every assignment updates all positive categories
in a multi-label pair. Remaining zero-label pairs are assigned according to
the remaining subset capacities. This preserves positive-pair proportions as
closely as multi-label co-occurrence permits while enforcing the exact subset
sizes.

Final split sizes:

- Train: 99,133 pairs
- Validation: 11,015 pairs
- Test: 36,849 pairs

The manifest and model-output class order is:

`MP, OL, PC1, LA, GL, PC2, TA, BL, CO, NL, UM, CG`

The corresponding indices are `0: MP`, `1: OL`, `2: PC1`, `3: LA`, `4: GL`,
`5: PC2`, `6: TA`, `7: BL`, `8: CO`, `9: NL`, `10: UM`, and `11: CG`.
For presentation and category analysis, Table 3 reports classes in the order
`MP, OL, PC1, PC2, LA, GL, TA, BL, NL, CO, UM, CG`, using the permutation
`[0, 1, 2, 5, 3, 4, 6, 7, 9, 8, 10, 11]` from model-output order. This is a
reporting/presentation-only reorder; it does not change manifests, labels,
predictions, per-class AP values, or macro mAP.

All methods reported in the paper's LDXray comparison use the same converted
manifests. Manifest paths are relative to the root of an official LDXray data
download. Training and validation manifests both reference disjoint pairs in
`dataset/train_A` and `dataset/train_B`; no separate validation image directory
is required. The split seed 42 is independent of the model training seed 3408.
No raw LDXray images are included here.
