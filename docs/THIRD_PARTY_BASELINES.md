# Literature comparator provenance

HASNet's original single-view, Plain, Feature Fusion, and Late Fusion controls
are implemented in this repository. Literature methods are not copied into the
MIT-licensed tree because their code and datasets may have different terms.

The manuscript adaptations are based on these primary publications:

| Label in manuscript | Primary source |
|---|---|
| CHR / AHCR | Ma et al., *Toward Dual-View X-ray Baggage Inspection*, [IEEE TIFS 2024](https://doi.org/10.1109/TIFS.2024.3372797) |
| DAGNet | Hong et al., *DAGNet*, [IJCNN 2025](https://doi.org/10.1109/IJCNN64981.2025.11227990) |
| SXMNet | Hu et al., *Multi-label X-ray Imagery Classification via Bottom-Up Attention and Meta Fusion*, [ACCV](https://doi.org/10.1007/978-3-030-69544-6_11) |
| DOAM | Wei et al., *Occluded Prohibited Items Detection*, [ACM MM](https://doi.org/10.1145/3394171.3413828) |
| MVCNN | Su et al., *Multi-view Convolutional Neural Networks for 3D Shape Recognition*, [ICCV](https://doi.org/10.1109/ICCV.2015.114) |
| GVCNN | Feng et al., *Group-view Convolutional Neural Networks*, [CVPR](https://doi.org/10.1109/CVPR.2018.00035) |

For a legally complete archival package, publish each adaptation only after
confirming the upstream license, retain attribution and the exact upstream
commit, and document changes needed for synchronized two-view multi-label
classification. Do not silently present copied third-party files as MIT code.

The shared evaluation contract for all comparisons is the fixed DvXray split,
15 independent labels, 256×256 synchronized inputs, the optimizer/schedule in
the main YAML, model selection by validation mAP, and no TTA.
