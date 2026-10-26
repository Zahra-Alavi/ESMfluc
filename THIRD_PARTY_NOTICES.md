# Third-party software and data

MIT applies to the ESMfluc authors' code and trained weights, not to every dependency or input dataset. Raw ATLAS, ECOD, PEGASUS and DynaMine source copies are not distributed here; obtain them from their providers. Preserve their attribution and consult the applicable terms before reuse.

| Source | Use and attribution |
| --- | --- |
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) | Molecular-dynamics simulations and Protein Block targets; cite the ATLAS paper. This repository supplies identifiers, split assignments and derived results, not a mirror of the source database. |
| [mdCATH](https://github.com/compsciencelab/mdCATH) | Independent simulation benchmark. mdCATH-derived tables and the domain list (`data/mdcath/domains.txt`) are included under CC-BY-4.0 with attribution to its authors; retrieve trajectories separately. `esmfluc/datasets/mdcath_convert.py` is derived from the mdCATH user utilities (MIT, Copyright (c) 2024 The authors). |
| [PBxplore](https://github.com/pierrepo/PBxplore) | Protein Block assignment for the mdCATH Neq targets. MIT. Installed as a dependency, not bundled. |
| [Protein Data Bank](https://www.wwpdb.org/about/usage-policies) | Experimental-coordinate-derived features; PDB data are CC0. Cite the relevant structure and PDB resources. The guanylate-kinase example uses 1ZNX, chain A. |
| [ECOD](http://prodata.swmed.edu/ecod/) | Domain classifications used for grouping and split separation. Retrieve the recorded release; raw domain annotations are not mirrored here. Cite ECOD. |
| [NetSurfP-3.0](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/) | Secondary structure, solvent exposure and other residue annotations were obtained from the web server. Raw server exports remain private; regeneration and validation are described in the analysis guide. The standalone program is not bundled. |
| [PEGASUS](https://doi.org/10.1002/pro.70221) | External dynamics predictions; cite Vander Meersche et al., Protein Science (2025), and retrieve code/models from the provider identified by the benchmark workflow. Only derived benchmark results are included. |
| [DynaMine / Bio2Byte](https://bio2byte.be/dynamine/) | External backbone dynamics scores; cite DynaMine. The retained client submits sequences to Bio2Byte's service. Raw prediction caches are not distributed. |
| ESM2 | Upstream MIT notice is retained in `release/licenses/ESM2-MIT.txt`. |
| ESM3 | Retained current MIT license, model card and historical license index are in `release/licenses/`. These notices distinguish the verified current weight file from earlier licensing terms; see `NOTICE.txt` for revision and checksum. Upstream file: `biohub/esm3-sm-open-v1`, revision `47f0545b2b6daf26a93439a3cd610f4f7f3d5478`, `data/weights/esm3_sm_open_v1.pth`, SHA-256 `5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9`. |
| [MDAnalysis](https://www.mdanalysis.org/) | GPL-2.0-or-later. Installed as a dependency, not bundled source; its license is not replaced by ESMfluc's MIT license. |
| MDStrainMapper | Separate software used to produce the strain annotations. Version v1.0 (commit `bbe8efc`) was used; see `provenance/mdstrainmapper.json`. Source is not bundled. |

Other installed libraries retain their respective licenses. The older pinned ESM SDK's software terms must not be confused with the verified current upstream weight-file license. This repository does not relicense third-party software or data.
