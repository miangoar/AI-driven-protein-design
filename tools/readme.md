# Tools

Tool categories:
- [Sequence-level analysis](#sequence-level-analysis)
- [Multiple sequence alignment](#multiple-sequence-alignment)
- [Phylogenetic analysis](#phylogenetic-analysis)
- [Structure-level analysis](#structure-level-analysis)
- [Data access](#data-access)
- [Structure search & comparison](#structure-search--comparison)
- [Structure prediction](#structure-prediction)
- [Structure generation](#structure-generation)
- [Protein design](#protein-design)
- [Representation learning](#representation-learning)
- [Sequence generation](#sequence-generation)
- [Inverse folding](#inverse-folding)
- [Function prediction](#function-prediction)
- [Molecular simulation](#molecular-simulation)
- [Biological data visualization](#biological-data-visualization)
- [Molecules & cheminformatics](#molecules--cheminformatics)
- [Machine learning](#machine-learning)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Natural Language Processing](#natural-language-processing)
- [Deep learning frameworks](#deep-learning-frameworks)
- [Tabular data](#tabular-data)
- [Hardware-accelerated computation](#hardware-accelerated-computation)
- [Statistics](#statistics)
- [Data visualization](#data-visualization)
- [Chatbots and agents](#chatbots-and-agents)



# Sequence-level analysis
Tools for sequence manipulation, search, comparison, and analysis of DNA, RNA, and protein sequences, including classical and language-model-based approaches
| Name | Description | 
|-----------|-----------| 
| [SeqKit](https://bioinf.shenwei.me/seqkit/) | FASTA/Q file manipulation (Check out this ([tutorial](https://sandbox.bio/tutorials/seqkit-intro))|
| [Diamond2](https://github.com/bbuchfink/diamond) | accelerated BLAST  |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | ultra fast and sensitive search and clustering suite (Check out the  [tutorial](https://github.com/soedinglab/MMseqs2/wiki/Tutorials) and the  [GPU implementation](https://github.com/soedinglab/MMseqs2/wiki#compile-from-source-for-linux-with-gpu-support)) |
| [seqlike](https://github.com/modernatx/seqlike) |  sequence manipulation |
| [BioNumpy](https://github.com/bionumpy/bionumpy/) | array programming on biological datasets |
| [scikit-bio](https://github.com/scikit-bio/scikit-bio) | sequence analysis, phylogenetic trees, and multivariate statistics for ecological and omic data. |
| [pLM-BLAST](https://github.com/labstructbioinf/pLM-BLAST) | remote homology detection with protein language models |  
| [PLMSearch](https://github.com/maovshao/PLMSearch) | homologous protein search with protein language models |
| [LexicMap](https://github.com/shenwei356/LexicMap) | sequence alignment against millions of genomes |
| [pyfastx](https://github.com/lmdu/pyfastx) | fast random access to sequences from plain and gzipped FASTA/Q files |
| [any2fasta](https://github.com/tseemann/any2fasta) | Convert various sequence formats to FASTA |
| [Spacedust](https://github.com/soedinglab/Spacedust) | identification of conserved gene clusters among genomes based on homology and conservation of gene neighborhood |
| [ugene](https://ugene.net/) | genome analysis suite with graphic user interface |
| [BuddySuite](https://github.com/biologyguy/BuddySuite) | manipulating sequence, alignment, and phylogenetic tree files |
| [biotite](https://www.biotite-python.org/latest/) | sequence and structure manipulation and analysis |
| [MPI Bioinformatics Toolkit!](https://toolkit.tuebingen.mpg.de/) | multiple bioinformatics tools |

# Multiple sequence alignment
Tools for building, trimming, evaluating, and manipulating multiple sequence alignments, from traditional algorithms to embedding-based methods
| Name | Description | 
|-----------|-----------| 
| [hh-suite](https://bioinf.shenwei.me/seqkit/) | remote homology detection  |
| [DeepMSA](https://zhanggroup.org/DeepMSA/) | create high-quality MSAs |
| [NEFFy](https://github.com/Maryam-Haghani/NEFFy) | calculating the Normalized Effective Number of Sequences (neff) for protein/nt MSAs. Also for format conversion |
| [PLMAlign](https://github.com/maovshao/PLMAlign) | create MSAs using per-residue embeddings from protein language models |
| [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) | trimming algorithm for accurate phylogenomic inference and msa manipulation |
| [CIAlign](https://github.com/KatyBrown/CIAlign) | clean, interpret, visualise and edit MSAs |
| [TWILIGHT](https://github.com/TurakhiaLab/TWILIGHT) | ultrafast and ultralarge MSA |
| [termal](https://github.com/sib-swiss/termal) |  examining MSAs in a terminal |

# Phylogenetic analysis
Tools for phylogenetic inference, tree reconstruction, and evolutionary analysis using sequences, alignments, and structural information
| Name | Description | 
|-----------|-----------|
| [automlst2](https://automlst2.ziemertlab.com/index) | automatic generation of species phylogeny with reference organisms |
| [iqtree3](https://github.com/iqtree/iqtree3)| Phylogenetic analysis | 
| [piqtree](https://github.com/iqtree/piqtree) | IQ-TREE ported to Python |
| [torchtree](https://github.com/4ment/torchtree) | probabilistic framework in PyTorch for phylogenetic models |
| [fold_tree](https://github.com/DessimozLab/fold_tree) | construct trees from protein structures |
| [3diphy](https://github.com/nmatzke/3diphy) | Maximum likelihood structural phylogenetics by including Foldseek 3Di characters |
| [PhyKIT](https://github.com/JLSteenwyk/PhyKIT) | toolkit for processing and analyzing MSAs and phylogenies |
| [unicore](https://github.com/steineggerlab/unicore) | phylogenetic reconstruction with structural core genes using Foldseek and ProstT5 |

# Structure-level analysis
Tools for analysis, manipulation, validation, and characterization of protein structures, including dynamics, interfaces, pockets, and interactions
| Name | Description | 
|-----------|-----------| 
| [PyRosetta](https://github.com/RosettaCommons/PyRosetta.notebooks) | Rosetta ported to python (check out the installation guides for [Conda and Google Colab](https://x.com/miangoar/status/1835176497063030798) as well as the [documentation](https://graylab.jhu.edu/PyRosetta.documentation/index.html)) |
| [protkit](https://github.com/silicogenesis/protkit) | Unified Approach to Protein Engineering |
| [ProDy](http://prody.csb.pitt.edu/tutorials/) | protein structure, dynamics, and sequence analysis |
| [BioPandas](https://biopandas.github.io/biopandas/) | working with molecular structures in pandas |
| [foldcomp](https://github.com/steineggerlab/foldcomp) | compressing protein structures |
| [protestar](https://github.com/refresh-bio/protestar) | compress collections structures |
| [afpdb](https://github.com/data2code/afpdb) |  manipulation of protein structures in Python |
| [LocalPDB](https://github.com/labstructbioinf/localpdb) | manage structures and their annotations |
| [pdb-tools](https://github.com/haddocking/pdb-tools) | manipulating and editing PDB files |
| [pdbfixer](https://github.com/openmm/pdbfixer) | fixes problems in PDB files |
| [cath-tools](https://github.com/UCLOrengoGroup/cath-tools) | structure comparison tools |
| [Merizo-search](https://github.com/psipred/merizo_search) | domain structure embedding+search tool |
| [freesasa](https://github.com/mittinatten/freesasa) | calculating Solvent Accessible Surface Areas |
| [openstructure](https://openstructure.org/) | protein structure, complexes and docking comparison |
| [opendock](https://github.com/guyuehuo/opendock)| protein-Ligand Docking and Modeling |
| [pdb-redo](https://pdb-redo.eu/)| automated procedure to refine, rebuild and validate your models |
| [proteinshake](https://github.com/BorgwardtLab/proteinshake) | preprocessed and cleaned structure datasets |
| [GEMMI](https://github.com/project-gemmi/gemmi) | macromolecular crystallography library and utilities |
| [fpocket](https://github.com/Discngine/fpocket) | protein pocket detection based on Voronoi tessellation |
| [pocketeer](https://github.com/cch1999/pocketeer) | A lightweight, fast pocket finder |
| [dssp](https://github.com/PDB-REDO/dssp) | assign secondary structure to proteins (check out [the webserver](https://pdb-redo.eu/dssp)) |
| [PyDSSP](https://github.com/ShintaroMinami/PyDSSP) | PyTorch implementation of DSSP algorithm |
| [PDBCleanV2](https://github.com/fatipardo/PDBCleanV2) | create a curated ensemble of molecular structures |
| [ProLIF](https://github.com/chemosim-lab/ProLIF) | Interaction Fingerprints for protein-ligand complexes |
| [plip](https://github.com/pharmai/plip) | Analyze and visualize non-covalent protein-ligand and protein-protein interactions |
| [af_analysis](https://github.com/samuelmurail/af_analysis) | Analysis of alphafold and colabfold results |
| [InteracTor](https://github.com/Dias-Lab/InteracTor) |  structure analysis and conversion, allowing the extraction of molecular interactions (e.g.  Hbonds, van der Waals, hydrophobic contacts, and surface tension |
| [ProteinTools ](https://proteintools.uni-bayreuth.de/) | Analyze Hydrophobic Clusters, Hydrogen Bond Networks, Contact maps, Salt Bridges and Charge Segregation |
| [libraryPDB](https://github.com/CJ438837/libraryPDB) | searching, downloading, parsing, cleaning and analyzing protein structures |
| [lahuta](https://bisejdiu.github.io/lahuta/) | calculate atomic interactions |
| [InterfaceAnalyzerMover](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/analysis/InterfaceAnalyzerMover) | Calculate binding energies, buried interface surface areas, packing statistics, and other useful interface metrics for the evaluation of protein interfaces |
| [Relax your structure using OpenMM/Amber](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/relax_amber.ipynb) | Relax your structure |
| [ProteinsPlus](https://proteins.plus/) | structure mining and modeling, focussing on protein-ligand interactions |
| [CoEVFold](https://github.com/MishterBluesky/CoEVFold) | analyze protein co-evolution e.g. generating contact maps, gene networks, identifying homomeric and heteromeric interfaces |


# Data access
Tools for downloading and mapping biological data from public databases like NCBI, PDB, UniProt, and AlphaFold DB
| Name | Description | 
|-----------|-----------| 
| [iSeq](https://github.com/BioOmics/iSeq) | download data from sequence databases like GSA, SRA, ENA, and DDBJ |
| [NCBI datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/) | download data from NCBI  |
| [ncbi-genome-download](https://github.com/kblin/ncbi-genome-download) | download data from the NCBI  |
| [ncbi-acc-download](https://github.com/kblin/ncbi-acc-download) | download data from NCBI Entrez by accession |
| [Pyrcsbsearchapi](https://github.com/rcsb/py-rcsbsearchapi) | Python interface for the RCSB PDB search API |
| [PyPDB](https://github.com/williamgilpin/pypdb) | python API for the PDB |
| [profet](https://github.com/alan-turing-institute/profet) | retrieves the cif or pdb files from either the PDB (using pypdb) or Alphafold using the Uniprot ID |
| [ActSeek](https://github.com/vttresearch/ActSeek) | mining in the AFDB based on the position of few amino acids |
| [Uniprot-PDB-mapper](https://github.com/iriziotis/Uniprot-PDB-mapper) | mapping of Uniprot sequences to PDB (see also this option to [map IDs between RefSeq and Uniprot](https://ncbiinsights.ncbi.nlm.nih.gov/2023/11/08/compare-ncbi-refseq-and-uniprot-datasets/) using the file `gene_refseq_uniprotkb_collab.gz`)|
| [PDBe API Training Notebooks](https://github.com/glevans/pdbe-api-training) | understand how the PDBe REST API works |
| [RCSB API](https://github.com/rcsb/py-rcsb-api) | Python interface for RCSB PDB API services (check out the [guidelines and tutorials](https://pdb101.rcsb.org/news/684078fe300817f1b5de793a)) |
| [AFDB Structure Extractor](https://project.iith.ac.in/sharmaglab/alphafoldextractor/index.html) | download structures using AF IDs, Uniprot IDs, Locus tags, RefSeq Protein IDs and NCBI TaxIDs |

# Structure search & comparison
Tools for protein structure search, alignment, and comparison using geometric, topological, and embedding-based representations.
| Name | Description | 
|-----------|-----------| 
| [Pairwise Structure Alignment Tool](https://www.rcsb.org/alignment) | webserver for structure alignment using PDB IDs, AFDB IDs or local files (check out the detailed description of [how to use the tool](https://www.rcsb.org/docs/tools/pairwise-structure-alignment#4-align-multiple-structures-to-an-alphafold-structure-))|
| [FoldSeek](https://github.com/steineggerlab/foldseek) | fast and sensitive comparisons of large structure sets (check out [Foldseek clusters](https://cluster.foldseek.com/) to perform a representative structure seach)|
| [folddisco](https://github.com/steineggerlab/folddisco) | indexing and search of discontinuous motifs |
| [USalign](https://github.com/pylelab/USalign) |structure alignment of monomeric and complex proteins and nucleic acids |
| [progres](https://github.com/greener-group/progres) | structure searching by structural embeddings (check out the [webserver](https://progres.mrc-lmb.cam.ac.uk/))|
| [pyScoMotif](https://github.com/3BioCompBio/pyScoMotif) | protein motif search |
| [pyRMSD](https://github.com/salilab/pyRMSD) | RMSD calculations of large sets of structures |
| [reseek](https://github.com/rcedgar/reseek) | structure alignment and search algorithm (check out the [webserver](https://reseek.online/))|
| [tmtools](https://github.com/jvkersch/tmtools) | Python bindings for the TM-align algorithm and code for protein structure comparison |
| [SoftAlign](https://github.com/jtrinquier/SoftAlign) | compare 3D protein structures |
| [foldmason](https://github.com/steineggerlab/foldmason) | Multiple Protein Structure Alignment at Scale |
| [pyjess](https://github.com/althonos/pyjess) | constraint-based structural template matching to identify catalytic residues from a known template |
| [gtalign](https://bioinformatics.lt/comer/gtalign/) | High-performance search and alignment for protein structures |
| [Muscle-3D](https://github.com/rcedgar/muscle) | multiple protein structure alignment |
| [PAthreader](http://zhanglab-bioinf.com/PAthreader/) | remote homologous template recognition |
| [alphafind](https://alphafind.fi.muni.cz/search) | structure-based search of the entire AFDB using Uniprot ID, PDB ID, or Gene Symbol |

# Structure prediction
Tools for protein structure prediction and biomolecular complexes
| Name | Description | 
|-----------|-----------| 
| [Alphafold2](https://github.com/google-deepmind/alphafold)| protein structure prediction |
| [Alphafold3](https://github.com/google-deepmind/alphafold3) | predict biomolecular interactions using AlphaFold3 (check out the [webserver](https://alphafoldserver.com/) as well as this [solution to predict ~10k structures](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/cloudnext25/examples/science/af3-slurm/README.md))|
| [ColabFold](https://github.com/sokrypton/ColabFold)| protein structure prediction on Google colab with a graphical user interface|
| [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) | ColabFold on your local PC | 
| [ESMFold](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold) | structure prediction with protein language models (check out the [webserver](https://esmatlas.com/resources?action=fold)) |
| [flashfold](https://github.com/chayan7/flashfold) | command-line tool for faster protein structure prediction |
| [PAE Viewer](https://gitlab.gwdg.de/general-microbiology/pae-viewer) | view the predicted aligned error (PAE) of multimers, and integrates visualization of crosslink data (check out the [webserver](https://subtiwiki.uni-goettingen.de/v4/paeViewerDemo)) |
| [PyMOLfold](https://github.com/colbyford/PyMOLfold) | Plugin for folding sequences directly in PyMOL |
| [AFsample2](https://github.com/iamysk/AFsample2/) |  induce significant conformational diversity for a given protein |
| [alphafold3 tools](https://github.com/cddlab/alphafold3_tools) | Toolkit for input generation and output analysis |
| [af3cli](https://github.com/SLx64/af3cli) | generating AlphaFold3 input files |
| [RareFold](https://github.com/patrickbryant1/RareFold) | Structure prediction and design of proteins with 29 noncanonical amino acids |
| [AFusion](https://github.com/Hanziwww/AlphaFold3-GUI) | AlphaFold 3 GUI & Toolkit with Visualization |
| [Hackable AlphaFold 3](https://github.com/chaitjo/alphafold3/) | a lightweight, hackable way to run AF3 to experiment without the massive MSA databases or Docker overhead |
| [AF2 single sequence input](https://twitter.com/sokrypton/status/1535857255647690753) | Optimized lightweight version of AlphaFold2 enabling rapid structure prediction at reduced accuracy, intended primarily for educational and exploratory use |
| [ABCFold](https://github.com/rigdenlab/ABCFold) | Scripts to run AlphaFold3, Boltz-1 and Chai-1 with MMseqs2 MSAs and custom templates  |
| [AlphaPulldown](https://www.embl-hamburg.de/AlphaPulldown/) | Complex modeling with AF-Multimer |
| [LazyAF](https://github.com/ThomasCMcLean/LazyAF) | protein-protein interaction with AF2 |
| [CombFold](https://github.com/dina-lab3D/CombFold) | structure predictions of large complexes |
| [Cfold](https://github.com/patrickbryant1/Cfold) | structure prediction of alternative protein conformations |
| [AF_unmasked](https://github.com/clami66/AF_unmasked) | structure prediction for huge protein complexes (~27 chains and ~8400aa) |
| [ConservFold](https://www.rodrigueslab.com/resources) | map residue conservation into structures with AF2 (check out the notebook for [multimers](https://colab.research.google.com/drive/1Lv-akfLE7kTCFCWaEyHAtsPCeXYD3xvH?usp=sharing)) |
| [IPSAE](https://github.com/DunbrackLab/IPSAE) | Scoring function for interprotein interactions in AlphaFold2 and AlphaFold3 |
| [peppr](https://github.com/aivant/peppr) | a package for evaluation of predicted poses like RMSD, TM-score, lDDT, lDDT-PLI, fnat, iRMSD, LRMSD, DockQ  |

# Structure generation 
Tools for creating novel protein structures or complexes, with or without functional or structural conditioning
| Name | Description | 
|-----------|-----------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) |  structure generation, with or without conditional information (see also the [extended Documentation](https://sites.google.com/omsf.io/rfdiffusion) with a lot of descriptions and tutorials)|
| [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) | all-atom version at sub-angstrom resolution of RFdiffusion |
| [chroma](https://github.com/generatebio/chroma) | programmable protein design |
| [protein_generator](https://github.com/RosettaCommons/protein_generator) | Joint sequence and structure generation with RoseTTAFold sequence space diffusion |
| [RFdiffusion_all_atom](https://github.com/baker-laboratory/rf_diffusion_all_atom) | RFdiffusion with all atom modeling |
| [salad](https://github.com/mjendrusch/salad) | structure generation with sparse all-atom denoising models |
| [EnzymeFlow](https://github.com/WillHua127/EnzymeFlow) | generate catalytic pockets for specific substrates and catalytic reactions |
| [GENzyme](https://github.com/WillHua127/GENzyme) | design of catalytic pockets, enzymes, and enzyme-substrate complexes for any reaction |
| [BoltzDesign1](https://github.com/yehlincho/BoltzDesign1) |  designing protein-protein interactions and biomolecular complexes |

# Protein design
Tools for AI-driven protein design, including binders, stability optimization, and functional engineering
| Name | Description | 
|-----------|-----------| 
| [GraphRelax](https://github.com/delalamo/GraphRelax) | residue repacking and design   |
| [SCHEMA-RASPP](https://github.com/mattasmith/SCHEMA-RASPP) | structure-guided protein recombination (download the file [`schema-tools-doc.html`](https://github.com/mattasmith/SCHEMA-RASPP/blob/master/schema-tools-doc.html) for documentation)|
| [ProtLego](https://hoecker-lab.github.io/protlego/) | constructing protein chimeras and its structural analysis |
| [FoldPAthreader](http://zhanglab-bioinf.com/PAthreader/) | folding pathway prediction | 
| [consurf](https://consurf.tau.ac.il/consurf_index.php) |identification of functionally important regions in proteins by conservation |
| [ColabDock](https://github.com/JeffSHF/) | protein-protein docking |
| [ColabDesign](https://github.com/sokrypton/ColabDesign) | protein design pipelines |
| [Replacement Scan](https://colab.research.google.com/github/sokrypton/ColabBio/blob/main/notebooks/replacement_scan.ipynb) | find how many residue replacements your protein can tolerate (Check out the [announcement](https://x.com/sokrypton/status/1812769477228200086)) |
| [TRILL](https://github.com/martinez-zacharya/TRILL) | Generative AI Sandbox with popular models |
| [PyPEF](https://github.com/Protein-Engineering-Framework/PyPEF) | sequence-based machine learning-assisted protein engineering |
| [DeepProtein](https://github.com/jiaqingxie/DeepProtein) | protein Property Prediction |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | binder design  (check out the [tutorial](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)) |
| [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) | BindCraft modified to make PyRosetta use and installation optional, i.e. no license needed ([more details](https://www.ariax.bio/resources/freebindcraft-open-source-unleashed))|
| [damietta](https://damietta.de/) | protein design toolkit |
| [prosculpt](https://github.com/ajasja/prosculpt) | encapsulates RFDiffusion, ProteinMPNN, AlphaFold2, and Rosetta into an easy-to-use workflow |
| [BinderFlow](https://github.com/cryoEM-CNIO/BinderFlow) | parallelised pipeline for protein binder design (i.e. RFD > ProteinMPNN > AF2 + Scoring) | 
| [proteindj](https://github.com/PapenfussLab/proteindj) | pipeline for de novo binder design (i.e. RFD > ProteinMPNN > AF2 + Scoring) |
| [ovo](https://github.com/MSDLLCpapers/ovo) | ecosystem for de novo protein design |
| [bagel](https://github.com/softnanolab/bagel) | model-agnostic and gradient-free exploration of an energy landscape in the sequence space |
| [Protein Design Skills](https://proteinbase.com/protein-design-skills) | Claude Code skills for protein design |
| [ProtFlow](https://github.com/mabr3112/ProtFlow) | automate protein design workflows with a python wrapper around common protein design tools (check out this [tutorial](https://www.youtube.com/watch?v=Rji1WPt_gig)) |

# Representation learning
Tools for computing vectorial representations of protein sequences and structures for downstream tasks
| Name | Description | 
|-----------|-----------| 
| [ESM](https://github.com/facebookresearch/esm) | protein language models from the ESM family |
| [ProtTrans](https://github.com/agemagician/ProtTrans)| protein language models from the ProtTrans family (check out [ProtT5-EvoTuning](https://github.com/RSchmirler/ProtT5-EvoTuning) and [Fine-Tuning](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning) for LoRA fine-tunning tutorials) |
| [FAESM](https://github.com/pengzhangzhi/faesm) | Pytorch Implementation of ESM and ProGen that can save ~60% of memory usage and 70% of inference time | 
| [ESM-Efficient](https://github.com/uci-cbcl/esm-efficient) | Efficient implementation of ESM family | 
| [finetune-esm](https://github.com/naity/finetune-esm) | Finetuning with Distributed Learning and Advanced Training Techniques | 
| [ProtLearn](https://github.com/tadorfer/protlearn) | extracting protein sequence features |
| [Pfeature](https://github.com/raghavagps/Pfeature) | computing features of peptides and proteins |
| [bio_embeddings](https://github.com/sacdallago/bio_embeddings) | compute protein embeddings from sequences |
| [Graph-Part](https://github.com/graph-part/graph-part) | data partitioning method for ML |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow) | data processing pipeline for ML |
| [docktgrid](https://github.com/gmmsb-lncc/docktgrid) | Create customized voxel representations of protein-ligand complexes |
| [Prop3D](https://github.com/bouralab/Prop3D) | toolkit for protein structure dataset creation and processing  |
| [SaProt](https://github.com/westlake-repl/SaProt) | Protein Language Model with Structural Alphabet |
| [SaprotHub](https://github.com/westlake-repl/SaprotHub) | platform that facilitates training, fine-tuning and prediction as well as storage and sharing of models |
| [ProstT5](https://github.com/mheinzinger/ProstT5) | Bilingual Language Model for Protein Sequence and Structure (check out the [Foldseek adaptation](https://github.com/steineggerlab/foldseek?tab=readme-ov-file#structure-search-from-fasta-input)) |
| [Graphein](https://github.com/a-r-j/graphein) | geometric representations of biomolecules and interaction networks |
| [PyUUL](https://pyuul.readthedocs.io/index.html) | encode structures into differentiable data structures |
| [colav](https://github.com/Hekstra-Lab/colav) | feature extraction methods like dihedral angles, CA pairwise distances, and strain analysis |
| [masif](https://github.com/LPDI-EPFL/masif) | molecular surface interaction fingerprints |
| [peptidy](https://github.com/molML/peptidy) | vectorize proteins for machine learning applications |
| [pypropel](https://github.com/2003100127/pypropel) | sequence and structural data preprocessing, feature generation, and post-processing for model performance evaluation and visualisation |
| [atomworks](https://github.com/RosettaCommons/atomworks) | A generalized computational framework for biomolecular modeling |
| [mini3di](https://github.com/althonos/mini3di) | NumPy port of the foldseek code for encoding protein structures to 3di |

# Sequence generation
Tools for designing protein sequences 
| Name | Description | 
|-----------|-----------|
| [ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) | conditional language model for the generation of artificial functional enzymes |
| [REXzyme_aa](https://huggingface.co/AI4PD/REXzyme_aa) | generate sequences that are predicted to perform their intended reactions |
| [ProGen2-finetuning](https://github.com/hugohrban/ProGen2-finetuning) | Finetuning ProGen2 for generation of sequences from selected families |
| [Pinal](https://github.com/westlake-repl/Denovo-Pinal) | Text-guided protein generation |
| [ProtRL](https://github.com/AI4PDLab/ProtRL) | Reinforcement Learning framework for autoregressive protein Language Models |
| [protein_scoring](https://github.com/seanrjohnson/protein_scoring) | generating and scoring novel enzyme sequences  |
| [AncFlow](https://github.com/rrouz/AncFlow) | pipeline for the ancestral sequence reconstruction of clustered phylogenetic subtrees |
| [ByProt (LM-Design)](https://github.com/bytedprotein/ByProt) | reprogramming pretrained protein LMs as generative models |
| [EvoProtGrad](https://github.com/NREL/EvoProtGrad) | directed evolution with MCMC and protein language models |

# Inverse folding
Tools for designing protein sequences given protein backbone or structure
| Name | Description | 
|-----------|-----------|
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Fixed backbone design ([see webserver](https://huggingface.co/spaces/simonduerr/ProteinMPNN))|
| [ProteinMPNN in JAX](https://github.com/sokrypton/ColabDesign/tree/main/mpnn) | Fast implementation of ProteinMPNN |
| [ligandMPNN](https://github.com/dauparas/LigandMPNN) | Fixed backbone design sensible to ligands ([see colab notebook](https://github.com/ullahsamee/ligandMPNN_Colab))|
| [SolubleMPNN](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights) | Retrained version of ProteinMPNN by excluding transmembrane structures (see [Goverde et al. 2024](https://www.nature.com/articles/s41586-024-07601-y#Sec7) for more details)|
| [fampnn](https://github.com/richardshuai/fampnn) | full-atom version of ProteinMPNN |
| [LASErMPNN ](https://github.com/polizzilab/LASErMPNN) | All-Atom (Including Hydrogen!) Ligand-Conditioned Protein Sequence Design & Sidechain Packing Model |
| [HyperMPNN](https://github.com/meilerlab/HyperMPNN) | design thermostable proteins learned from hyperthermophiles |
| [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) | predict changes in thermodynamic stability for protein point mutants |
| [ByProt](https://github.com/BytedProtein/ByProt) | Efficient non-autoregressive ProteinMPNN variant |
| [Caliby](https://github.com/ProteinDesignLab/caliby) | Potts model-based protein sequence design method that can condition on structural ensembles |
| [PottsMPNN](https://github.com/KeatingLab/PottsMPNN) | generate sequences and predict energies of mutations |



# Function prediction
Tools for functional annotation, active-site detection, ligand binding prediction, and  other properties
| Name | Description | 
|-----------|-----------|
| [ProtNLM](https://www.uniprot.org/help/ProtNLM) | UniProt's Automatic Annotation pipeline for protein sequences (check out this notebooks for the ([2023_01_version](https://colab.research.google.com/github/google-research/google-research/blob/master/protnlm/protnlm_evidencer_uniprot_2023_01.ipynb)) and [2022_04_version](https://colab.research.google.com/github/google-research/google-research/blob/master/protnlm/protnlm_use_model_for_inference_uniprot_2022_04.ipynb)) |
| [CLEAN](https://github.com/tttianhao/CLEAN) | assign EC numbers to enzymes |
| [DeepFRI](https://github.com/flatironinstitute/DeepFRI) | Deep functional residue identification |
| [interproscan](https://github.com/ebi-pf-team/interproscan) | interpro pipeline for functional annotation with multiple DBs |
| [ProTrek](https://github.com/westlake-repl/ProTrek) | multimodal (sequence-structure-function) protein representations and annotations (check out the [webserver](http://search-protrek.com/)) |
| [HiQBind](https://github.com/THGLab/HiQBind) | Workflow to clean up and fix structural problems in protein-ligand binding datasets |
| [PandaDock](https://github.com/pritampanda15/PandaDock) | Physics-Based Molecular Docking |
| [ligysis](https://www.compbio.dundee.ac.uk/ligysis/) | analysis of biologically meaningful ligand binding sites |
| [AF2BIND](https://github.com/sokrypton/af2bind) | Predicting ligand-binding sites based on AF2 |
| [p2rank](https://github.com/rdk/p2rank) | Protein-ligand binding site prediction from protein structure |
| [PLACER](https://github.com/baker-laboratory/PLACER) |  local prediction of protein-ligand conformational ensembles |
| [easifa](http://easifa.iddd.group/) | active and binding site annotations for enzymes |

# Molecular simulation
Tools for molecular dynamics, physical modeling, and analysis of trajectories
| Name | Description | 
|-----------|-----------|
| [making-it-rain](https://github.com/pablo-arantes/making-it-rain) | Cloud-based molecular simulations for everyone |
| [bioemu](https://github.com/microsoft/bioemu) |  emulation of protein equilibrium ensembles  (check out [this notebook](https://colab.research.google.com/github/pablo-arantes/making-it-rain/blob/main/BioEmu_HPACKER.ipynb) from Make it rain that combines bioemu with [H-Packer](https://github.com/gvisani/hpacker) for side-chain reconstruction)| 
| [orb](https://github.com/orbital-materials/orb-models) | forcefield models from Orbital Materials |
| [logMD](https://github.com/log-md/logmd) | visualize MD trajectories in colab |
| [proprotein](https://proprotein.cs.put.poznan.pl/) | set up, configure, and run an MD simulations |
| [packmol](https://github.com/m3g/packmol) | creates an initial point for MD simulations |
| [mdanalysis](https://github.com/MDAnalysis/mdanalysis) | analyze molecular dynamics simulations |
| [VMD-2](https://www.ks.uiuc.edu/Research/vmd/vmd2intro/index.html) | Visual Molecular Dynamics |
| [gromacs_copilot](https://github.com/ChatMol/gromacs_copilot) | AI-powered assistant for Molecular Dynamics simulations |
| [moleculatio](https://moleculatio.yamlab.app) | chemoinformatics, quantum chemistry and molecular dynamics simulations or small molecules |


# Biological data visualization
Tools for visualization of proteins, structures, alignments, phylogenies, and molecular data
| Name | Description | 
|-----------|-----------| 
| [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) | Toolbox for molecular animations in Blender |
| [prettymol](https://github.com/zachcp/prettymol) | automatic protein structure plots with MolecularNodes  |
| [CellScape](https://github.com/jordisr/cellscape) | Vector graphics cartoons from protein structure |
| [chimerax](https://www.rbvi.ucsf.edu/chimerax/)| molecular visualization program (check out the [ChimeraX apps](https://cxtoolshed.rbvi.ucsf.edu/apps/all) as well as this [Color Palettes](https://github.com/smsaladi/chimerax_viridis) and this config to set-up a [Baker lab palette](https://bsky.app/profile/jameslingford.bsky.social/post/3lmw5h3xxec2n))|
| [SSDraw](https://github.com/ncbi/SSDraw) |  secondary structure diagrams from protein structures |
| [bioalphabet](https://github.com/davidhoksza/bioalphabet/) | convertor of texts to bio-domain alphabets |
| [ChatMol](https://github.com/ChatMol/ChatMol) | a PyMOL ChatGPT Plugin that allow to interact with PyMOL using natural language  |
| [plot_phylo](https://github.com/KatyBrown/plot_phylo) | plot a phylogenetic tree on an existing matplotlib axis |
| [NIH bioart](https://bioart.niaid.nih.gov/) | 2,000+ science and medical icons |
| [bioicons](https://bioicons.com/) | icons for science illustrations in biology and chemistry |
| [moldraw](https://moldraw.com/) | draw molecules |
| [Mol* at RCSB/PDB](https://onlinelibrary.wiley.com/doi/10.1002/pro.70093) |  web-based, 3D visualization software suite for examination and analyses of biostructures |
| [MolViewSpec](https://github.com/molstar/mol-view-spec/) | Python toolkit allows for describing views used in molecular visualizations |
| [PoseEdit](https://proteins.plus/) | interactive 2D ligand interaction diagrams (see this [tutorial](https://www.youtube.com/watch?v=8W1TvSvatSA&ab_channel=BioinformaticsInsights)) |
| [FlatProt](https://github.com/t03i/FlatProt) | 2D protein visualization aimed at improving the comparability of structures  |
| [quarto-molstar](https://github.com/jmbuhr/quarto-molstar) | embed proteins and trajectories with Mol* |
| [alphabridge](https://alpha-bridge.eu/) | summarise predicted interfaces and intermolecular interactions |
| [weblogo](https://weblogo.threeplusone.com/) |  generation of sequence logos |
| [interprot](https://interprot.com/#/) | inspect relevant features derived from protein language models in a particular protein |
| [Protein icons](https://bsky.app/profile/maxfus.bsky.social/post/3lobecnwdsc2w) |  How to create specific protein icons with ChatGPT |
| [py2Dmol](https://github.com/sokrypton/py2Dmol) | visualizing biomolecular structures in 2D in Google Colab and Jupyter environments ([check out the website](https://py2dmol.solab.org/))|
| [Nano Protein Viewer](https://marketplace.visualstudio.com/items?itemName=StevenYu.nano-protein-viewer) | protein structure viewer in VScode (try it using [web app](https://stevenyuyy.us/protein-viewer/) and also [check out this tutorial](https://youtu.be/srDyhfhoDm8))|
| [Protein Viewer](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer) | visualization of protein structures and molecular data in VScode |
| [molview](https://github.com/54yyyu/molview) | IPython/Jupyter widget for interactive molecular visualization, based on Molstar |
| [ProteinCHAOS](https://dzyla.github.io/ProteinCHAOS/) | an artistic tool inspired by molecular dynamics to capture protein flexibility over time |

# Molecules & cheminformatics
Tools for small-molecule representation, analysis, docking, and computational chemistry
| Name | Description | 
|-----------|-----------|
| [rdkit](https://www.rdkit.org/docs/index.html) | cheminformatics and machine-learning software (check out this collection of [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) ) |
| [PDBe CCDUtils](https://pdbeurope.github.io/ccdutils/index.html)  | tools to deal with PDB chemical components and visualization ([see also](https://github.com/PDBeurope/pdbe-notebooks/tree/main/pdbe_ligands_tutorials))|
| [PDBe Arpeggio](https://github.com/PDBeurope/arpeggio) |  calculation of interatomic interactions in molecular structures|
| [PDBe RelLig](https://github.com/PDBeurope/rellig) | classifies ligands based on their functional role| 
| [MolPipeline](https://github.com/basf/MolPipeline) | processing molecules with RDKit in scikit-learn |
| [roshambo](https://github.com/molecularinformatics/roshambo) | molecular shape comparison |
| [FlexMol](https://github.com/Steven51516/FlexMol) | construction and evaluation of diverse model architectures |
| [molli](https://github.com/SEDenmarkLab/molli) | general purpose molecule library generation and handling  |
| [deepchem](https://github.com/deepchem/deepchem) | toolkit for drug discovery, materials science, quantum chemistry, and biology |
| [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) | GPU-accelerated library for key computational chemistry (e.g. molecular similarity, conformer generation, and geometry relaxation) |

# Machine learning 
Tools for machine learning
| Name | Description | 
|-----------|-----------|
| [Google colab](https://colab.google/) | cloud computing with free GPUs. Check out this [post](https://colab.research.google.com/notebooks/forms.ipynb) to convert a notebook into a user interface as well as this [VScode extension](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/)| 
| [Best-of Machine Learning](https://github.com/ml-tooling/best-of-ml-python) | list of awesome machine learning Python libraries |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | machine learning built on top of SciPy |
| [LazyPredict](https://github.com/shankarpandala/lazypredict) | build a lot of basic models without much code |
| [PySR](https://github.com/MilesCranmer/PySR) | High-Performance Symbolic Regression |
| [pyod](https://github.com/yzhao062/pyod) | Outlier and Anomaly Detection, Integrating Classical and Deep Learning Techniques |
| [cleanlab](https://github.com/cleanlab/cleanlab) | clean data and labels by automatically |
| [xgboost](https://github.com/dmlc/xgboost) | all you need | 
| [LightGBM](https://github.com/microsoft/LightGBM) | high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithm  | 
| [catboost](https://github.com/catboost/catboost) | high performance Gradient Boosting on Decision Trees | 


https://developers.googleblog.com/google-colab-is-coming-to-vs-code/

# Dimensionality Reduction
Tools for the visualization of high-dimensional data
| Name | Description | 
|-----------|-----------|
| [TorchDR](https://github.com/TorchDR/TorchDR) | PyTorch Dimensionality Reduction |
| [pca](https://erdogant.github.io/pca/pages/html/index.html) | perform PCA and create insightful plots |
| [openTSNE](https://opentsne.readthedocs.io/en/stable/) | faster implementation of t-SNE that includes other optimizations |
| [DADApy](https://github.com/sissa-data-science/DADApy) | characterization of manifolds in high-dimensional spaces |
| [tmap](https://github.com/reymond-group/tmap) | tree-like and fast visualization library for large, high-dimensional data set |
| [umap](https://github.com/lmcinnes/umap) | Uniform Manifold Approximation and Projection | 
| [tsne-cuda](https://github.com/CannyLab/tsne-cuda) | GPU Accelerated t-SNE for CUDA with Python bindings | 


# Natural Language Processing
Tools for text analysis, information extraction, topic modeling, and semantic clustering
| Name | Description | 
|-----------|-----------|
| [unsloth](https://github.com/unslothai/unsloth) | Fine-tuning & Reinforcement Learning for LLM | 
| [BERTopic](https://github.com/MaartenGr/BERTopic) | create clusters for easily interpretable topics |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT) | Minimal keyword extraction with BERT |
| [PolyFuzz](https://github.com/MaartenGr/PolyFuzz) | Fuzzy string matching, grouping, and evaluation |
| [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | accessing, using, and training embedding and reranker models (check out the guide to [Fine-tuning Embedding Models with Unsloth](https://unsloth.ai/docs/new/embedding-finetuning))| 
| [setfit](https://github.com/huggingface/setfit) | Efficient few-shot learning with Sentence Transformers |
| [langextract](https://github.com/google/langextract) | extracting structured information from unstructured text using LLMs |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | mechanistic interpretability of GPT-style language models (check out this [Mech. Interp. cheatsheet](https://jrosser.co.uk/assets/TransformerLens___PyTorch_Quick_Reference.pdf)) |



# Deep learning frameworks 
Tools for building, training, and deploying deep learning models
| Name | Description | 
|-----------|-----------|
| [pytorch](https://github.com/pytorch/pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration | 
| [keras](https://github.com/keras-team/keras) | multi-backend deep learning framework, with support for JAX, TensorFlow, PyTorch, and OpenVINO (check out the [code examples](https://keras.io/examples/) computer Vision, language modeling, generative AI, reinforcement learning, etc.)| 
| [jax](https://github.com/jax-ml/jax) | high-performance numerical computing and large-scale machine learning | 
| [pytorch lightning](https://github.com/Lightning-AI/pytorch-lightning) | Finetune and pretrain any AI model with PyTorch - or build your own | 
| [Kerasify](https://github.com/moof2k/kerasify) | running trained Keras models from a C++ application |
| [autokeras](https://autokeras.com/) | AutoML system based on Keras |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | 100+ PyTorch metrics implementations |
| [skorch](https://github.com/skorch-dev/skorch) | train PyTorch models in a way similar to Scikit-learn (eg. No need to manually write a training loop, just using fit(), predict(), score()) |
| [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) | estimates the GPU vRAM required to train and run inference based on model size, largest layer, training setup, and numerical precision. Use `facebook/esm2_t33_650M_UR50D` as an example for ESM2 650M parameters (check out this other [gpu_poor calculator](https://github.com/RahulSChand/gpu_poor)) |
| [hf-mem](https://github.com/alvarobartt/hf-mem) | CLI to estimate inference memory requirements for Hugging Face models |
| [Do I Have The VRAM?](https://github.com/cneuralnetwork/do-i-have-the-vram) | estimates exactly how much VRAM you need to run a Hugging Face model without downloading it |

# Tabular data
Tools for tabular data
| Name | Description | 
|-----------|-----------|
| [csvtk](https://github.com/shenwei356/csvtk) | CSV/TSV manip MSA (3M protein sequences in 5min and 24GB of RAM) |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | model for tabular data that outperforms traditional methods while being dramatically faster |
| [tabm](https://github.com/yandex-research/tabm) | tabular DL architecture that efficiently imitates an ensemble of MLPs |
| [tabicl](https://github.com/soda-inria/tabicl) | tabular model for classification |
| [skrub](https://github.com/skrub-data/skrub) | preprocessing and feature engineering for tabular machine learning (check out the [learning materials](https://skrub-data.org/skrub-materials/index.html) |
| [dtype_diet](https://github.com/noklam/dtype_diet) | Optimize your memory consumption when using pandas by changing dtypes without data loss  |

# Hardware-accelerated computation
Tools that use the GPU for scientific computing, machine learning, and large-scale similarity search
| Name | Description | 
|-----------|-----------|
| [cuml](https://github.com/rapidsai/cuml) | GPU-based implementations of common machine learning algorithms ([more info for umap optimization](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/) and [cuml.accel](https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/) to boost scikit-learn and other libs in colab)|
| [cupy](https://github.com/cupy/cupy) |NumPy & SciPy for GPU|
| [hummingbird](https://github.com/microsoft/hummingbird) | compiles trained ML models into tensor computation for faster inference |
| [Faiss](https://github.com/facebookresearch/faiss) | efficient similarity search and clustering of dense vectors |
| [einops](https://github.com/arogozhnikov/einops) | tensor operations for readable and reliable code |
| [numba](https://github.com/numba/numba) | NumPy aware dynamic Python compiler  |

# Statistics
Tools for statistics 
| Name | Description | 
|-----------|-----------|
| [scikit-posthocs](https://scikit-posthocs.readthedocs.io/en/latest/) |  post hoc tests for pairwise multiple comparisons |
| [statannotations](https://github.com/trevismd/statannotations) | add statistical significance annotations on seaborn plots |
| [ggstatsplot](https://github.com/IndrajeetPatil/ggstatsplot) | creating graphics with details from statistical tests included in the information-rich plots themselves |
| [ggbetweenstats](https://indrajeetpatil.github.io/ggstatsplot/articles/web_only/ggbetweenstats.html) | making publication-ready plots with relevant statistical details |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration |
| [pingouin](https://pingouin-stats.org/build/html/index.html#) | Statistical package |
| [performance](https://github.com/easystats/performance) | computing indices of regression model quality and goodness of fit  |


# Data visualization
Tools for creating static and interactive plots
| Name | Description | 
|-----------|-----------| 
| [matplotlib](https://github.com/matplotlib/matplotlib) | creating static, animated, and interactive visualizations |
| [seaborn](https://github.com/mwaskom/seaborn) | Statistical data visualization |
| [plotly](https://github.com/plotly/plotly.py) | interactive graphing library |
| [d3blocks](https://github.com/d3blocks/d3blocks) | create stand-alone and interactive d3 charts |
| [holoviews](https://github.com/holoviz/holoviews) | make data analysis and visualization seamless and simple |
| [bokeh](https://github.com/bokeh/bokeh) | Interactive Data Visualization |
| [altair](https://github.com/vega/altair) | declarative statistical visualization library |
| [pypalettes](https://github.com/JosephBARBIERDARNAL/pypalettes) | +2500 color maps  |
| [distinctipy ](https://github.com/alan-turing-institute/distinctipy) |  generating visually distinct colours |
| [bivario](https://github.com/RaczeQ/bivario) | plotting bivariate choropleth maps |
| [morethemes](https://github.com/JosephBARBIERDARNAL/morethemes) | More themes for matplotlib |
| [tidyplots](https://jbengler.github.io/tidyplots/index.html) | creation of publication-ready plots for scientific papers |
| [pyCirclize](https://github.com/moshi4/pyCirclize) | Circular visualization in Python (Circos Plot, Chord Diagram, Radar Chart)  |
| [pycircular](https://github.com/albahnsen/pycircular) | circular data analysis |
| [great-table](https://github.com/posit-dev/great-tables) |  display tables |
| [plottable](https://github.com/znstrider/plottable) | plotting beautifully customized, presentation ready tables |
| [datamapplot](https://github.com/TutteInstitute/datamapplot) | creating beautiful, interactive, annotated and massive scatterplots |
| [jsoncrack](https://github.com/AykutSarac/jsoncrack.com) | transforms data formats such as JSON, YAML, XML, CSV and more, into interactive graphs |
| [torchvista](https://github.com/sachinhosmani/torchvista) | visualize the forward pass of a PyTorch model directly in the notebook |
| [Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network) | set of tools (like [NN-SVG](https://alexlenail.me/NN-SVG/LeNet.html)) to plot neural nets |
| [How to Vectorize Plots from R/Python in PowerPoint](https://nalinan.medium.com/how-to-vectorize-plots-from-r-in-powerpoint-bad7c238e86a) | import a vectorized image into PowerPoint for easy manipulation (check out this [tutorial](https://www.youtube.com/watch?v=hoHkc7N6FZA&ab_channel=GenomicsBootCamp)) |
| [pylustrator](https://github.com/rgerum/pylustrator) | interactive interface to find the best way to present your data in a figure for publication |

# chatbots and agents
Tools for scientific reasoning, literature review, and biological data exploration
| Name | Description | 
|-----------|-----------| 
| [Evolla](https://github.com/westlake-repl/Evolla) | chat about the function of a protein using its sequence and structure  (check out the [webserver](http://www.chat-protein.com/) using the 10B param. version of the model) |
| [NotebookLM](https://notebooklm.google.com/) | organizing, summarizing, and reasoning over your own documents and notes | 
| [ChatGPT](https://chat.openai.com/) | General-purpose conversational AI  |
| [Gemini](https://gemini.google.com/) | General-purpose conversational AI |
| [claude](https://claude.ai/) | General-purpose conversational AI | 
| [HuggingChat](https://huggingface.co/chat/) | General-purpose conversational AI (checkout the [Spaces](https://huggingface.co/spaces) for more models as well as [AI in Biology Demos](https://huggingface.co/collections/hf4h/ai-in-biology-demos-65007d936a230e55a66cd31e) for models used in biology) | 
| [Futurehouse](https://platform.futurehouse.org/) |  agents for literature review and scientific reasoning |
| [biomni](https://github.com/snap-stanford/biomni) | Multimodal AI system for biological data analysis  |
| [OpenScholar](https://openscilm.allen.ai/) | retrieval-augmented LM that answers scientific queries |


