# Content list

Tools for:
1. [protein sequences](#sequences)
2. [protein structures](#structures)
3. [structure prediction](#structure-prediction)
4. [molecular dynamics](#molecular-dynamics)
5. [representation learning](#representation-learning)
6. [protein engineering](#protein-engineering)
7. [generative AI with difussion models](#generative-AI-with-difussion-models)
8. [generative AI with LLM](#generative-AI-with-LLM)
9. [docking (ligands and proteins)](#docking-ligands-and-proteins)
10. [molecules](#molecules)
11. [machine learning](#machine-learning)
12. [statistics](#statistics)
13. [datavis for bio](#datavis-for-bio)
14. [datavis](#datavis)
15. [webservers](#webservers)
16. [chatbots and agents](#chatbots-and-agents)


# sequences
| Name | Description | 
|-----------|-----------| 
| [SeqKit](https://bioinf.shenwei.me/seqkit/) ([tutorial](https://sandbox.bio/tutorials/seqkit-intro)) | ultrafast toolkit for FASTA/Q file manipulation |
| [hh-suite](https://bioinf.shenwei.me/seqkit/) | remote protein homology detection suite |
| [Diamond2](https://github.com/bbuchfink/diamond) | Accelerated BLAST compatible local sequence aligner |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | ultra fast and sensitive search and clustering suite (see also the [tutorials](https://github.com/soedinglab/MMseqs2/wiki/Tutorials) and [gpu-support](https://github.com/soedinglab/MMseqs2/wiki#compile-from-source-for-linux-with-gpu-support)) |
| [ProDy](http://prody.csb.pitt.edu/tutorials/) | protein structure, dynamics, and sequence analysis |
| [seqlike](https://github.com/modernatx/seqlike) |  Unified biological sequence manipulation |
| [BioNumpy](https://github.com/bionumpy/bionumpy/) | array programming on biological datasets |
| [pLM-BLAST](https://github.com/labstructbioinf/pLM-BLAST) | detection of remote homology by protein language models |  
| [PLMSearch](https://github.com/maovshao/PLMSearch) | homologous protein search with protein language models |
| [iSeq](https://github.com/BioOmics/iSeq) | download data from sequence databases like GSA, SRA, ENA, and DDBJ |
| [LexicMap](https://github.com/shenwei356/LexicMap) | sequence alignment against millions of genomes |
| [NCBI datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/) | download data from NCBI databases |
| [ncbi-genome-download](https://github.com/kblin/ncbi-genome-download) | scripts to download genomes from the NCBI FTP servers |
| [ncbi-acc-download](https://github.com/kblin/ncbi-acc-download) | download files from NCBI Entrez by accession |
| [ProtNLM](https://www.uniprot.org/help/ProtNLM) | UniProt's Automatic Annotation pipeline for protein sequences (see the [Colab notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/protnlm/protnlm_evidencer_uniprot_2023_01.ipynb)) |
| [pyfastx](https://github.com/lmdu/pyfastx) | fast random access to sequences from plain and gzipped FASTA/Q files |
| [DeepMSA](https://zhanggroup.org/DeepMSA/) |  a hierarchical approach to create high-quality multiple sequence alignments |
| [NEFFy](https://github.com/Maryam-Haghani/NEFFy) | calculating the Normalized Effective Number of Sequences (neff) for protein/nt MSAs. Also for format conversion |
| [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) | a MSA-trimming algorithm for accurate phylogenomic inference |
| [PLMAlign](https://github.com/maovshao/PLMAlign) | utilizes per-residue embeddings as input to obtain specific alignments and more refined similarity |
| []() |  |

# Homology search
| Name | Description | 
|-----------|-----------|
| []() |  |


# Download data and metadata
| Name | Description | 
|-----------|-----------|
| []() |  |


# structures
| Name | Description | 
|-----------|-----------| 
| [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) | ColabFold on your local PC | 
| [BioPandas](https://biopandas.github.io/biopandas/) | working with molecular structures in pandas |
| [FoldSeek](https://github.com/steineggerlab/foldseek) | fast and sensitive comparisons of large structure sets|
| [foldcomp](https://github.com/steineggerlab/foldcomp) | Compressing protein structures |
| [Foldmason](https://github.com/steineggerlab/foldmason) | multiple Protein Structure Alignment at Scale |
| [folddisco](https://github.com/steineggerlab/folddisco) | indexing and search of discontinuous motifs |
| [PyPDB](https://github.com/williamgilpin/pypdb) | python API for the PDB |
| [afpdb](https://github.com/data2code/afpdb) | efficient manipulation of protein structures in Python |
| [LocalPDB](https://github.com/labstructbioinf/localpdb) | manage protein structures and their annotations |
| [pdb-tools](https://github.com/haddocking/pdb-tools) | manipulating and editing PDB files |
| [pdbfixer](https://github.com/openmm/pdbfixer) | fixes problems in PDB files |
| [PDBe API Training Notebooks](https://github.com/glevans/pdbe-api-training) | for understanding how the PDBe REST API works |
| [USalign](https://github.com/pylelab/USalign) |structure alignment of monomeric and complex proteins and nucleic acids |
| [cath-tools](https://github.com/UCLOrengoGroup/cath-tools) | structure comparison tools |
| [Pyrcsbsearchapi](https://github.com/rcsb/py-rcsbsearchapi) | Python interface for the RCSB PDB search API |
| [protestar](https://github.com/refresh-bio/protestar) | compress collections structures |
| [Merizo-search](https://github.com/psipred/merizo_search) | domain segmentation |
| [progres](https://github.com/greener-group/progres) | structure searching by structural embeddings ([see also the webserver](https://progres.mrc-lmb.cam.ac.uk/))|
| [freesasa](https://github.com/mittinatten/freesasa) | for calculating Solvent Accessible Surface Areas |
| [openstructure](https://openstructure.org/docs/2.8/) | protein structure, complexes and docking comparison |
| [opendock](https://github.com/guyuehuo/opendock)| protein-Ligand Docking and Modeling |
| [pyScoMotif](https://github.com/3BioCompBio/pyScoMotif) | protein motif search |
| [pyRMSD](https://github.com/salilab/pyRMSD) | RMSD calculations of large sets of structures |
| [Muscle-3D](https://github.com/rcedgar/muscle) | multiple protein structure alignment |
| [pdb-redo](https://pdb-redo.eu/)| automated procedure to refine, rebuild and validate your model |
| [proteinshake](https://github.com/BorgwardtLab/proteinshake) | preprocessed and cleaned protein 3D structure datasets |
| [profet](https://github.com/alan-turing-institute/profet) | Retrieves the cif or pdb files from either thePDB (using pypdb) or Alphafold using the Uniprot ID |
| [mini3di](https://github.com/althonos/mini3di) | NumPy port of the foldseek code for encoding protein structures to 3di |
| [GEMMI](https://github.com/project-gemmi/gemmi) | macromolecular crystallography library and utilities |
| [fpocket](https://github.com/Discngine/fpocket) | protein pocket detection based on Voronoi tessellation |
| [PyDSSP](https://github.com/ShintaroMinami/PyDSSP) | implementation of DSSP (i.e. secondary structure annotation) algorithm for PyTorch |
| [dssp 4.5](https://github.com/PDB-REDO/dssp) | assign secondary structure using the eight-letter code (see also [the webserver](https://pdb-redo.eu/dssp)) | 
| [flashfold](https://github.com/chayan7/flashfold) | command-line tool for faster protein structure prediction |
| [ActSeek](https://github.com/vttresearch/ActSeek) | enzyme mining in the Alphafold database based on the position of few amino acids |
| [p2rank](https://github.com/rdk/p2rank) | Protein-ligand binding site prediction from protein structure |
| [PLACER](https://github.com/baker-laboratory/PLACER) |  local prediction of protein-ligand conformational ensembles |
| [PDBCleanV2](https://github.com/fatipardo/PDBCleanV2) | create a curated ensemble of molecular structures |
| [ProLIF](https://github.com/chemosim-lab/ProLIF) | Interaction Fingerprints for protein-ligand complexes and more |
| [plip](https://github.com/pharmai/plip) | Analyze and visualize non-covalent protein-ligand and protein-protein interactions |
| [PyRosetta](https://github.com/RosettaCommons/PyRosetta.notebooks) | Rosetta suite for protein desing ported to python (See also these instructions for an [easy installation in Colab](https://x.com/miangoar/status/1835176497063030798) as well as the [documentation](https://graylab.jhu.edu/PyRosetta.documentation/index.html)) |
| [af_analysis](https://github.com/samuelmurail/af_analysis) | Analysis of alphafold and colabfold results |
| [Uniprot-PDB-mapper](https://github.com/iriziotis/Uniprot-PDB-mapper) | mapping of Uniprot sequences to PDB (see also this option to [map IDs between RefSeq and Uniprot](https://ncbiinsights.ncbi.nlm.nih.gov/2023/11/08/compare-ncbi-refseq-and-uniprot-datasets/) using the file `gene_refseq_uniprotkb_collab.gz`)|
| [peppr](https://github.com/aivant/peppr) | a package for evaluation of predicted poses like RMSD, TM-score, lDDT, lDDT-PLI, fnat, iRMSD, LRMSD, DockQ  |
| [InteracTor](https://github.com/Dias-Lab/InteracTor) |  structure analysis and conversion, allowing the extraction of interactions such as hydrogen bonds, van der Waals interactions, hydrophobic contacts, and surface tension |
| [unicore](https://github.com/steineggerlab/unicore) | core gene phylogeny with Foldseek and ProstT5 (i.e. 3Di alphabet) |
| [PAthreader and FoldPAthreader](https://github.com/iobio-zjut/PAthreader/tree/main/PAthreader_main) | PAthreader improve AF2 template selection by looking remote homologous in PDB/AFDB and FoldPAthreader predict the folding pathway  (see also [the webserver](http://zhanglab-bioinf.com/PAthreader/))  |
| [SoftAlign](https://github.com/jtrinquier/SoftAlign) | compare 3D protein structures |
| [SCHEMA-RASPP](https://github.com/mattasmith/SCHEMA-RASPP) | structure-guided protein recombination (download and check the file [`schema-tools-doc.html`](https://github.com/mattasmith/SCHEMA-RASPP/blob/master/schema-tools-doc.html) for documentation)|
| [ProteinTools ](https://proteintools.uni-bayreuth.de/) | Analyze Hydrophobic Clusters, Hydrogen Bond Networks, Contact maps, Salt Bridges and Charge Segregation |
| [ProtLego](https://hoecker-lab.github.io/protlego/) | constructing protein chimeras and its structural analysis |
| [reseek](https://github.com/rcedgar/reseek) | structure alignment and search algorithm |
| [tmtools](https://github.com/jvkersch/tmtools) | Python bindings for the TM-align algorithm and code for protein structure comparison |
| []() |  |

# Phylogeny
| Name | Description | 
|-----------|-----------|
| [automlst2](https://automlst2.ziemertlab.com/index) | automatic generation of species phylogeny with reference organisms |
| [unicore](https://github.com/steineggerlab/unicore) | Universal and efficient core gene phylogeny with Foldseek and ProstT5  |
| [ugene](https://ugene.net/) | bioinformatic suite with graphic user interface |
| []() |  |


# structure prediction
| Name | Description | 
|-----------|-----------|
| [PAE Viewer](https://gitlab.gwdg.de/general-microbiology/pae-viewer) |  view the PAE (predicted aligned error) of multimers, and integrates visualization of crosslink data (use the [webserver](https://subtiwiki.uni-goettingen.de/v4/paeViewerDemo)) |
| [PyMOLfold](https://github.com/colbyford/PyMOLfold) | Plugin for folding sequences directly in PyMOL |
| [AFsample2](https://github.com/iamysk/AFsample2/) |  induce significant conformational diversity for a given protein |
| [alphafold3 tools](https://github.com/cddlab/alphafold3_tools) | Toolkit for input generation and output analysis |
| [alphafold3](https://github.com/google-deepmind/alphafold3) | dude ... you can also use the [AF3 weberser](https://alphafoldserver.com/welcome).Se also the [High Throughput Solution to predict up-to 10s thousands of structures](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/cloudnext25/examples/science/af3-slurm/README.md) using the [google cloud services](https://blog.google/products/google-cloud/scientific-research-tools-ai/?utm_source=x&utm_medium=social&utm_campaign=&utm_content=#aimodels) |
| [af3cli](https://github.com/SLx64/af3cli) | generating AlphaFold3 input files |
| [AFDB Structure Extractor](https://project.iith.ac.in/sharmaglab/alphafoldextractor/index.html) | download structures using AF IDs, Uniprot IDs, Locus tags, RefSeq Protein IDs and NCBI TaxIDs |
| [RareFold](https://github.com/patrickbryant1/RareFold) | Structure prediction and design of proteins with 29 noncanonical amino acids |
| [AFusion](https://github.com/Hanziwww/AlphaFold3-GUI) | GUI & Toolkit with Visualization to AF3 |
| [BoltzDesign1](https://github.com/yehlincho/BoltzDesign1) |  designing protein-protein interactions and biomolecular complexes |
| [Hackable AlphaFold 3](https://github.com/chaitjo/alphafold3/) | a lightweight, hackable way to run AF3 to experiment without the massive MSA databases or Docker overhead |
| [ABCFold](https://github.com/rigdenlab/ABCFold) | Scripts to run AlphaFold3, Boltz-1 and Chai-1 with MMseqs2 MSAs and custom templates  |
| []() |  |

# Sequence generation
| Name | Description | 
|-----------|-----------|
| [ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) | conditional language model for the generation of artificial functional enzymes |
| [REXzyme_aa](https://huggingface.co/AI4PD/REXzyme_aa) | generate sequences that are predicted to perform their intended reactions |
| [ProGen2-finetuning](https://github.com/hugohrban/ProGen2-finetuning) | Finetuning ProGen2 for generation of sequences from selected families |
| [Pinal](https://github.com/westlake-repl/Denovo-Pinal) | Text-guided protein design |
| [Evolla](https://github.com/westlake-repl/Evolla) | chat about the function of a protein using its sequence and structure  (i.e. ChatGPT for proteins; see also the [webserver using the 10B param. version of the model](http://www.chat-protein.com/)) |
| []() |  |


# Structure generation
| Name | Description | 
|-----------|-----------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) |  structure generation, with or without conditional information |
| [chroma](https://github.com/generatebio/chroma) | programmable protein design |
| [protein_generator](https://github.com/RosettaCommons/protein_generator) | Joint sequence and structure generation with RoseTTAFold sequence space diffusion |
| [RFdiffusion_all_atom](https://github.com/baker-laboratory/rf_diffusion_all_atom) | RFdiffusion with all atom modeling |
| [salad](https://github.com/mjendrusch/salad) | structure generation with sparse all-atom denoising models |
| []() |  |

# Inverse folding
| Name | Description | 
|-----------|-----------|
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Fixed backbone design ([see webserver](https://huggingface.co/spaces/simonduerr/ProteinMPNN))|
| [ProteinMPNN in JAX](https://github.com/sokrypton/ColabDesign/tree/main/mpnn) | Fast implementation of ProteinMPNN |
| [ligandMPNN](https://github.com/dauparas/LigandMPNN) | Fixed backbone design sensible to ligands ([see colab notebook](https://github.com/ullahsamee/ligandMPNN_Colab))|
| [SolubleMPNN](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights) | Retrained version of ProteinMPNN by excluding transmembrane structures (see [Goverde et al. 2024](https://www.nature.com/articles/s41586-024-07601-y#Sec7) for more details)|
| [fampnn](https://github.com/richardshuai/fampnn) | full-atom version of ProteinMPNN |
| [LASErMPNN ](https://github.com/polizzilab/LASErMPNN) | All-Atom (Including Hydrogen!) Ligand-Conditioned Protein Sequence Design & Sidechain Packing Model |
| []() |  |


# Sequence-structure co-generation
| Name | Description | 
|-----------|-----------|
| []() |  |


# molecular dynamics
| Name | Description | 
|-----------|-----------|
| [making-it-rain](https://github.com/pablo-arantes/making-it-rain) | Cloud-based molecular simulations for everyone |
| [bioemu](https://github.com/microsoft/bioemu) |  emulation of protein equilibrium ensembles  (see also this [this notebook](https://colab.research.google.com/github/pablo-arantes/making-it-rain/blob/main/BioEmu_HPACKER.ipynb) from ["Make it rain"](https://github.com/pablo-arantes/making-it-rain) that combines bioemu + [H-Packer](https://github.com/gvisani/hpacker) for side-chain reconstruction)| 
| [orb](https://github.com/orbital-materials/orb-models) | forcefield models from Orbital Materials |
| [logMD](https://github.com/log-md/logmd) | visualize MD trajectories in colab |
| []() |  |


# representation learning
| Name | Description | 
|-----------|-----------|
| [FAESM](https://github.com/pengzhangzhi/faesm) | An Efficient Pytorch Implementation of ESM and ProGen PLM that can save up to 60% of memory usage and 70% of inference time | 
| [ESM-Efficient](https://github.com/uci-cbcl/esm-efficient) | Efficient implementatin of ESM family | 
| [ProtLearn](https://github.com/tadorfer/protlearn) | extracting protein sequence features |
| [Pfeature](https://github.com/raghavagps/Pfeature) | computing features of peptides and proteins |
| [bio_embeddings](https://github.com/sacdallago/bio_embeddings) | compute protein embeddings from sequences |
| [Graph-Part](https://github.com/graph-part/graph-part) | data partitioning method for ML |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow) | data processing pipeline fo ML |
| [docktgrid](https://github.com/gmmsb-lncc/docktgrid) | Create customized voxel representations of protein-ligand complexes |
| [Prop3D](https://github.com/bouralab/Prop3D) | toolkit for protein structure dataset creation and processing  |
| [SaProt](https://github.com/westlake-repl/SaProt) | Protein Language Model with Structural Alphabet (AA+3Di) (See also [ColabSaprot for structure-aware PLM](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub_v2.ipynb) and [ColabSeprot for sequence-only PLM](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/ColabSeprot.ipynb?hl=en)) |
| [ProstT5](https://github.com/mheinzinger/ProstT5) | Bilingual Language Model for Protein Sequence and Structure (see the [Foldseek adaptation](https://github.com/steineggerlab/foldseek?tab=readme-ov-file#structure-search-from-fasta-input)) |
| [Graphein](https://github.com/a-r-j/graphein) | geometric representations of biomolecules and interaction networks |
| [PyUUL](https://pyuul.readthedocs.io/index.html) | encode structures into differentiable data structures |
| [colav](https://github.com/Hekstra-Lab/colav) | feature extraction methods like dihedral angles, CA pairwise distances, and strain analysis |
| [ProTrek](https://github.com/westlake-repl/ProTrek) <br> [webserver](http://search-protrek.com/)| multimodal (sequence-structure-function) protein representations and annotations |
| [masif](https://github.com/LPDI-EPFL/masif) | molecular surface interaction fingerprints |
| [peptidy](https://github.com/molML/peptidy) | vectorize proteins for machine learning applications |
| [pypropel](https://github.com/2003100127/pypropel) | sequence and structural data preprocessing, feature generation, and post-processing for model performance evaluation and visualisation, |
| []() |  |

# protein engineering
| Name | Description | 
|-----------|-----------|
| [biotite](https://www.biotite-python.org/latest/) | sequence and structure manipulation and analysis |
| [protkit](https://github.com/silicogenesis/protkit) | Unified Approach to Protein Engineering |
| [EvoProtGrad](https://github.com/NREL/EvoProtGrad) | directed evolution with MCMC and protein language models |
| [ConservFold](https://www.rodrigueslab.com/resources) | map amino acid conservation intro structures with the AF2 pipeline |
| [consurf](https://consurf.tau.ac.il/consurf_index.php) |identification of functionally important regions in proteins |
| [AlphaPulldown](https://www.embl-hamburg.de/AlphaPulldown/) | Complex moedeling with AF-Multimer |
| [ColabDock](https://github.com/JeffSHF/) | protein-protein docking |
| [ColabDesign](https://github.com/sokrypton/ColabDesign) | protein design pipelines |
| [LazyAF](https://github.com/ThomasCMcLean/LazyAF) | protein-protein interaction with AF2|
| [CombFold](https://github.com/dina-lab3D/CombFold) | structure predictions of large complexes |
| [Cfold](https://github.com/patrickbryant1/Cfold) | structure prediction of alternative protein conformations |
| [Replacement Scan](https://colab.research.google.com/github/sokrypton/ColabBio/blob/main/notebooks/replacement_scan.ipynb) | find how many amino acid replacements your protein can tolerate [see tw](https://x.com/sokrypton/status/1812769477228200086) |
| [protein_scoring](https://github.com/seanrjohnson/protein_scoring) | generating and scoring novel enzyme sequences  |
| [AF_unmasked](https://github.com/clami66/AF_unmasked) | structure prediction for huge protein complexes (~27 chains and ~8400aa) |
| [AncFlow](https://github.com/rrouz/AncFlow) | pipeline for the ancestral sequence reconstruction of clustered phylogenetic subtrees |
| [TRILL](https://github.com/martinez-zacharya/TRILL) | Sandbox for Deep-Learning based Computational Protein Design |
| [AF2BIND](https://github.com/sokrypton/af2bind) | Predicting ligand-binding sites based on AF2 |
| [PyPEF](https://github.com/Protein-Engineering-Framework/PyPEF) | sequence-based machine learning-assisted protein engineering |
| [DeepProtein](https://github.com/jiaqingxie/DeepProtein) | protein Property Prediction |
| [FlexMol](https://github.com/Steven51516/FlexMol) | construction and evaluation of diverse model architectures  |
| [Pinal](https://github.com/westlake-repl/Denovo-Pinal) | text-guided protein design |
| [ByProt (LM-Design)](https://github.com/bytedprotein/ByProt) | reprogramming pretrained protein LMs as generative models |
| [scikit-bio](https://github.com/scikit-bio/scikit-bio) | data structures, algorithms and educational resources for bioinformatics |
| []() |  |






# docking (ligands and proteins)
| Name | Description | 
|-----------|-----------|
| [HiQBind](https://github.com/THGLab/HiQBind) | Workflow to clean up and fix structural problems in protein-ligand binding datasets |
| [InterfaceAnalyzerMover](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/analysis/InterfaceAnalyzerMover) | Calculate binding energies, buried interface surface areas, packing statistics, and other useful interface metrics for the evaluation of protein interfaces |
| [PandaDock](https://github.com/pritampanda15/PandaDock) | Physics-Based Molecular Docking |
| [ligysis](https://www.compbio.dundee.ac.uk/ligysis/) | analysis of biologically meaningful ligand binding sites |
| []() |  |

# molecules
| Name | Description | 
|-----------|-----------| 
| [PDBe CCDUtils](https://pdbeurope.github.io/ccdutils/index.html)  | tools to deal with PDB chemical components and visualization ([see also](https://github.com/PDBeurope/pdbe-notebooks/tree/main/pdbe_ligands_tutorials))|
| [PDBe Arpeggio](https://github.com/PDBeurope/arpeggio) |  calculation of interatomic interactions in molecular structures|
| [PDBe RelLig](https://github.com/PDBeurope/rellig) | classifies ligands based on their functional role| 
| [MolPipeline](https://github.com/basf/MolPipeline) | processing molecules with RDKit in scikit-learn |
| [roshambo](https://github.com/molecularinformatics/roshambo) | molecular shape comparison |
| [FlexMol](https://github.com/Steven51516/FlexMol) | construction and evaluation of diverse model architectures |
| [molli](https://github.com/SEDenmarkLab/molli) | general purpose molecule library generation and handling  |
| [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) | A collection of useful RDKit and sci-kit learn functions |
| []() |  |

# machine learning
| Name | Description | 
|-----------|-----------| 
| [Colab forms](https://colab.research.google.com/notebooks/forms.ipynb) | how to convert a colab notebook to a user interface |
| [cuml](https://github.com/rapidsai/cuml) | GPU-based implementations of common machine learning algorithms ([more info for umap optimization](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/) and [cuml.accel](https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/) to boost scikit-learn and other libs in colab)|
| [LazyPredict](https://github.com/shankarpandala/lazypredict) | build a lot of basic models without much code |
| [TorchDR](https://github.com/TorchDR/TorchDR) | PyTorch Dimensionality Reduction |
| [Kerasify](https://github.com/moof2k/kerasify) | running trained Keras models from a C++ application |
| [pca](https://erdogant.github.io/pca/pages/html/index.html) | perform PCA and create insightful plots |
| [openTSNE](https://opentsne.readthedocs.io/en/stable/) | faster implementation of t-SNE that includes other optimizations |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | model for tabular data that outperforms traditional methods while being dramatically faster |
| [setfit](https://github.com/huggingface/setfit) | Efficient few-shot learning with Sentence Transformers |
| [skrub](https://github.com/skrub-data/skrub) | preprocessing and feature engineering for tabular machine learning |
| [cupy](https://github.com/cupy/cupy) |NumPy & SciPy for GPU|
| [Best-of Machine Learning](https://github.com/ml-tooling/best-of-ml-python) | list of awesome machine learning Python libraries |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | 100+ PyTorch metrics implementations |
| [DADApy](https://github.com/sissa-data-science/DADApy) | characterization of manifolds in high-dimensional spaces |
| [PySR](https://github.com/MilesCranmer/PySR) | High-Performance Symbolic Regression |
| [BERTopic](https://github.com/MaartenGr/BERTopic) | create clusters for easily interpretable topics |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT) | Minimal keyword extraction with BERT |
| [PolyFuzz](https://github.com/MaartenGr/PolyFuzz) | Fuzzy string matching, grouping, and evaluation. |
| [hummingbird](https://github.com/microsoft/hummingbird) | compiles trained ML models into tensor computation for faster inference |
| [skorch](https://github.com/skorch-dev/skorch) | train PyTorch models in a way similar to Scikit-learn (eg. No need to manually write a training loop, just using fit(), predict(), score()) |
| [Faiss](https://github.com/facebookresearch/faiss) | efficient similarity search and clustering of dense vectors |
| []() |  |

# statistics
| Name | Description | 
|-----------|-----------| 
| [scikit-posthocs](https://scikit-posthocs.readthedocs.io/en/latest/) |  post hoc tests for pairwise multiple comparisons |
| [statannotations](https://github.com/trevismd/statannotations) | add statistical significance annotations on seaborn plots |
| [ggstatsplot](https://github.com/IndrajeetPatil/ggstatsplot) | creating graphics with details from statistical tests included in the information-rich plots themselves |
| [ggbetweenstats](https://indrajeetpatil.github.io/ggstatsplot/articles/web_only/ggbetweenstats.html) | making publication-ready plots with relevant statistical details |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration |
| [pingouin](https://pingouin-stats.org/build/html/index.html#) | Statistical package |
| [performance](https://github.com/easystats/performance) | computing indices of regression model quality and goodness of fit  |
| []() |  |


# datavis for bio
| Name | Description | 
|-----------|-----------| 
| [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) | Toolbox for molecular animations in Blender |
| [CellScape](https://github.com/jordisr/cellscape) | Vector graphics cartoons from protein structure |
| [ChimeraX apps](https://cxtoolshed.rbvi.ucsf.edu/apps/all) | chimeraX  extensions |
| [chimerax_viridis](https://github.com/smsaladi/chimerax_viridis) | Colorblind-friendly, perceptually uniform palettes for chimerax (see [this tips](https://bsky.app/profile/jameslingford.bsky.social/post/3lmw5h3xxec2n) tore-create the RFDifussion style in ChimeraX or the [original pyMOL implementation](https://bsky.app/profile/spellock.bsky.social/post/3lnsup5m5ks22))|
| [SSDraw](https://github.com/ncbi/SSDraw) | generates secondary structure diagrams from 3D protein structures |
| [bioalphabet](https://github.com/davidhoksza/bioalphabet/) | convertor of texts to bio-domain alphabetss |
| [ChatMol](https://github.com/ChatMol/ChatMol) | a PyMOL ChatGPT Plugin that allow to interact with PyMOL using natural language  |
| [plot_phylo](https://github.com/KatyBrown/plot_phylo) | plot a phylogenetic tree on an existing matplotlib axis |
| [prettymol](https://github.com/zachcp/prettymol) | automatic protein structure plots with MolecularNodes  |
| [VMD-2](https://www.ks.uiuc.edu/Research/vmd/vmd2intro/index.html) | Visual Molecular Dynamics |
| [gromacs_copilot](https://github.com/ChatMol/gromacs_copilot) | AI-powered assistant for Molecular Dynamics  simulations |
| [NIH bioart](https://bioart.niaid.nih.gov/) | 2,000+ science and medical icons |
| [bioicons](https://bioicons.com/) | icons for science illustrations in biology and chemistry |
| [moldraw](https://moldraw.com/) | draw molecules |
| [Mol* at RCSB/PDB](https://onlinelibrary.wiley.com/doi/10.1002/pro.70093) |  web-based, 3D visualization software suite for examination and analyses of biostructures |
| [PoseEdit](https://proteins.plus/) | interactive 2D ligand interaction diagrams (see this [tutorial](https://www.youtube.com/watch?v=8W1TvSvatSA&ab_channel=BioinformaticsInsights)) |
| [FlatProt](https://github.com/t03i/FlatProt) | 2D protein visualization aimed at improving the comparability of structures  |
| [quarto-molstar](https://github.com/jmbuhr/quarto-molstar) | embed proteins and trajectories with Mol* |
| [alphabridge](https://alpha-bridge.eu/) | summarise predicted interfaces and intermolecular interactions |
| [weblogo](https://weblogo.threeplusone.com/) |  generation of sequence logos |
| [interprot](https://interprot.com/#/) | inspect relevant features derived from protein language models in a particular protein |
| []() |  |


# datavis
| Name | Description | 
|-----------|-----------| 
| [datamapplot](https://github.com/TutteInstitute/datamapplot) | creating beautiful, interactive and massive scatterplots |
| [pypalettes](https://github.com/JosephBARBIERDARNAL/pypalettes) | +2500 color maps  |
| [distinctipy ](https://github.com/alan-turing-institute/distinctipy) |  generating visually distinct colours |
| [Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network) | set of tools (like [NN-SVG](https://alexlenail.me/NN-SVG/LeNet.html)) to plot neural nets |
| [tidyplots](https://jbengler.github.io/tidyplots/index.html) | creation of publication-ready plots for scientific papers |
| [pyCirclize](https://github.com/moshi4/pyCirclize) | Circular visualization in Python (Circos Plot, Chord Diagram, Radar Chart)  |
| [pycircular](https://github.com/albahnsen/pycircular) | circular data analysis |
| [great-table](https://github.com/posit-dev/great-tables) |  display tables |
| [plottable](https://github.com/znstrider/plottable) | plotting beautifully customized, presentation ready tables |
| [d3blocks](https://github.com/d3blocks/d3blocks) | create stand-alone and interactive d3 charts |
| [How to Vectorize Plots from R/Python in PowerPoint](https://nalinan.medium.com/how-to-vectorize-plots-from-r-in-powerpoint-bad7c238e86a) | import a vectorized image into ProwerPoint for easy manipulation ([see also this tutorial](https://www.youtube.com/watch?v=hoHkc7N6FZA&ab_channel=GenomicsBootCamp)) |
| []() |  |


# webservers
| Web | Description | 
|-----------|-----------| 
| [ProteInfer](https://google-research.github.io/proteinfer/) | predicting functional properties from sequences |
| [GoPredSim](https://embed.protein.properties/) | Predict protein properties from embeddings |
| [DeepFRI](https://beta.deepfri.flatironinstitute.org/) | structure-based protein function prediction and functional residue identification |
| [protein structure relaxation](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/relax_amber.ipynb) | Relax your structure using OpenMM/Amber |
| [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) | calculate how much vRAM is needed to train and perform inference on a model hosted on Hugging Face (see also [gpu_poor calculator](https://github.com/RahulSChand/gpu_poor) or [this ecuation](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)) |
| [alphafind](https://alphafind.fi.muni.cz/search) | structure-based search |
| [DiffDock-Web](https://huggingface.co/spaces/reginabarzilaygroup/DiffDock-Web) | molecular docking with ligands |
| [ESMFold](https://esmatlas.com/resources?action=fold) | structure prediction with ESMFold |
| [Foldseek clusters](https://cluster.foldseek.com/) | search for sctructural clusters in AFDB |
| [damietta](https://damietta.de/) | protein design toolkit |
| [easifa](http://easifa.iddd.group/) | active and binding site annotations for enzymes |
| [InterProt](https://interprot.com/#/) |  interpretability of features derived from protein language models using sparse autoencoders |
| [MPI Bioinformatics Toolkit!](https://toolkit.tuebingen.mpg.de/) | multiple bioinformatics tools |
| [moleculatio](https://moleculatio.yamlab.app) | chemoinformatics, quantum chemistry and molecular dynamics simulations or small molecules |
| [AI in Biology Demos]( https://huggingface.co/collections/hf4h/ai-in-biology-demos-65007d936a230e55a66cd31e) | applications of AI in biology and biochemistry |
| [ProteinsPlus](https://proteins.plus/) | structure mining and modeling, focussing on protein-ligand interactions |
| [gtalign](https://bioinformatics.lt/comer/gtalign/) | High-performance search and alignment for protein structures |
| []() |  |


# chatbots and agents
| Name | 
|-----------|
| [ChatGPT](https://chat.openai.com/) |
| [Gemini](https://gemini.google.com/) |
| [claude](https://claude.ai/) |
| [Bing](https://www.bing.com/?cc=es) |
| [HuggingChat](https://huggingface.co/chat/) |
| [huggingface spaces](https://huggingface.co/spaces) | 
| [biologpt](https://biologpt.com/) |
| [consensus](https://consensus.app/) |  
| [typeset](https://typeset.io/) |  
| [mistral-chat](https://mistral.ai/news/mistral-chat/) |  
| [aistudio by Google](https://aistudio.google.com/) |
| [AI Python Libraries](https://www.aipythonlibraries.com/libraries/) | 
| [paperfinder](https://paperfinder.allen.ai/chat) | 
| [AI Scientist agents by futurehouse](https://platform.futurehouse.org/) |
| [OpenAI Deep Research Guide (by DAIR.AI)](https://docs.google.com/document/d/1vLaEMu5jirQT5RK0cW8RUXNFQyszMQ-xrjxUZF2wOg4/edit?tab=t.0#heading=h.2y9eo2rdwxv2) |  
| []() |  

