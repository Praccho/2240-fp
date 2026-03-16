# Neural world models meet 3D latent diffusion on consumer GPUs

A streaming multiplayer world model built on 3D spatially-indexed latent diffusion is architecturally feasible today by combining several converging research threads: MAR's parallel per-token diffusion, SLAT's sparse voxel latents, few-step distillation techniques, and the RTX 5080's **900 TFLOPS of FP4 tensor throughput** within a **128 KB per-SM shared memory** hierarchy. The critical constraint is fitting the per-token diffusion MLP entirely in SMEM—achievable at FP4 precision for MLPs under ~64 K parameters. This report surveys the ten foundational research areas underpinning this proposal, with architectural details, key equations, and concrete design implications.

---

## 1. MAR decouples autoregressive structure from sequential bottlenecks

**Paper:** "Autoregressive Image Generation without Vector Quantization" — Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He (MIT/DeepMind, 2024). NeurIPS 2024 Spotlight. arXiv:2406.11838.

MAR replaces discrete token prediction with **per-token continuous diffusion**, eliminating vector quantization entirely. The architecture splits into a bidirectional transformer (MAE-style encoder-decoder, 16+16 blocks for MAR-L at **479M parameters**) and a small denoising MLP per token. The transformer produces a conditioning vector **z ∈ ℝ^D** for each masked position; each MLP then independently runs a full diffusion process conditioned on z.

The diffusion loss per token is: **L(z, x) = E_{ε,t}[‖ε − ε_θ(x_t | t, z)‖²]**, where x_t = √ᾱ_t·x + √(1−ᾱ_t)·ε follows a cosine noise schedule. The MLP uses **AdaLN** (adaptive layer normalization) to inject the sum of z and a timestep embedding. Default MLP size is **3 residual blocks at width 1024 (~21M params)**, adding only ~5% overhead to the transformer. Critically, inference runs **100 diffusion steps per token but only 64 autoregressive steps** total (vs. 256 sequential steps for standard AR), because MAR predicts sets of tokens simultaneously using a cosine unmasking schedule from MaskGIT.

The per-token MLP runs independently across all predicted positions—**perfectly parallel on GPU**. At inference, the MLP adds only ~10% to total time regardless of width, because it is compute-bound rather than memory-bound. For the proposed system, this architecture is ideal: the transformer handles global context (which tokens exist, what their spatial relationships are), while tiny MLPs handle local continuous generation. Each MLP could fit in shared memory at FP4 precision.

Follow-ups include **NOVA** (ICLR 2025), extending MAR to video with spatial set-by-set + temporal frame-by-frame AR, and **MarDini** (Meta, ICLR 2025), which combines MAR-based temporal planning with DiT spatial generation.

---

## 2. JiT proves prediction targets should be low-dimensional

**Paper:** "Back to Basics: Let Denoising Generative Models Denoise" — Tianhong Li, Kaiming He (MIT, 2025). arXiv:2511.13720.

JiT ("Just image Transformers") is the philosophical complement to MAR. Its core finding: **x-prediction** (predicting the clean image directly) succeeds where ε-prediction and v-prediction catastrophically fail in high-dimensional patch spaces. For 16×16×3 = 768-dimensional patches, ε-prediction produces FID > 300 (complete failure), while x-prediction achieves FID ~7–10. The explanation is the **manifold hypothesis**: clean images occupy a low-dimensional manifold within the high-dimensional pixel space, making x the only prediction target that doesn't require the network to model full-rank noise.

The architectural consequence is a **bottleneck linear embedding**: two linear layers with intermediate dimension d' ≪ D (e.g., d' = 128 for 768-dim patches), with no activation between them. This bottleneck **improves** FID from 9.40 (no bottleneck) to **7.35** (128-dim bottleneck), confirming that forcing the network onto the data manifold helps rather than hurts. The full JiT-G/16 achieves **FID 1.82** on ImageNet 256×256 without any VAE, tokenizer, or auxiliary loss.

For the proposed system, JiT's insight applies directly to per-token diffusion MLPs operating on 3D latent patches. If latent tokens represent local 3D regions, the "clean" token lies on a low-dimensional manifold of valid local geometry/appearance. Using x-prediction with a bottleneck embedding in the diffusion MLP reduces the effective dimensionality the MLP must handle, enabling smaller MLPs that fit in SMEM. The related **k-Diff** framework (Jin & Wang, 2026, arXiv:2601.21419) provides a theoretical basis for choosing the optimal prediction parameterization given known intrinsic vs. ambient dimensionality.

---

## 3. VGGT reconstructs 3D geometry in a single forward pass

**Paper:** "VGGT: Visual Geometry Grounded Transformer" — Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny (Oxford/Meta, 2025). **CVPR 2025 Best Paper**. arXiv:2503.11651.

VGGT is a **1.2B-parameter** transformer that takes 1–N images and produces camera poses, depth maps, point maps, and tracking features in **under one second** for 20 frames. Its architecture uses **alternating attention**: frame-wise self-attention (tokens attend within their own image) alternates with global self-attention (tokens attend across all images) across **24 transformer blocks**. Input tokens come from a frozen DINOv2 encoder (14×14 patches), with special camera tokens appended per frame.

The key design insight is **over-complete prediction**: the model predicts depth maps, point maps (3D coordinates in world frame), and camera parameters independently, even though these are mathematically redundant. Multi-task training with loss L = L_camera + L_depth + L_pmap + 0.05·L_track improves every individual prediction through complementary gradient signals. Dense outputs use **DPT heads** extracting multi-scale features from blocks 4, 11, 17, and 23.

For the proposed system, VGGT provides the **encoder pathway**: given multiple player viewpoints, VGGT-style alternating attention can reconstruct a shared 3D representation. The alternating attention pattern—local per-view normalization followed by global cross-view reasoning—is directly applicable to a multiplayer world model where each player's frame tokens first attend within-view, then attend across all player views to maintain consistency. The model's ability to process variable numbers of views and its sub-second inference make it architecturally compatible with real-time constraints.

---

## 4. RTX 5080 Blackwell: 128 KB SMEM is the binding constraint

The RTX 5080 (GB203, Compute Capability 12.0) provides **84 SMs**, each containing **128 CUDA cores**, **4 fifth-generation tensor cores**, **128 KB configurable L1/SMEM**, and **256 KB register file**. The aggregate numbers matter: **10.5 MB total SMEM**, **21.5 MB total register file**, **64 MB L2 cache**, and **16 GB GDDR7 at 960 GB/s**.

**The FP4 tensor core throughput is transformative**: **900.4 TFLOPS** (1,801 with sparsity) on the RTX 5080—4× the FP16 rate. At FP4 precision, a 21M-parameter diffusion MLP compresses to roughly **10.5 MB**, fitting entirely in L2 cache. A stripped-down MLP of ~50K parameters (sufficient for per-token denoising on low-dimensional latents) compresses to **~25 KB at FP4**, fitting in a single SM's shared memory with room for activations and input/output buffers.

The critical architectural constraint: **consumer Blackwell (CC 12.0) lacks many datacenter features**. TMA (Tensor Memory Accelerator), TMEM (Tensor Memory), tcgen05 instructions, and Thread Block Clusters are all primarily CC 10.0 (datacenter). Consumer GPUs retain cp.async for asynchronous global→SMEM copies, standard wmma tensor operations, FP4/FP6 tensor core support, and L2 persistence control via cudaAccessPolicyWindow. The L1/SMEM bandwidth is **128 bytes/cycle per SM** (~307 GB/s per SM, ~25.8 TB/s aggregate).

A practical SMEM-resident inference strategy for the per-token diffusion MLP:

- **Model weights**: 50K params × 0.5 bytes (FP4) = **25 KB** in SMEM
- **Activations**: Two layers of width 256 at FP16 = **1 KB** per token
- **I/O buffers**: Input/output latent vectors = **~0.5 KB**
- **Total per SM**: ~27 KB, well within the 99 KB per-block limit
- **Throughput**: At 900 TFLOPS FP4, a 50K-param MLP forward pass takes ~0.1 μs; 100 diffusion steps = **~10 μs per token per SM**

The memory hierarchy has **not changed** from Ada Lovelace at the per-SM level (128 KB L1/SMEM, 256 KB registers). The gains come from more SMs, FP4 tensor cores, and GDDR7 bandwidth. Notably, L2 latency has **regressed to ~130 ns** (from Ada's ~107 ns), making SMEM-resident computation even more important for latency-sensitive workloads.

---

## 5. SLAT embeds latent tokens in sparse 3D voxel grids

**Paper:** "Structured 3D Latents for Scalable and Versatile 3D Generation" (TRELLIS) — Jianfeng Xiang et al. (Tsinghua/Microsoft Research, 2024). CVPR 2025 Spotlight. arXiv:2412.01506.

SLAT (Structured LATent) represents 3D content as **z = {(z_i, p_i)}^L_{i=1}**, where z_i are feature vectors and p_i are 3D voxel positions in a sparse grid. Only surface-intersecting voxels carry latents, so **L ≪ N³**. The generation pipeline uses two stages of **rectified flow transformers**: Stage 1 generates which voxels are active (sparse structure); Stage 2 generates latent features for active voxels.

The transformer processes serialized active voxel features with **sinusoidal positional encodings from 3D coordinates** and **3D shifted window attention** (Swin3D-style, 8³ windows). This enforces 3D locality while allowing global communication through window shifts. A Sparse VAE with sparse convolutional up/downsamplers compresses to 64³ grid resolution with 8 latent channels. The decoded SLATs map to multiple output formats (3D Gaussians, radiance fields, meshes) via separate lightweight decoders.

The 2025 follow-up **TRELLIS.2** introduces O-Voxels (omni-voxels) encoding geometry via Flexible Dual Grids and full PBR materials, scaled to a **4B parameter** DiT. **XCube** (NVIDIA, CVPR 2024 Highlight) extends the concept to hierarchical sparse voxels at **1024³** resolution using VDB data structures and cascaded latent diffusion. **GaussianAnything** (ICLR 2025) structures latents as point clouds on the 3D object manifold, enabling geometry-texture disentanglement.

No current work explicitly parameterizes latent tokens in full **SE(3) space** (position + SO(3) orientation). Orientation is encoded implicitly in feature channels. This remains an open frontier for the proposed system—explicit SO(3) parameterization could enable rotation-equivariant local diffusion and better view-dependent effects.

For the proposed multiplayer world model, SLAT's sparse voxel approach provides the 3D latent backbone. A world state represented as sparse 3D-indexed latent tokens can be: (1) decoded into per-player views via projection + rendering, (2) updated via action-conditioned diffusion on the sparse token set, (3) spatially indexed for efficient render-distance-based loading/unloading.

---

## 6. World models achieve 20 FPS through aggressive step reduction

Five systems define the current state of real-time neural world models:

**GameNGen** (Google, ICLR 2025, arXiv:2408.14837) fine-tunes Stable Diffusion v1.4 for next-frame prediction in DOOM, achieving **20 FPS at 320×240 on a single TPU** with only **4 denoising steps**. The critical innovation is **noise augmentation**: adding Gaussian noise to context frames during training teaches the model to correct imperfect autoregressive inputs. Without this, generation diverges within seconds.

**DIAMOND** (NeurIPS 2024 Spotlight, arXiv:2405.12399) demonstrates that the **EDM formulation** (Karras et al.) is dramatically more stable than DDPM for world modeling, even at **3 denoising steps**. Its Atari model is only **4.4M parameters**, proving that small models suffice for constrained domains. The CS:GO variant uses a two-stage pipeline (dynamics at 3 steps + upsampler at 10 steps) achieving ~10 FPS on an RTX 3090.

**Oasis** (Decart, October 2024) uses a **Diffusion Transformer** (DiT) with interleaved temporal attention and **Diffusion Forcing** (independent per-token noise levels). It generates Minecraft gameplay at **20 FPS on an H100** at 360p. Dynamic noising adjusts inference noise schedules to reduce autoregressive error accumulation.

**Genie 2** (DeepMind, December 2024) uses an autoregressive latent diffusion model with a large causal transformer, demonstrating emergent object interactions, NPC behaviors, and ~60 seconds of consistent gameplay. The August 2025 **Genie 3** achieves **24 FPS at 720p** with multi-minute consistency.

The universal pattern: **fewer denoising steps** (3–4) via EDM formulation, noise augmentation, or diffusion forcing; **action conditioning** via embedding concatenation or classifier-free guidance; and **autoregressive frame generation** with per-frame context windows. All systems struggle with long-range state consistency—the fundamental limitation of lacking an explicit world state representation, which the proposed 3D latent approach would address.

---

## 7. Few-step diffusion and SMEM-resident inference are production-ready

**Latent Consistency Models** (Luo et al., 2023, arXiv:2310.04378) distill the consistency function f_θ(z_t, ω, c, t) → z_0 from pretrained diffusion models, enabling **1–4 step generation** by learning direct mappings from any noise level to the clean output. LCM-LoRA (arXiv:2311.05556) makes this composable with any fine-tuned model via **~67 MB** of LoRA parameters. **DMD2** (NeurIPS 2024 Oral) achieves **FID 1.28** on ImageNet 64×64 in a single step—surpassing the teacher model.

**StreamDiffusion** (Kodaira et al., 2023, ICCV 2025) achieves **91 FPS on an RTX 4090** for image-to-image streaming by pipelining denoising: Stream Batch stagers images at different denoising stages, Residual CFG eliminates redundant unconditional evaluations, and a Stochastic Similarity Filter skips frames when input changes are minimal. **StreamDiffusionV2** (MLSys 2026) extends this to video with rolling KV caches and motion-aware noise control, achieving **31–62 FPS** on 1.3B–14B parameter models.

For tiled generation, **MultiDiffusion** (ICML 2023) runs diffusion independently on overlapping patches and fuses via least-squares averaging. **SpotDiffusion** (2024) eliminates the 75% overlap requirement by using time-shifted non-overlapping windows—a **6× speedup**. This is directly applicable to a 3D world model where spatial patches correspond to 3D regions.

**SVDQuant** (arXiv:2411.05007) achieves 4-bit quantization with a low-rank outlier branch, compressing FLUX.1 from 22.2 GB to **6.1 GB** with **3.0× speedup** on RTX 4090. NVIDIA's TensorRT unlocks native **FP4 inference on Blackwell** (May 2025), with Adobe demonstrating 60% latency reduction. Combined with the RTX 5080's FP4 tensor cores, a quantized diffusion MLP could run entirely within the SM memory hierarchy.

---

## 8. Multiplayer world models are nascent but architecturally tractable

**Solaris** (NYU VisionX, February 2025, arXiv:2602.22208) is the **first multiplayer video world model**, generating consistent observations for two Minecraft players simultaneously. It adapts a DiT with **Multiplayer Self-Attention** for cross-player consistency, trained on **12.64 million synchronized multiplayer frames** collected via a custom engine. Checkpointed Self Forcing enables memory-efficient long-horizon generation.

**MultiGen** (March 2026, arXiv:2603.06679) takes the most architecturally relevant approach for the proposed system: it decomposes the world model into three modules—a **persistent external memory** storing map geometry and player poses, an **observation module** generating first-person views conditioned on ray-traced disparity maps, and a **dynamics module** updating poses. The shared memory acts as the ground-truth world state from which per-player views are independently decoded—exactly the paradigm needed for 3D-indexed latent diffusion.

**ShareVerse** (March 2026, arXiv:2603.02697) builds on CogVideoX-5B with **cross-agent attention blocks** that concatenate video features from multiple agents for spatial-temporal information sharing, producing consistent views for two CARLA driving agents.

The critical research gap: **no existing system uses a persistent 3D latent representation as shared world state with per-agent decoders**. MultiGen's external memory is the closest analog, but operates on geometric maps rather than learned latent spaces. Combining SLAT's sparse voxel latents with MultiGen's memory/observation/dynamics decomposition—where the "memory" is a 3D-indexed latent grid and the "observation module" is a per-player projection + decoding pipeline—is the natural synthesis.

---

## 9. 3D-aware generation converges on explicit intermediate representations

**CAT3D** (Google/DeepMind, NeurIPS 2024 Oral, arXiv:2405.10314) generates multiple consistent views by running a multi-view latent diffusion model with **camera ray coordinate conditioning** and cross-view 3D self-attention. Output views feed into 3DGS reconstruction for real-time rendering.

**SV3D** (Stability AI, ECCV 2024 Oral) fine-tunes Stable Video Diffusion for camera-pose-conditioned multi-view synthesis at **576×576**, exploiting video diffusion's temporal consistency as a proxy for 3D consistency. **SV4D** (ICLR 2025) extends this to 4D with joint view attention and frame attention operating on a V×F image grid.

**World-consistent Video Diffusion (WVD)** (Apple, CVPR 2025, arXiv:2412.01821) trains a diffusion transformer on **6D videos** (3 RGB + 3 XYZ channels per pixel), providing explicit 3D geometry supervision per frame. This unifies single-image-to-3D, multi-view stereo, camera control, and depth estimation in a single model.

**Diffusion as Shader (DaS)** (SIGGRAPH 2025, arXiv:2501.03847) conditions video diffusion on **3D tracking videos**—colored 3D point trajectories that provide explicit correspondence across frames via a ControlNet-like architecture fine-tuned in only 3 days on 8 H800s.

**World Labs' Marble** represents the commercial frontier: a generative multimodal world model producing persistent, navigable 3D worlds from various inputs, with **RTFM** (October 2025) enabling real-time generation on a single H100 using spatially-grounded frames as spatial memory.

The clear trend: explicit 3D structure as an intermediate representation (triplanes, point clouds, XYZ coordinate images, 3D tracking) consistently outperforms purely implicit attention-based consistency. For the proposed system, the 3D-indexed latent space serves as this explicit intermediate—views are rendered from it via projection, guaranteeing geometric consistency by construction.

---

## 10. Neural LOD and spatial streaming are reinventing game engine concepts

The neural rendering community has independently converged on game-engine-style spatial management patterns. **NGLOD** (Takikawa et al., CVPR 2021 Oral) stores learned features in a **sparse voxel octree** with discrete LOD levels selected by depth heuristics, achieving real-time rendering via sparse sphere tracing. **Mip-NeRF** (Barron et al., ICCV 2021) provides continuous-scale anti-aliasing through integrated positional encoding of conical frustums.

**Block-NeRF** (Google/Waymo, CVPR 2022) decomposes city-scale scenes into independently trained NeRF blocks from **2.8 million images**, directly analogous to Minecraft-style chunk loading. **CityGaussian** (ECCV 2024) combines divide-and-conquer 3DGS training with **block-wise LOD selection** based on camera distance.

The most complete render-distance analog is **"A LoD of Gaussians"** (Windisch et al., July 2025): it stores the entire Gaussian hierarchy in CPU RAM and **dynamically streams only visible subsets to GPU memory** based on camera position, using Sequential Point Trees for parallel LOD cuts. This achieves **bounded-memory training and rendering** of 60M+ Gaussian scenes on consumer GPUs (≤24 GB VRAM)—almost exactly analogous to game engine asset streaming.

**Instant-NGP's multi-resolution hash encoding** (Müller et al., NVIDIA, 2022) provides the standard spatial indexing mechanism: L grid levels at exponentially increasing resolutions, each with T trainable feature vectors indexed by a spatial hash function, achieving 1000× speedup over NeRF.

For the proposed system, these patterns suggest: sparse 3D-indexed latent tokens organized in a spatial hierarchy (octree or hash grid), with LOD-aware streaming that loads fine-grained tokens near each player and coarse tokens farther away. **PRoGS** (2024) demonstrates priority-based progressive streaming where the most visually important Gaussians load first. Combined with predictive prefetching based on player velocity—an unexplored gap in neural rendering—this could enable constant-memory, variable-detail world model inference.

---

## Synthesis: architecture for a streaming multiplayer 3D latent world model

The converging research threads point toward a specific architecture. The **world state** is a SLAT-style sparse voxel grid of latent tokens, each carrying a feature vector z_i at 3D position p_i. Updates use a **MAR-style** bidirectional transformer for global context, with **per-token diffusion MLPs** (JiT x-prediction, bottleneck embeddings, FP4 quantized) running entirely in RTX 5080 SMEM. Each denoising step of each token's MLP fits in ~27 KB of shared memory at FP4 precision, well within the 99 KB per-block limit.

**Multiplayer** follows MultiGen's decomposition: a shared 3D latent memory is read by per-player observation modules that project and decode local views. Cross-player consistency is guaranteed by the shared latent state rather than requiring cross-attention between player streams. **Spatial streaming** uses octree-hierarchical LOD selection (following "A LoD of Gaussians") to load fine tokens near each player and coarse tokens elsewhere, bounding GPU memory usage to the L2 cache capacity (64 MB) for the latent state and SMEM for the diffusion MLPs.

Real-time inference targets **3–4 diffusion steps** per token update (following DIAMOND's EDM stability results and LCM distillation), with **SpotDiffusion-style shifted windowing** over 3D spatial patches to maintain consistency without redundant overlap computation. StreamDiffusion's pipelining approach—staggering frames at different denoising stages—converts latency into throughput for the per-player rendering pipeline.

The primary open problems are: (1) explicit SE(3) parameterization of latent tokens for rotation-equivariant local diffusion, (2) action conditioning in the sparse 3D latent space rather than in 2D frame space, (3) predictive spatial prefetching based on player trajectories, and (4) scaling multiplayer attention beyond 2 players. Each represents a concrete thesis-level contribution building on the surveyed foundations.