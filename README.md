# MOD3D — Moving Object Detection in 2D and 3D Scenes

## Introduction
MOD3D is an implementation of a traditional computer vision approach to moving object detection, inspired by the work of Michal Irani and P. Anandan in *A Unified Approach to Moving Object Detection in 2D and 3D Scenes* (1998) [1].

Developed by researchers at Ho Chi Minh City University of Technology (HCMUT), Vietnam National University, the project explores geometric constraints, parametric motion estimation, optical flow, and graph-based optimization for detecting independently moving objects in video sequences.

The implementation builds upon the plane-plus-parallax framework, which separates image motion caused by camera movement from motion caused by independently moving objects. Rather than relying on a single global motion model, MOD3D progressively handles more complex scene geometries through layered alignment and multi-view projective rigidity.

This repository is intended for studying and analyzing traditional computer vision and geometric methods for moving object detection.

## Algorithm Overview

### 1. Dominant-Plane Motion Estimation
We begin by considering a 2D scene, where the dominant scene geometry can be sufficiently approximated by a single plane in 3D space.

The induced image displacement field is represented using an 8-parameter quadratic parametric motion model [1]:

$$
\begin{bmatrix} u(x,y) \\ v(x,y) \end{bmatrix} = \begin{bmatrix} a+bx+cy+gx^2+hxy \\ d+ex+fy+gxy+hy^2 \end{bmatrix}
$$

where $u(x,y)$ and $v(x,y)$ represent horizontal and vertical image displacement, respectively, and $(a,b,c,d,e,f,g,h)$ are the motion parameters.

This model works well when the scene is sufficiently distant, predominantly planar, or when the camera undergoes pure rotation, such that the image motion can be adequately explained by a single projective plane.

Using multi-scale Gauss–Newton optimization, we estimate the dominant motion and warp the inspection images $I_j$ and $I_k$ toward the reference image $I$.

We then evaluate the local gradient-normalized normal-flow residual to identify regions that remain significantly misaligned:

$$
M(x,y) = \frac{\sum_{(x',y')\in W(x,y)} |I_w(x',y')-I(x',y')| \|\nabla I(x',y')\|}{\sum_{(x',y')\in W(x,y)} \|\nabla I(x',y')\|^2 + \varepsilon}
$$

where:
- $I_w$ is the warped inspection image.
- $I$ is the reference image.
- $\nabla I$ is the spatial image gradient.
- $W(x,y)$ is a local window centered at $(x,y)$.
- $\varepsilon$ is a small constant for numerical stability.

Regions with high residual values become candidates for independent motion.

### 2. Multi-Planar Alignment and Residual Parallax
A single dominant-plane model is insufficient when the camera translates through a 3D scene containing significant depth variations.

In such cases, static objects at different depths exhibit different image displacements due to motion parallax. A single global model incorrectly classifies these residual displacements as independently moving objects.

To address this limitation, MOD3D progressively decomposes the image motion into additional planar layers and residual planar parallax [1].

The process consists of:
1. **Dominant-plane alignment:** Estimate and compensate for the primary background motion using the 2D parametric model.
2. **Secondary-plane alignment:** Identify remaining misaligned regions and fit additional 2D parametric motion models to account for secondary rigid planar structures.
3. **Residual parallax estimation:** Apply coarse-to-fine pyramidal Lucas–Kanade optical flow to the remaining misaligned regions, obtaining dense residual planar-parallax vectors $\vec{\mu}^{\,j}$ and $\vec{\mu}^{\,k}$ for the two inspection views.

This layered decomposition helps distinguish static 3D structures from independently moving objects, even when their image motion cannot be explained by a single planar transformation.

### 3. Three-View Projective Rigidity
After compensating for planar motion, the remaining displacement contains information about the underlying 3D scene geometry.

The key observation is that static 3D points must satisfy a common projective structure constraint across three views, consisting of a reference image $I$ and two inspection images $I_j$ and $I_k$.

Consider two image points $\mathbf{p}_1$ and $\mathbf{p}_2$, with relative image separation:

$$
\Delta\mathbf{p}_w = \mathbf{p}_2 - \mathbf{p}_1
$$

Let $\boldsymbol{\mu}_1^{\,j}$ and $\boldsymbol{\mu}_2^{\,j}$ denote their residual parallax vectors in inspection view $j$, with corresponding vectors in view $k$.

For static 3D points, the following three-frame parallax rigidity constraint must hold [1]:

$$
\frac{\boldsymbol{\mu}_2^{j,T} (\Delta\mathbf{p}_w)^j_{\perp}}{\boldsymbol{\mu}_1^{j,T} (\Delta\mathbf{p}_w)^j_{\perp}} - \frac{\boldsymbol{\mu}_2^{k,T} (\Delta\mathbf{p}_w)^k_{\perp}}{\boldsymbol{\mu}_1^{k,T} (\Delta\mathbf{p}_w)^k_{\perp}} = 0
$$

Here, $`(\Delta\mathbf{p}_w)^j_{\perp}`$ and $`(\Delta\mathbf{p}_w)^k_{\perp}`$ denote the corresponding perpendicular components of the warped inter-point separation vectors used by the plane-plus-parallax geometry.

The constraint measures the consistency of relative parallax across the two inspection views. Points that significantly violate this constraint relative to a consensus set of static reference points provide evidence of independent 3D motion.

Consequently, we use projective rigidity inconsistency as an additional source of motion evidence, reducing false detections caused by static scene depth variations.

### 4. Foreground Segmentation with MRF Graph Cuts
Motion evidence obtained from layered image alignment and projective rigidity is not always spatially coherent. Optical flow noise, low-texture regions, and local estimation errors can produce isolated false positives.

To obtain spatially consistent foreground masks, we convert the continuous motion evidence $s_p$ at each pixel $p$ into bounded log-odds unary potentials:

$$
\ell_p = \mathrm{clip}\left(k\ln\frac{\max(s_p,\varepsilon)}{\tau}, -\ell_{\max}, \ell_{\max}\right)
$$

$$
D_p(1) = \max(0,-\ell_p), \qquad D_p(0) = \max(0,\ell_p)
$$

where:
- $s_p$ is the motion evidence at pixel $p$.
- $k$ controls the sensitivity of the log-odds transformation.
- $\tau$ is the motion-evidence threshold.
- $\varepsilon$ ensures numerical stability.
- $\ell_{\max}$ limits the magnitude of the unary potentials.
- $L_p \in \{0,1\}$ represents the pixel label, where $0$ denotes static background and $1$ denotes independently moving foreground.

We formulate the segmentation problem as a binary Markov Random Field (MRF) energy minimization problem:

$$
E(L) = \sum_{p\in\mathcal{V}} D_p(L_p) + \sum_{(p,q)\in\mathcal{N}} w_{pq}\,\mathbb{I}[L_p \neq L_q]
$$

where $\mathcal{V}$ denotes the set of image pixels and $\mathcal{N}$ is an 8-connected neighborhood.

The contrast-sensitive Potts pairwise weight is defined as:

$$
w_{pq} = \frac{\lambda}{\|p-q\|_2}\left(w_0 + \exp\left(-\frac{(I_p-I_q)^2}{2\sigma^2}\right)\right)
$$

where:
- $\lambda$ controls the overall regularization strength.
- $w_0$ provides a baseline smoothness penalty.
- $\sigma$ controls sensitivity to image intensity differences.
- $I_p$ and $I_q$ represent the normalized image intensities at neighboring pixels.
- $\mathbb{I}[\cdot]$ is the indicator function.

The pairwise weights encourage neighboring pixels to share the same label while allowing discontinuities around strong image boundaries.

Since all pairwise weights are nonnegative, the binary Potts energy is submodular. Therefore, its global minimum can be computed exactly using an equivalent minimum $s\text{-}t$ cut / maximum-flow formulation.

MOD3D uses the Boykov–Kolmogorov algorithm to obtain the final spatially coherent foreground segmentation.

## Overall Pipeline
The complete detection pipeline can be summarized as follows:
1. Estimate dominant background motion using 2D parametric alignment.
2. Warp the inspection images toward the reference image.
3. Detect remaining misaligned regions.
4. Fit secondary parametric motion models to account for additional rigid planar structures.
5. Estimate residual planar parallax using pyramidal Lucas–Kanade optical flow.
6. Evaluate three-view projective rigidity to distinguish static 3D structure from independent motion.
7. Convert motion evidence into unary potentials and perform MRF graph-cut segmentation.
8. Obtain the final independently moving object masks.

## Acknowledgement
This implementation is heavily inspired by the theoretical framework and geometric insights presented in the following work:

[1] Irani, M., & Anandan, P. (1998). A unified approach to moving object detection in 2D and 3D scenes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(6), 577–589. https://doi.org/10.1109/34.683770

We acknowledge the authors for their foundational contributions to geometric motion analysis and moving object detection.
