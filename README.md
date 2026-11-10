# MOD3D
## INTRODUCTION
This repository contains the implementation and description of the Moving Detection Algorithm originally outlined by Michal Irani and P. Anandan in their paper titled "A Unified Approach to Moving Object Detection in 2D and 3D Scenes" published in the IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 20, No.6, June 1998. The implementation is coded by a group of researcher from Ho Chi Minh University of Technology - Vietnam National University (HCMUT) for the purpose of studying and analysing traditional machine learning methods to moving object detection. The algorithm made extensive use of epipolar constraints in computer vision, in conjunction with co-moving planar surface assumption. Please refer to the original paper for proofs of the assumptions and theorems.

## MOVING DETECTION ALGORITHM
Assuming a 2D scene, that is a 3D scene that can be represented or modelled according to a plane in 3D space. We have the following equation to determine the optical flow:

$$ \begin{bmatrix} u(x, y) \\ v(x, y) \end{bmatrix} = \begin{bmatrix} a + b \cdot x + c \cdot y + g \cdot x^2 + h \cdot xy \\ d + e \cdot x + f \cdot y + g \cdot xy + h \cdot y^2 \end{bmatrix} $$

This formulation works well when the 3D scene is far away from the camera, or that it can be sufficiently modelled as to be on the plane. Using this formulation, we can easily compute the Intensity Difference between the Reference Image and the Inspection Image then mask the area with large intensity differences as motion.

Unfortunately, when the camera is sufficiently close to the 3D scene containing a lot of objects such that a single plane cannot accurately describe the scene, the above formulation naturally fails. Notably, parallax can quickly degrade our ability to detect motion objects because this formulation wrongly assumes parallax situations to be the same as motion. 

We can remedy this problem by using multiple planes. First we use the 2D parametric formulation to align the Reference Image and the Inspection Image. We then mask the areas that are not aligned. We apply 2D parametric alignment on the areas that are not aligned. We then subtract the result of the previous 2D parametric alignment from the current 2D parametric alignment on the misaligned area to obtain the parallax vector. We then use the following test to determine whether a point in the image is independently moving or the result of parallax. 

$$ \frac{\vec{\mu}_2^{j^T} \left( \Delta \vec{p}_w \right)^j_\perp}{\vec{\mu}_1^{j^T} \left( \Delta \vec{p}_w \right)^j_\perp} - \frac{\vec{\mu}_2^{k^T} \left( \Delta \vec{p}_w \right)^k_\perp}{\vec{\mu}_1^{k^T} \left( \Delta \vec{p}_w \right)^k_\perp} = 0. $$

## ACKNOWLEDGEMENT

Our algorithm made extensive use of the ideas outlined by the following paper, Irani, M., & Anandan, P. (1998). A unified approach to moving object detection in 2D and 3D scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(6), 577-589.
