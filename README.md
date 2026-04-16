# A-Two-Stage-Fully-Unsupervised-Learning-Framework-for-Liquid-State-Machines

This project implements a novel Dual‑Path Self‑Organizing Map (Dual SOM) classification system designed for image recognition tasks such as Fashion‑MNIST. The system dynamically routes samples based on a confidence evaluation mechanism:

Easy Samples: Processed rapidly by a Spiking SOM, generating memory vectors that are passed through a random projection and then classified by a traditional SOM.

Hard Samples: Routed to an LSM‑STDP (Liquid State Machine with Spike‑Timing‑Dependent Plasticity) deep feature extractor, which produces high‑dimensional spatio‑temporal features before final classification by the traditional SOM.

The framework combines the temporal processing capabilities of Spiking Neural Networks (SNNs) with the unsupervised clustering of SOMs and adaptive computational resource allocation, aiming for efficient and robust image recognition.

=================================================================================================================
Hybrid Spike Encoding: Combines TTFS (Time‑To‑First‑Spike) encoding with Poisson encoding to balance rapid response and information richness.

Adaptive Dual‑Path Architecture:

Fast Path (Spiking SOM + Memory Layer): Handles high‑confidence samples with low computational cost.

Deep Path (LSM‑STDP): Processes low‑confidence samples, extracting complex spatio‑temporal features.

Rich Visualization and Monitoring:

Memory Vector Visualization: Visualizes the information encoding of the Spiking SOM memory layer.

LSM Feature Visualization: Analyzes the spatio‑temporal feature distributions from the deep pathway.

t‑SNE Dimensionality Reduction: Evaluates clustering quality in the feature space (supports outlier removal).

Weight and Statistics Curves: Monitors quantization error, neuron activation rates, and other training metrics in real time.

Boundary Neuron Identification: Identifies neurons near class boundaries within the Spiking SOM to refine routing decisions.

=================================================================================================================
