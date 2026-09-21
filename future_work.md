# Future Work

This file records research directions beyond the current LocandKey CSI-denoising implementation. The items below are proposals, not completed or security-validated features.

## 1. Generalization and Denoising Research

Before building a security protocol on the reconstructed channels, the denoiser needs stronger evidence across data and architectures.

### Unseen-environment evaluation

Use leave-one-scenario-out experiments to measure whether a model trained on known environments transfers to an unseen one. The Phoenix holdout is the first such experiment: all Phoenix samples are excluded from training and validation, then compared with the existing mixed-scenario baseline on matched Phoenix indices and noise.

Future holdouts should cover indoor, outdoor, campus, sub-6 GHz, and millimeter-wave conditions. Holdout results must remain evaluation-only; tuning on them would turn the held-out environment into validation data.

### Architecture ablation

Train matched alternatives with identical splits, seeds, SNR sampling, epoch budgets, and evaluation code:

- plain convolutional autoencoder;
- vanilla U-Net;
- ConvNeXt U-Net without attention gates;
- ConvNeXt U-Net without bottleneck channel attention;
- the complete ResonanceModel.

Multiple random seeds are needed before attributing a difference to an architectural component rather than training variance.

### Real-world robustness

Validate with measured CSI that includes effects not fully represented by simulation:

- RF-chain mismatch and reciprocity calibration error;
- mobility and temporal channel evolution;
- synchronization and channel-estimation error;
- antenna-array and bandwidth changes;
- hardware and deployment domain shift.

## 2. Reciprocal CSI Collection

A physical-layer key-generation study requires paired observations from two legitimate endpoints, conventionally called Alice and Bob. They must probe the channel within its coherence interval so their observations are strongly correlated. An eavesdropper, Eve, should have a spatially separated observation for security analysis.

The present supervised denoising dataset provides clean/noisy reconstruction pairs. It does not yet establish paired reciprocal Alice/Bob measurements or an adversarial Eve dataset. Those data contracts must be defined before key-generation claims are made.

Required additions include:

- timestamped bidirectional channel probes;
- reciprocity calibration and alignment;
- coherence-time and mobility metadata;
- separated legitimate and adversarial observations;
- train/test isolation by location, session, and acquisition time.

## 3. CSI-to-Bit Quantization

After denoising, Alice and Bob would independently convert correlated channel measurements into candidate bits. Possible approaches include adaptive thresholding, multi-bit quantization, phase-based quantization, and guard bands that discard ambiguous samples.

The design must report both yield and agreement. A method that produces many bits with high disagreement is not useful, while an aggressive guard band may produce reliable but very few bits.

Key metrics include:

- key generation rate;
- raw key disagreement rate;
- retained samples after guard bands;
- entropy and bias of generated bits;
- sensitivity to SNR, mobility, and scenario;
- Eve's bit agreement and mutual information.

## 4. Information Reconciliation

Alice and Bob will generally obtain similar, not identical, bit strings. An authenticated reconciliation protocol can correct disagreements using error-correcting information exchanged over a public channel.

Any implementation must account for information leaked by reconciliation messages. Reporting only the post-reconciliation match rate would overstate security. Suitable future experiments should compare standard constructions such as secure sketches or code-based reconciliation and record their communication cost and leakage.

## 5. Privacy Amplification and Key Derivation

The reconciled string should not be used directly as an encryption key. Privacy amplification must compress it with a well-studied universal hash or standard key-derivation function, with the output length chosen from a defensible entropy estimate after subtracting reconciliation leakage and adversarial knowledge.

The protocol also needs:

- session separation and context binding;
- freshness guarantees;
- safe salt and public-parameter handling;
- key confirmation without revealing the key;
- explicit key erasure and lifetime rules.

Statistical randomness tests may reveal obvious defects, but passing them does not prove secrecy. Security analysis must connect the measured channel advantage, entropy estimate, public leakage, and final key length.

## 6. Authentication and Active Attack Resistance

Channel reciprocity alone does not authenticate the endpoints. An active attacker may relay, inject, jam, or manipulate channel probes. A real protocol therefore needs an authenticated public channel or an explicit bootstrapping assumption.

Future threat modeling should include:

- passive eavesdropping;
- man-in-the-middle attacks;
- pilot and signal injection;
- replay and relay attacks;
- selective jamming;
- denial of service;
- compromised endpoint state.

The security claim must state which attacks are prevented, detected, or outside scope.

## 7. Standard Authenticated Encryption

Once a validated shared key exists, payload protection should use a standard authenticated-encryption construction such as AES-GCM or ChaCha20-Poly1305 through a maintained cryptographic library. LocandKey should not implement a custom cipher.

The integration must define nonce generation, key rotation, associated data, authentication-failure handling, and message framing. Encryption success should be evaluated separately from CSI reconstruction quality.

## 8. Noise-Like or Covert Transmission

Making a waveform appear noise-like is a separate research objective from encrypting its contents. Encryption hides meaning; it does not necessarily hide that communication is taking place.

A future covert-communication study would need its own channel model, transmitter and receiver design, power constraints, synchronization method, and detector-aware evaluation. Relevant measures include detection probability, false-alarm probability, throughput, bit-error rate, and robustness to channel mismatch.

This topic should not be presented as an automatic consequence of CSI denoising or cryptographic key generation.

## 9. End-to-End Evaluation Plan

A future secure-channel pipeline should be accepted only after independently validating each stage:

1. **Denoising:** positive reconstruction gain on known and unseen environments.
2. **Reciprocity:** strong Alice/Bob correlation and materially weaker Eve correlation.
3. **Quantization:** useful key rate with low raw disagreement and measured entropy.
4. **Reconciliation:** high agreement with explicit leakage accounting.
5. **Privacy amplification:** justified final key length and successful key confirmation.
6. **Authentication:** documented resistance to the selected active attacks.
7. **Encryption:** standard AEAD interoperability and tamper rejection.
8. **System testing:** repeated trials across SNR, mobility, distance, hardware, and environments.

Until those stages are implemented and tested, the defensible description of LocandKey is a multi-scenario complex-CSI denoising system and a research foundation for later physical-layer security work.
