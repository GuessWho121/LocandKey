# LocandKey
Phisical layer key generation using Network CSI matrix along a pretrained ai that clean the matix of noise for a better output

## 1. Project Idea In One Sentence

This project aims to help two wireless devices create a shared secret key from their wireless connection, use that key to encrypt a message, and then send the encrypted message in a way that looks like noise to outsiders.

## 2. Why This Project Is Needed

Whenever two devices communicate, they need security.

For example:

- A phone talks to a Wi-Fi router.
- A car talks to a roadside unit.
- A drone talks to a control station.
- A mobile device talks to a 5G base station.

If the message is not protected, another person nearby may try to listen to it.

Normally, secure systems depend on secret keys. A secret key is like a password used to lock and unlock a message.

But there is one big question:

**How do two devices get the same secret key without directly sending the key over the air?**

If the key is sent openly, an attacker may capture it.

This project tries to solve that problem by using the wireless environment itself.

## 3. Basic Story Of The Project

Imagine three people:

- **Alice**: the sender
- **Bob**: the receiver
- **Eve**: the attacker trying to listen

Alice and Bob are connected wirelessly.

The wireless path between Alice and Bob has a unique pattern. This pattern is created by:

- distance,
- walls,
- reflections,
- movement,
- obstacles,
- signal strength changes,
- and the physical location of both devices.

Alice and Bob can both observe this same wireless pattern.

Eve is in a different physical position, so she observes a different wireless pattern.

The main idea is:

```text
Alice and Bob use their shared wireless pattern to generate the same secret key.
Eve sees a different pattern, so she generates the wrong key.
```

## 4. What Is A Wireless Channel?

A wireless signal does not travel in a perfect straight line.

When Alice sends a signal to Bob, the signal may:

- bounce off walls,
- reflect from cars,
- pass through objects,
- arrive through multiple paths,
- become weaker,
- shift slightly in timing and phase.

The final signal received by Bob is changed by the environment.

This environment effect is called the **wireless channel**.

In simple words:

```text
Wireless channel = the fingerprint of the wireless path between two devices
```

This fingerprint is different for different locations.

## 5. What Is CSI?

CSI stands for **Channel State Information**.

In simple words:

```text
CSI = measured details of the wireless channel
```

It tells us how the wireless signal changed while traveling from sender to receiver.

For this project, CSI can be imagined as a large table of numbers.

Each number describes a small part of the wireless signal behavior.

The CSI contains information about:

- signal strength,
- signal direction,
- reflections,
- delays,
- and phase changes.

## 6. Why CSI Can Be Used For Security

Alice and Bob are connected through the same wireless path.

Because of this, they can measure very similar CSI.

Eve is somewhere else, so her CSI is different.

So:

```text
Alice's CSI is similar to Bob's CSI.
Eve's CSI is different from Bob's CSI.
```

This makes CSI useful for creating secret keys.

The key is not sent over the network.

Instead:

```text
Alice creates the key from her CSI.
Bob creates the key from his CSI.
Eve creates a different key because her CSI is different.
```

## 7. Is The Noise The Key?

No.

The noise is not the key.

The key comes from the wireless channel pattern.

Noise is actually a problem because it can make Alice and Bob's measurements slightly different.

So:

```text
Wireless channel pattern = source of the key
Noise = disturbance
AI model = cleaner
Final key = secure password created from the cleaned pattern
```

## 8. Role Of AI In This Project

Wireless measurements are not perfect.

Alice and Bob may observe almost the same channel, but their measurements can still differ because of:

- noise,
- movement,
- weak signal strength,
- hardware imperfections,
- timing differences.

If their measurements differ too much, they may generate different keys.

This is where AI is planned.

The AI model will learn how to clean noisy wireless channel data.

The goal of the AI is:

```text
Input: noisy CSI
Output: cleaner CSI
```

After cleaning, Alice and Bob should generate more similar keys.

The AI is not the encryption system.

The AI helps prepare better channel data before the key is generated.

## 9. Full Project Workflow

The complete project works like this:

```text
1. Alice and Bob measure the wireless channel.
2. Their measured CSI contains noise.
3. AI cleans the noisy CSI.
4. Cleaned CSI is converted into bits.
5. Alice and Bob fix small bit mismatches.
6. The final bits are converted into a secret key.
7. Alice encrypts a message using the key.
8. The encrypted message is hidden inside a noise-like signal.
9. Bob uses the same key to recover and decrypt the message.
10. Eve fails because her key is different.
```

## 10. How The Key Is Generated From CSI

The key generation process has several steps.

### Step 1: Measure CSI

Alice and Bob measure the wireless channel.

```text
Alice gets CSI_A
Bob gets CSI_B
```

These two should be similar.

Eve gets her own CSI:

```text
Eve gets CSI_E
```

This should be different.

### Step 2: Clean CSI Using AI

The planned AI model takes noisy CSI and produces cleaner CSI.

```text
Noisy CSI -> AI model -> Cleaned CSI
```

This helps Alice and Bob generate matching keys.

### Step 3: Convert CSI Into Useful Values

CSI is made of many numbers.

These numbers are processed to extract useful wireless features.

For example, the system may look at:

- strong signal parts,
- reflection patterns,
- delay patterns,
- stable values that are less affected by noise.

### Step 4: Convert Values Into Bits

A key is made of bits.

A bit is either:

```text
0 or 1
```

So the wireless values must be converted into bits.

Example:

```text
High value -> 1
Low value  -> 0
```

If a value is unclear, it is ignored.

Example:

```text
Very high value -> 1
Very low value  -> 0
Middle value    -> discard
```

This prevents unstable values from damaging the key.

### Step 5: Fix Small Differences

Alice and Bob may still have a few different bits.

Example:

```text
Alice: 10110110
Bob:   10100110
```

Only one bit is different.

The system will use a correction method to fix small mismatches.

Alice and Bob exchange limited checking information, but they do not reveal the full key.

### Step 6: Create Final Key

After correction, Alice and Bob should have the same bit sequence.

This bit sequence is then passed through a secure hashing method.

The output becomes a clean final key.

```text
Corrected bits -> secure hash -> final 256-bit key
```

This key is used for encryption.

## 11. Symmetric Or Asymmetric Encryption?

This project uses **symmetric encryption**.

That means:

```text
Same key is used for encryption and decryption.
```

Alice uses the key to lock the message.

Bob uses the same key to unlock the message.

This fits the project perfectly because Alice and Bob both generate the same key from the wireless channel.

The project does not mainly use asymmetric encryption.

In asymmetric encryption, there are two keys:

```text
public key  = shared with others
private key = kept secret
```

But our project does not need that because the wireless channel helps Alice and Bob create one shared secret key directly.

So the encryption style is:

```text
CSI-generated shared key -> AES encryption
```

## 12. How Message Encryption Works

After Alice and Bob generate the same key, Alice can encrypt a message.

Example message:

```text
Meet at 10 PM
```

Alice uses the generated key to turn it into unreadable data.

```text
Meet at 10 PM -> encrypted data
```

Bob uses the same key to recover the message.

```text
encrypted data -> Meet at 10 PM
```

Eve tries using her own key, but her key is wrong.

So Eve fails to decrypt the message.

## 13. Hiding The Encrypted Message Inside Noise

The project adds one more layer.

Instead of only encrypting the message, the encrypted data is also sent in a noise-like form.

This means the transmission should look random to outsiders.

Alice does this:

```text
Encrypted message + secret key -> noise-like signal
```

Bob does this:

```text
noise-like signal + same secret key -> encrypted message -> original message
```

Eve sees only something that looks like random noise.

Since Eve does not have the correct key, she cannot easily recover the hidden encrypted data.

## 14. Why Not Hide The Key Itself?

The project should not send or hide the key.

The key is created independently by Alice and Bob from the wireless channel.

Only the encrypted message is transmitted.

This is safer because:

```text
The secret key never travels over the air.
```

## 15. What The Attacker Can And Cannot Do

Eve may be on the same network or in the same cafe.

That does not automatically mean she gets the same key.

The key depends on the exact physical wireless path.

If Eve is in a different position, her channel pattern will usually be different.

However, if Eve is extremely close to Bob, her channel may become more similar.

So the project must honestly mention this limitation.

```text
Eve far from Bob    -> safer
Eve near Bob        -> less safe
Eve extremely close -> risky
```

## 16. What Happens If Devices Move?

Devices do not need to be perfectly still.

Some movement can actually help because it creates fresh wireless patterns.

But too much movement can make Alice and Bob's measurements less similar.

So the project will test different cases:

```text
Low movement    -> high key match
Medium movement -> AI should help
High movement   -> key mismatch may increase
```

## 17. Expected Results

The project should show:

```text
Alice and Bob generate matching keys.
Eve generates a different key.
Bob decrypts the message successfully.
Eve fails to decrypt the message.
AI improves Alice-Bob key matching.
Hidden transmission looks noise-like to outsiders.
```

## 18. Main Evaluation Metrics

The project will measure:

### Key Match Rate

How often Alice and Bob generate the same key.

```text
Higher is better.
```

### Eve Match Rate

How often Eve's key matches Alice and Bob's key.

```text
Lower is better.
```

### Decryption Success

Whether Bob can decrypt the message.

```text
Bob should succeed.
Eve should fail.
```

### Hidden Signal Recovery

Whether Bob can recover the encrypted data from the noise-like signal.

```text
Bob should recover it.
Eve should not.
```

### Performance Under Noise

The project will test different noise levels.

```text
Low noise  -> easier
High noise -> harder
```

## 19. Planned Technology Stack

The project will be built using:

- Python for implementation
- NumPy for numerical processing
- PyTorch for the AI model
- Matplotlib for graphs
- Cryptography library for secure encryption
- DeepMIMO-style wireless data for CSI samples

## 20. Main Parts To Build

### Part 1: CSI Data Preparation

Prepare wireless channel samples.

Goal:

```text
Create Alice, Bob, and Eve channel observations.
```

### Part 2: AI-Based CSI Cleaning

Build or train an AI model that can clean noisy CSI.

Goal:

```text
Make Alice and Bob's CSI more similar.
```

### Part 3: Key Generation

Convert cleaned CSI into secret bits.

Goal:

```text
Alice and Bob produce the same key.
Eve produces a different key.
```

### Part 4: Encryption

Use the generated key to encrypt a message.

Goal:

```text
Only Bob can decrypt the message.
```

### Part 5: Noise-Like Transmission

Hide the encrypted data inside a random-looking signal.

Goal:

```text
The message should look like noise to Eve.
```

### Part 6: Evaluation

Test the system under different conditions.

Goal:

```text
Prove that Bob succeeds and Eve fails.
```

## 21. Final Demo Example

The final demo can look like this:

```text
Original message:
"Meet at 10 PM"

Alice key:
Generated from Alice's CSI

Bob key:
Generated from Bob's CSI

Eve key:
Generated from Eve's CSI

Result:
Bob decrypts successfully.
Eve fails to decrypt.
```

Output:

```text
Bob received: Meet at 10 PM
Eve received: Decryption failed
```

## 22. Final Project Conclusion

This project combines wireless communication, artificial intelligence, and cryptography.

The main idea is:

```text
Use the wireless environment itself to create a secret key.
Use AI to make that key generation more reliable.
Use the key to encrypt data.
Send the encrypted data in a noise-like form.
Allow Bob to recover the message.
Prevent Eve from reading it.
```

This makes the project suitable for secure communication in future wireless systems such as 5G, 6G, drones, IoT devices, smart vehicles, and mobile networks.

## PyTorch Training (Windows)

From this project folder, create a separate Python 3.11 training environment with uv:

```powershell
uv venv --python 3.11 .torchvenv
uv pip install --python .torchvenv\Scripts\python.exe torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .torchvenv\Scripts\python.exe -r requirements.txt
.\.torchvenv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python model.py
python train.py --smoke --batch-size 4 --mixed-precision --require-gpu
python train.py --epochs 60 --batch-size 4 --mixed-precision --require-gpu
python eval.py --batch-size 4
```

The CUDA 13.2 wheel is intended for the RTX 5070 laptop with the reported driver.
See the [official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/).
Ubuntu and a separate CUDA Toolkit installation are not needed for these wheels.
For CPU checks, install torch from the `/whl/cpu` index and omit the GPU flags.
Keep `.genvenv` for DeepMIMO. Existing data and split files are reused without regeneration.

The NumPy files remain `(N, 128, 256, 2)`; training converts each batch to PyTorch's
`(N, 2, 128, 256)` layout. Evaluation converts predictions back before computing metrics.
The ConvNeXt U-Net, channel attention, attention gates, NMSE plus magnitude loss,
cosine learning rate, gradient clipping, early stopping, CSV logs and TensorBoard are retained.
PyTorch initialization differs from Keras, so retraining and evaluation are required.
TensorFlow `.h5` weights are preserved but cannot be loaded by this implementation.

Best weights: `weights/locandkey_multiscenario_best.pt`.
Smoke weights and logs use separate `weights/locandkey_smoke.pt` and `logs/smoke` paths.
Run `tensorboard --logdir logs` to view training curves.
Run `python test_pytorch.py` for a synthetic CPU training/checkpoint/evaluation check;
it creates temporary data and does not train on your generated scenarios.

### Stop and Resume

The best `.pt` file remains weights-only for evaluation. Each completed epoch also
saves a full checkpoint at `weights/locandkey_multiscenario_best.pt.last.pt`.
This includes the latest model, best model, optimizer, scheduler, AMP scaler,
epoch, early-stopping state and random states. Writes use a temporary file followed
by replacement. Ctrl+C or a crash loses the unfinished epoch; resume repeats that
epoch from its beginning. No full checkpoint exists until the first epoch finishes.

For a run started with this version:

```powershell
python train.py --resume --require-gpu
```

Resume restores the saved training settings, including the original total epoch
target and precision mode. Use the same data and splits. If you used custom output
paths, supply those again; `--resume PATH` selects a specific full checkpoint.
CSV logs append during resume. A completed or early-stopped run is not restarted.

For your already-running, older script, wait for a best checkpoint, stop it, then
start from those weights with a new output name to preserve the original:

```powershell
python train.py --init-weights weights/locandkey_multiscenario_best.pt --weights-path weights/locandkey_continued_best.pt --logs-dir logs/continued --epochs 60 --batch-size 4 --mixed-precision --require-gpu
```

This starts 60 new epochs with a fresh optimizer and schedule; the old script never
saved those states. Subsequent interruptions of that run can be resumed using:

```powershell
python train.py --resume --weights-path weights/locandkey_continued_best.pt --logs-dir logs/continued --require-gpu
```

Updating files does not change an already-running Python process. Install the
updated `train.py` on the training laptop before starting either command.

## Train Without Phoenix

This diagnostic excludes Phoenix from training and validation. It uses existing
channel files and preserves every original split index for the other scenarios.
Run from the project folder on the training laptop, where data and weights exist:

```powershell
python preprocess.py --holdout-scenario city_4_phoenix_28 --source-split-dir .\data\splits --output-dir .\data\splits\holdout_phoenix
python train.py --split-dir .\data\splits\holdout_phoenix --weights-path .\weights\holdout_phoenix_best.pt --logs-dir .\logs\holdout_phoenix --epochs 60 --patience 10 --batch-size 4 --seed 42 --mixed-precision --require-gpu
python eval.py --split-dir .\data\splits\holdout_phoenix --weights-path .\weights\holdout_phoenix_best.pt --baseline-weights .\weights\locandkey_multiscenario_best.pt --output-dir .\visualizations\holdout_phoenix --batch-size 4
```

Keep the original channel files, splits, best weights and evaluation outputs.
Do not regenerate original splits. The split command rejects an existing output
and rejects overwriting the source. Phoenix's old train/validation/test union
becomes its new test set; its original test indices are stored separately in the
same NPZ file. No channel arrays are copied. Source counts, bounds, uniqueness
and disjointness are checked before writing.

The holdout model starts from scratch with the existing architecture and defaults:
Adam at 0.0003, cosine decay to 0.000001, noise from -10 to 30 dB, validation at
10 dB, and NMSE plus 0.05 magnitude loss. `--init-weights` is blocked for this
experiment. Resume uses a fingerprint of both split files to reject checkpoints
from other splits (including old checkpoints without a fingerprint):

```powershell
python train.py --resume --split-dir .\data\splits\holdout_phoenix --weights-path .\weights\holdout_phoenix_best.pt --logs-dir .\logs\holdout_phoenix --require-gpu
```

Evaluation keeps `metrics.csv`, `nmse_vs_snr.png`, and `gain_by_scenario.png`.
The additional outputs are:

- `group_metrics.csv`: seen scenarios combined, and all usable Phoenix samples.
- `original_holdout_test.csv`: Phoenix restricted to the original test indices.
- `baseline_comparison.csv`: baseline and new model on identical test samples
  and identical noisy inputs; positive degradation means the new model is worse.
- `holdout_nmse_vs_snr.png`: unseen Phoenix noisy versus denoised CSI.
- `evaluation_summary.json`: all results, split fingerprint and decision flags.

The optional `--baseline-weights` re-evaluates the old model to make the comparison
controlled; it does not trust rounded values or different sample sets from an old
report. Without it, baseline comparison is explicitly unavailable. Only the
original Phoenix test subset is compared with that model; its previous training
samples must not be treated as an independent baseline test. Do not use sample
limits for the final report. The same test samples are reused across nine SNRs.

Decision rules: Phoenix should gain at least 2 dB at both 0 and 10 dB; flag any
Phoenix SNR with negative gain. Flag each seen scenario that worsens by more than
1 dB against the baseline at 0 or 10 dB. Missing target SNRs leave the Phoenix
target undecided. Inspect validation convergence in `logs/holdout_phoenix` before
interpreting failure as poor generalization. Do not tune against Phoenix.

Phoenix was chosen after inspecting prior results: this is an environment-exclusion
diagnostic, not an untouched final benchmark. No architecture or key-generation
changes are included. Run `python test_pytorch.py` to verify split isolation, invalid
input rejection, fresh training, resume guards, and matched evaluation on temporary
synthetic data before the real experiment.
