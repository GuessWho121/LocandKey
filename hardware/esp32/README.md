# ESP32 Host-Assisted Capture Contract

The current bench pair is two ESP32-D0WD-V3 boards, connected as Alice on COM5 and Bob on COM7. `securelink_probe/securelink_probe.ino` uses Espressif's ESP-NOW and CSI APIs through the installed Arduino ESP32 core. The planned product target remains two matching ESP32-S3 boards with external 2.4 GHz antennas. Cryptographic processing stays in Python until real capture gates pass.

The sketch identifies this particular pair by their Wi-Fi MAC addresses. Upload it to each board using the bundled Arduino CLI or open the sketch in Arduino IDE and select **ESP32 Dev Module**. The reliable upload speed on this pair was 115200 baud; the initial 921600-baud attempt stopped mid-write. The configured ESP-NOW channel is 6; a temporary channel-11 comparison is documented below. The sketch does not need an SSID or password. It captures HT-LTF only, records packet rate, MCS, channel width, antenna, receive status, and the invalid-first-word flag with the CSI, and sends Bob's replies from a separate task so serial printing does not delay them. Invalid first-word bytes are zeroed before capture.

The CSI probe now also relays Bob's CSI to Alice using encrypted ESP-NOW unicast after each probe reply. Both boards must already have the 32-byte pair secret provisioned by the encrypted-reading sketch; the probe derives a separate telemetry key and refuses to start if the secret is missing. Alice prints both `device: alice` and `device: bob` records over her USB serial port. Bob keeps printing his own records over USB when connected. Relay records are bound to Alice's random capture ID and sequence; missing packets are reported, not retried during probing.

```powershell
$cli = "$env:LOCALAPPDATA\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
& $cli compile --fqbn esp32:esp32:esp32 hardware\esp32\securelink_probe
& $cli upload --fqbn esp32:esp32:esp32:UploadSpeed=115200 --port COM5 hardware\esp32\securelink_probe
& $cli upload --fqbn esp32:esp32:esp32:UploadSpeed=115200 --port COM7 hardware\esp32\securelink_probe
```

Run one monitor for both ports. It prints sequence, RSSI, round-trip time, and CSI length while saving full JSONL captures. Press Ctrl+C to stop it before opening Arduino IDE's Serial Monitor or flashing again, because only one program can own a COM port at a time.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File hardware\esp32\capture_serial.ps1 `
  -DurationSeconds 30 -OutputDir captures\esp32\smoke
```

For a power-only Bob within ESP-NOW range, leave Alice on COM5 and use the one-port mode:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File hardware\esp32\capture_serial.ps1 `
  -RelayBob -DurationSeconds 90 -OutputDir captures\esp32\relay-test
```

It writes the same `alice.jsonl` and `bob.jsonl` files as two-port mode and reports sequence pairs, gaps, and duplicates. The files still work with `securelink_capture.py` and `hardware.esp32.diagnose_csi`. The paired key evaluator's 5 ms timing filter can remove additional pairs. This is local diagnostic transport: it does not extend radio range, prove CSI secrecy, or justify activating physical keys from relayed measurements.

Each board writes one JSON object per CSI packet over USB serial:

```json
{"device":"alice","sequence":1,"timestamp_us":1000,"round_trip_us":900,"rssi_dbm":-48,"noise_floor_dbm":-94,"channel":6,"bandwidth_mhz":20,"signal_mode":1,"csi_length":128,"csi":[12,-3,8,1]}
```

`csi` is an interleaved real/imaginary integer list. Alice and Bob must use the same channel, bandwidth, subcarrier count, and sequence number. Their local clocks are unsynchronized, so the matcher checks Alice's same-clock `round_trip_us` against the 5 ms limit when it is present. The sketch sends open ESP-NOW probes on its configured channel; it does not use Wi-Fi credentials or provide radio authentication.

Capture at least four reciprocal rounds with eight packets per round. Eight packet positions multiplied by 64 complex subcarriers gives 512 candidate features before the reliability guard removes weak values. Fewer packet positions may not retain the 256 bits required by the current gate.

Convert captured JSONL files with:

```powershell
.\.venv\Scripts\python.exe securelink_capture.py `
  --alice-jsonl captures\alice.jsonl `
  --bob-jsonl captures\bob.jsonl `
  --output captures\paired_csi.npz
```

The output contains `alice` and `bob` arrays shaped `(rounds, packets_per_round, subcarriers, 2)`. On this stationary pair, moving from serial-blocked replies to a separate reply task reduced p95 on-air round-trip time from 52.1 ms to 5.3 ms and raised matches within 5 ms from 516 to 816 in comparable 90-second captures. All 25 post-fix four-round sessions still fell back because their CSI changes were not reciprocal. Keeping the boards fixed is supported by authenticated-secret fallback; accepting a physical key requires new evidence of shared unpredictable channel variation. More placements and an active condition remain untested.

Run `python -m hardware.esp32.diagnose_csi <alice.jsonl> <bob.jsonl> --output <diagnostics.json>` from the repo root to compare packet metadata, repeatability on each board, and paired temporal changes. An assert-based check is `python -m hardware.esp32.test_diagnose_csi`. In two stationary captures, 541 and 283 packets matched. Both boards used the same channel, width, mode, rate, and MCS on every matched packet; receive status was zero, and the flagged invalid first CSI word was masked. The median change between consecutive packets was 3.9-4.0% on Alice and 8.9-9.4% on Bob. After removing whole-packet gain, the remaining shape change was about 1.8-2.0% on each. Paired temporal correlation remained -0.001/0.018 raw and 0.233/0.049 after gain normalization, below the 0.8 gate. These data suggest receiver-side gain variation contributes to the mismatch, but do not identify its exact cause or establish fresh secret entropy. Reports: `captures/esp32/csi-diagnostics-2026-10-23/diagnostics.json` and `captures/esp32/csi-diagnostics-repeat-2026-10-23/diagnostics.json`.

Two fresh 90-second fixed-board ambient captures on 2026-10-27 matched 819 and 788 packets. Their p95 round-trip times were 3.92 and 3.98 ms. The first had 0.857 gain-normalized temporal correlation across the entire capture, yet no four-round session reached the 0.8 reciprocity gate (best 0.738); the second peaked at 0.304. All 49 sessions fell back. Median adjacent paired-phase-offset changes were 91.3 and 90.6 degrees; phase instability is a separate observation because the current key gate uses CSI magnitudes. The encrypted-reading sketch was restored afterward and its link test passed. Reports: `captures/esp32/fixed-2026-10-27-a/`, `captures/esp32/fixed-2026-10-27-b/`, and `captures/esp32/sensor-link-restored-2026-10-27-retry/summary.json`.

The encrypted relay bench run on 2026-10-27 received 709 Bob records on Alice's USB output. All 709 also appeared on Bob's independent USB output with identical JSON data. The 90-second capture produced 640 Alice/Bob pairs within 5 ms, 20 complete four-round sessions, and 4.01 ms p95 round-trip time. All 20 sessions stayed in fallback. The final firmware build passed a second one-port smoke capture. Reports: `captures/esp32/relay-bench-2026-10-27/relay_validation.json` and `captures/esp32/relay-final-smoke-2026-10-27/`.

With Bob powered across the room, two 90-second one-port captures produced 592 and 575 pairs within 5 ms, or 18 and 17 complete four-round sessions. Alice-only/Bob-only record counts were 0/44 and 1/78; another 69 and 62 sequence numbers had no record from either side. The p95 round-trip times were 3.68 and 3.46 ms, and every session remained in fallback. All observed records within each capture had the same capture ID, and no duplicates or rejected relay records were reported. The 20-session remote target has not passed. A 230400-baud Alice trial corrupted serial records, so the verified 115200-baud firmware was restored and checked with a 10-second one-port capture. Reports: `captures/esp32/relay-remote-2026-10-27/` and `captures/esp32/relay-remote-repeat-2026-10-27/`.

After Bob returned to COM7 on 2026-10-28, both boards were restored to the encrypted-reading sketch without reprovisioning. Its link test passed with 121 accepted readings, zero duplicate acceptances, two replay rejects, and one tamper reject. This run did not repeat the physical power-cycle check. Report: `captures/esp32/sensor-link-restored-2026-10-28/summary.json`.

The next CSI probe build adds periodic `alice_stats`, `bob_stats`, and encrypted-piggybacked `bob_relay_stats` counters. A 90-second two-USB capture on 2026-10-28 produced 646 pairs within 5 ms and 20 complete fallback sessions, with no reported queue drops. Alice printed both streams at 115200 baud; their JSON size implies about 98% line capacity before status text, so the remote counter run must distinguish serial pressure from radio losses. One malformed local Bob serial line was rejected by the host parser. Report: `captures/esp32/counter-bench-2026-10-28/` and `captures/esp32/counter-bench-console-2026-10-28.log`.

The 2026-10-28 power-only remote counter run shows a different limit: between its first and last periodic snapshots, Alice queued 734 probes and Bob saw 538, while Alice captured 477 reciprocal CSI replies. Bob's CSI and reply-queue counts tracked his received probes; all firmware queue-drop counts were zero. The 90-second host capture yielded 503 Alice and 567 Bob records, 474 pairs within 5 ms, and 14 complete four-round sessions, all in fallback; there were no duplicates. The remote RSSI was substantially weaker than on the two-USB bench. The main observed loss is probe reception at Bob, followed by some missing replies at Alice; serial output alone does not explain it. Reports: `captures/esp32/counter-remote-2026-10-28/` and `captures/esp32/counter-remote-console-2026-10-28.log`.

Both boards were restored to the encrypted-reading sketch without reprovisioning after Bob returned to COM7. The first test attempt missed Bob's garbled startup status; the unchanged retry passed with 121 accepted readings, zero duplicate acceptances, two replay rejects, and one tamper reject. The physical power-cycle check was not repeated. Report: `captures/esp32/sensor-link-restored-counter-2026-10-28-retry/summary.json`.

For a controlled radio comparison, both boards were flashed with the otherwise unchanged CSI probe on channel 11. A 15-second two-USB smoke captured 119 Alice and 118 Bob records, all tagged channel 11 and 20 MHz, with 118 sequence pairs and no duplicates or malformed records. The unchanged 5 ms host matcher formed three four-round sessions and kept all three in fallback. Report: `captures/esp32/channel11-bench-smoke-2026-10-28/`.

Two 90-second power-only channel-11 captures at the prior across-room spot produced 557 and 536 pairs within 5 ms and 17 and 16 complete four-round sessions, against 474 pairs and 14 sessions in the channel-6 counter capture. The first channel-11 run kept all 17 sessions in fallback; the second had 15 fallbacks and one host-only `PHY_ACTIVE` result. That result used 32 matched packets spanning 50 probe sequence numbers, so 18 probes were skipped in its assembled window. It confirmed a key and host-local payload only; neither ESP32 derived a physical key, and no Eve or controlled-static condition was tested. The 20-session remote transport gate still fails. Bob's median RSSI was -77 dBm in the first run and -83 dBm in the repeat, so the observed channel comparison is confounded by changing radio conditions. The source sketch was reset to channel 6 after the trial; the boards remain on the flashed channel-11 probe until the encrypted-reading sketch is restored. Reports: `captures/esp32/channel11-remote-2026-10-28/` and `captures/esp32/channel11-remote-repeat-2026-10-28/`.

Check the captured tensors with `python evaluate_securelink_capture.py captures\esp32\smoke\paired_csi.npz`. The output distinguishes the raw endpoint handshake from the host's reciprocity gate. An endpoint confirmation alone is not permission to use a physical key; the host policy falls back if the joint channel gate rejects the sample. No encrypted payload is sent over the radio in this host-assisted test.

## Fixed-Node Encrypted Link

`securelink_sensor/securelink_sensor.ino` is a separate bench sketch for the same two boards. It sends encrypted ESP-NOW unicast from Alice (COM5) to Bob (COM7) without using stationary CSI as key material. The reading is Alice's uptime in milliseconds; no physical sensor is attached. The sketch derives link and AES-256-GCM keys from a 32-byte pair secret, stores a boot counter before sending, and stores Bob's last accepted boot/sequence before reporting a reading. Replays and altered GCM tags are rejected.

Close any serial monitor before flashing or testing, then compile and upload to both boards:

```powershell
$cli = "$env:LOCALAPPDATA\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
& $cli compile --fqbn esp32:esp32:esp32 hardware\esp32\securelink_sensor
& $cli upload --fqbn esp32:esp32:esp32:UploadSpeed=115200 --port COM7 hardware\esp32\securelink_sensor
& $cli upload --fqbn esp32:esp32:esp32:UploadSpeed=115200 --port COM5 hardware\esp32\securelink_sensor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File hardware\esp32\test_sensor_link.ps1 -Provision
```

Use `-Provision` only when **both** boards have no SecureLink secret in NVS. The harness generates one random secret in host memory, sends it to each board over local USB, and does not write it to source or logs. Later runs omit `-Provision`. Each run sends 120 readings, injects a replay and a bad tag, reboots Bob to check persistent replay rejection, and reboots Alice to check a new boot counter. Results and serial logs are saved under `captures/esp32/sensor-link-test/` unless `-OutputDir` is supplied.

Add `-Live` to print both boards' events in the terminal while the test runs. Do not open Arduino Serial Monitor at the same time; the test owns both COM ports.

To associate both boards with a 2.4 GHz hotspot while retaining their direct encrypted ESP-NOW link, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File hardware\esp32\test_sensor_link.ps1 `
  -HotspotSsid BlackMetal -Live -OutputDir captures\esp32\sensor-link-hotspot-2026-10-23
```

The harness prompts for the hotspot password without putting it in the command line, firmware, or saved report. Both boards must join the same Wi-Fi channel; a software reboot requires the harness to reconnect them because the credentials are RAM-only. The hotspot supplies network association, while encrypted readings still travel directly between the boards over ESP-NOW. This test does not use the hotspot to generate a CSI key.

For a physical USB power-loss check, run the harness with `-PowerCycleCheck`. Wait for `POWER_CYCLE_BOB_NOW`, then unplug and reconnect Bob's USB cable while leaving Alice powered. The harness observes the disconnect, verifies Bob's persisted replay state after reconnection, and resends the previously accepted frame; Bob must reject it. This checks a real power cut after a completed NVS write; interruption during a write remains untested.

The first and repeat hardware runs each accepted 120 readings plus one after Alice reboot, rejected two replay attempts and one altered tag, and accepted no duplicate `(boot, sequence)` pair. Their reports are in `captures/esp32/sensor-link-first-2026-10-23/` and `captures/esp32/sensor-link-repeat-2026-10-23/`.

The physical-cycle run observed Bob's COM7 disconnect and reconnection, restored its stored high-water mark, rejected the old frame, accepted 122 unique readings in total, rejected three replays and one altered tag, and accepted no duplicate. See `captures/esp32/sensor-link-physical-cycle-2026-10-23/summary.json`.

After the CSI diagnostic captures, both boards were reflashed with the encrypted-reading sketch without reprovisioning. The restoration run accepted 121 unique readings, rejected two replays and one altered tag, and accepted no duplicate. See `captures/esp32/sensor-link-restored-2026-10-23/summary.json`.

The hotspot run associated Alice and Bob with `BlackMetal` on channel 11 and assigned distinct local IP addresses. It accepted 121 readings, rejected two replays and one altered tag, and accepted no duplicate. Both boards rejoined after software reboot. See `captures/esp32/sensor-link-hotspot-2026-10-23/summary.json`. This does not establish CSI reciprocity or fresh physical key generation.

This is a bench prototype. The provisioned secret is stored in NVS without device flash encryption or secure boot, and USB access is trusted. Writing Bob's replay state for every accepted message suits this bounded low-rate test but needs flash-wear and interrupted-write testing before deployment. The CSI-based mode is not connected to this radio sketch and remains disabled for the fixed-node condition.
