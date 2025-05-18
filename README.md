
# OpenCL/CPU Hash Brute-Forcer

**Installation:**
```bash
pip install pyopencl numpy
```

**Important Prerequisites:**
For `pyopencl` to function correctly (especially for GPU acceleration), you must have an OpenCL runtime/SDK and appropriate drivers installed for your GPU or CPU device.
-   **NVIDIA:** Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (includes OpenCL drivers).
-   **AMD:** Install the [AMD ROCm platform](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) (for Linux) or Adrenalin Software (for Windows).
-   **Intel:** Install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) (includes OpenCL CPU/GPU runtimes).

---

## Description

This Python script is a versatile hash brute-forcing tool designed to find the original string corresponding to a given cryptographic hash. It utilizes OpenCL for GPU acceleration on shorter candidate strings in "charset" mode and falls back to CPU for longer strings, direct wordlist attacks, and rule-based mutation attacks. It can achieve speeds around 1M hashes/s in my tests using gpu (RX580), use with caution as it uses alot of gpu for max processing.

## Features

-   **Supported Hash Algorithms:** MD5, SHA1, SHA256.
-   **Dual Generation Modes:**
    -   **Charset Mode:** Generates candidates from a specified character set and length range.
        -   Uses OpenCL (GPU preferred) for lengths up to `KERNEL_MAX_STRING_LENGTH` (default 55).
        -   Falls back to CPU for lengths exceeding this limit or if OpenCL is unavailable.
    -   **Rules Mode:** Applies common mutations (case changes, leetspeak, suffix additions) to words from a provided wordlist (CPU-based).
-   **Direct Wordlist Attack:** Initially checks a provided wordlist for a direct match before attempting more complex generation modes (CPU-based).
-   **OpenCL Acceleration:** Leverages available OpenCL devices (GPUs or CPUs) for significantly faster hashing in charset mode for supported lengths.
-   **CPU Fallback & Operation:** Handles scenarios where OpenCL is not viable and powers wordlist/rules modes.
-   **Configurable Parameters:**
    -   Hash algorithm selection.
    -   Target hash input.
    -   Custom charset and length ranges for charset mode.
    -   Wordlist input for direct and rules-based attacks.
    -   Preferred OpenCL device type (GPU, CPU, any).
    -   GPU batch size tuning for performance optimization.
-   **Real-time Progress:** Displays attempts, hashing speed (Hashes Per Second - H/s), and percentage completed where applicable.
-   **Leetspeak Transformations:** Applies basic leetspeak substitutions in "rules" mode (e.g., `a` -> `@`, `e` -> `3`).

## Usage

The script is executed from the command line. The primary required argument is the target hash.

```bash
python brute.py -H <TARGET_HASH> [OPTIONS]
```

## Command-Line Arguments

-   `-a ALGO`, `--algo ALGO`:
    -   Description: Hash algorithm to use.
    -   Choices: `md5`, `sha1`, `sha256`
    -   Default: `sha256`
-   `-H HASH`, `--hash HASH`:
    -   Description: **(Required)** The target hash hex string to crack.
-   `-w WORDLIST`, `--wordlist WORDLIST`:
    -   Description: Path to a wordlist `.txt` file. This list is first checked for a direct match. It is **required** if `--gen-mode` is `rules`.
    -   Default: `None`
-   `--gen-mode {charset,rules}`:
    -   Description: Specifies the primary generation mode after the initial wordlist check (if any).
        -   `charset`: Brute-force using a character set (GPU/CPU).
        -   `rules`: Apply mutation rules to a wordlist (CPU).
    -   Default: `charset`
-   `--device-type {gpu,cpu,any}`:
    -   Description: Preferred OpenCL device type for the GPU-accelerated part of 'charset' mode.
    -   Default: `gpu`

### Charset Mode Options (these are applicable if `--gen-mode charset` is selected or default)

-   `-c CHARSET`, `--charset CHARSET`:
    -   Description: Character set to use for 'charset' mode generation.
    -   Default: `abcdefghijklmnopqrstuvwxyz0123456789`
-   `--min-len MIN_LEN`:
    -   Description: Minimum string length for 'charset' mode.
    -   Default: `1`
-   `--max-len MAX_LEN`:
    -   Description: Maximum string length for 'charset' mode.
    -   Default: GPU up to `KERNEL_MAX_STRING_LENGTH` (55), then CPU is limited by a soft cap (approx. `KERNEL_MAX_STRING_LENGTH + 10`). Set explicitly for longer CPU attempts.
-   `--batch-mult BATCH_MULT`:
    -   Description: GPU batch size multiplier for 'charset' mode. Affects how many candidate hashes are processed by the GPU in a single batch.
    -   Default: `8192`

## Examples

1.  **Basic SHA256 charset brute-force (default: lowercase alphanumeric, lengths 1 up to GPU/CPU limit):**
    ```bash
    python brute.py -H 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
    ```

2.  **MD5 charset brute-force for a specific length (digits only):**
    ```bash
    python brute.py -a md5 -H e10adc3949ba59abbe56e057f20f883e --charset "0123456789" --min-len 6 --max-len 6
    ```

3.  **Charset brute-force with a custom charset, min/max length, and forcing CPU OpenCL device:**
    ```bash
    python brute.py -H <YOUR_TARGET_HASH> --algo sha1 --charset "abc123!@" --min-len 4 --max-len 5 --device-type cpu
    ```

4.  **Using a wordlist for direct match, then applying rules-based mutations (CPU only for rules part):**
    ```bash
    python brute.py -a sha1 -H <YOUR_SHA1_HASH> --wordlist /path/to/common_passwords.txt --gen-mode rules
    ```
    *(This will first check `common_passwords.txt` directly. If no match, it applies mutations to words from that list.)*

5.  **Using a wordlist for direct match, and if not found, fallback to charset generation (GPU/CPU):**
    ```bash
    python brute.py -a sha256 -H <YOUR_SHA256_HASH> --wordlist /path/to/my_list.txt --gen-mode charset --charset "abcdef" --min-len 3 --max-len 5
    ```
    *(The wordlist is checked first. If no match, `charset` mode proceeds with the specified charset and length.)*

## Modes of Operation Explained

The script operates in a sequence:

1.  **Direct Wordlist Check (CPU):**
    -   If a `--wordlist` is provided, the script first iterates through this file.
    -   Each word is hashed (using the specified algorithm) and compared against the target hash.
    -   This step is always performed on the CPU. If a match is found, the script exits.

2.  **Generation Mode (`--gen-mode`):**
    If no match is found in the direct wordlist check (or if no wordlist was provided), the script proceeds based on the `--gen-mode` argument.

    *   **`charset` (Default Mode):**
        -   Generates candidate strings based on the `--charset`, `--min-len`, and `--max-len` parameters.
        -   **GPU (OpenCL):** For candidate lengths up to `KERNEL_MAX_STRING_LENGTH` (currently 55), the script attempts to use an OpenCL-capable device (GPU preferred by default via `--device-type gpu`). This offers significant speed advantages.
        -   **CPU:**
            -   If OpenCL initialization fails or no suitable OpenCL device is found.
            -   For candidate lengths greater than `KERNEL_MAX_STRING_LENGTH`.
            -   If `--device-type cpu` is specified and an OpenCL CPU runtime is available.
        -   The script also checks for an empty string (length 0) if `min_len <= 0`.

    *   **`rules` Mode:**
        -   This mode **requires** a `--wordlist` to be specified.
        -   It operates entirely on the **CPU**.
        -   For each `base_word` from the provided wordlist, it applies a series of mutations to generate new candidates:
            -   **Case Variations:** original, lowercase, uppercase, capitalized.
            -   **Leetspeak:** Common substitutions (e.g., `a` -> `@`, `e` -> `3`, `o` -> `0`, `s` -> `S`).
            -   **Suffixes:** Appends common suffixes (e.g., `1`, `123`, `!`, `2023`, `2024`).
        -   Each mutated word is then hashed and compared against the target hash.

## Important Notes

-   **`KERNEL_MAX_STRING_LENGTH`:** The OpenCL kernel used for GPU acceleration has an internal limit on the maximum string length it can efficiently process. This is currently set to `55` characters. For `charset` generation of strings longer than this, the script automatically switches to a slower, CPU-based method.
-   **Performance:**
    -   GPU-accelerated hashing (in `charset` mode for lengths <= 55) is typically orders of magnitude faster than CPU-based hashing.
    -   The `rules` mode and direct wordlist checks are CPU-bound and their speed depends on CPU performance and wordlist size/complexity of rules.
-   **Computational Complexity:** Brute-forcing cryptographic hashes is computationally intensive. The time required to find a match increases exponentially with the length of the unknown string and the size of the character set. Cracking long, complex passwords with large charsets can be impractical.
-   **Ethical Considerations:** This tool is intended for educational purposes, security testing, and password recovery **only where you have explicit authorization**. Unauthorized use against systems or data you do not own is illegal and unethical.
```
