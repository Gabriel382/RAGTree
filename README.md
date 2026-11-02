# KBS GPT-OSS-20B bounded fallback — compatibility installer

This is a replacement for `git apply kbs_bounded_fallback.patch` **when the patch fails at `gateway.py:39`**. The original `.patch` expects one exact intermediate revision of the KBS gateway and cannot be applied reliably to different Windows/Linux checkouts.

## Purpose

- Only in `controlled_revision_v2`, with the GPT-OSS-20B configured generation policy:
  - Initial `max_tokens=4096`, `reasoning_effort=low`.
  - If the final answer is empty and the response ends with `finish_reason=length`, retry that call once with `max_tokens=8192`.
  - Log both requests and token usage separately. No infinite fallback loop.
- Preserve the original RAG strategy algorithms, historical protocol, and datasets.
- Handle different earlier KBS gateway revisions (original, diagnostic hotfix, or already patched).
- Refuse unknown code variants instead of forcing a patch. Create a separate `.before_kbs_fallback.bak` copy of every modified file.

## Windows PowerShell

Extract the whole ZIP, including `gateway_retry_block.txt`, to a folder, such as `C:\Users\henri\Downloads\KBS_Fallback_Compat`. Then from the RAGTree root:

```powershell
cd C:\Users\henri\Documents\git\postdoc\RAGTree

git branch --show-current
git status --short

# Dry-run: validate all four files, change nothing.
python C:\Users\henri\Downloads\KBS_Fallback_Compat\install_compatible.py --repo .

# Apply if and only if the above succeeds.
python C:\Users\henri\Downloads\KBS_Fallback_Compat\install_compatible.py --repo . --apply

# Optional: add the isolated fallback regression tests.
Copy-Item C:\Users\henri\Downloads\KBS_Fallback_Compat\test_generation_fallback.py tests\kbs\test_generation_fallback.py
python -m pytest -q tests/kbs

git diff --stat
```

The script does not delete any pre-existing results/checkpoints. Because source files and the protocol YAML are hashed into run IDs, the new protocol creates separate run identities and **may not reuse prior successful checkpoints**. Do not rerun the complete estimator solely because this patch has been installed.

### After validating and committing on Windows

Commit only the intended source/config/test files (do not commit `.before_kbs_fallback.bak` copies), push the correct branch, and `git pull` that branch on IMMOTEP **only if you want to synchronize the two machines**. You can alternatively run this compatibility installer directly against the IMMOTEP checkout, after testing and committing any existing work there.

## Linux

```bash
cd ~/git/postdoc/ragtree
python3 /path/to/KBS_Fallback_Compat/install_compatible.py --repo .
python3 /path/to/KBS_Fallback_Compat/install_compatible.py --repo . --apply
python3 -m pytest -q tests/kbs
```

## What was verified

- Compatibility dry-run/application/idempotence against three reconstructible KBS versions: base, pre-fallback, and already patched.
- Full KBS offline suite on a reconstructed full repository after installing this script: **21 passed, 4 skipped**.
- `verify_three_failed_docred.py` is included for targeted live testing, but **no live Windows inference** has been performed. It is only needed on the machine that has the original failed checkpoints and local LLM endpoint.

## Troubleshooting

If the installer says `NOT APPLIED: incompatible ...`, it has made no changes. Do not use `git apply --force`, delete your file, or overwrite it with a generic template. Share your current `gateway.py`, `engine.py`, and `protocol.py` (or their relevant diff) for a specifically targeted patch.
