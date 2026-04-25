# warpkit @VERSION@ — standalone binaries (@TARGET@)

This bundle contains the seven `wk-*` CLIs as standalone binaries. No Python
install or system ITK required — everything is in `_internal/`.

## Install

Extract anywhere and put the bundle directory on your `PATH`:

```sh
# example: install to /opt
sudo mv warpkit-@VERSION@ /opt/
echo 'export PATH=/opt/warpkit-@VERSION@:$PATH' >> ~/.bashrc
```

Or symlink each `wk-*` onto an existing `PATH` entry — the bootloader resolves
`_internal/` relative to the real binary, so symlinks work fine:

```sh
for bin in /opt/warpkit-@VERSION@/wk-*; do
    sudo ln -s "$bin" /usr/local/bin/$(basename "$bin")
done
```

**Do not separate the binaries from `_internal/`** — they all share the
embedded interpreter and dependency tree.

## macOS Gatekeeper

Binaries are ad-hoc signed, not Apple-notarized, so on first launch macOS will
quarantine them — and not just the top-level `wk-*` binaries: every `.dylib`
and `.so` inside `_internal/` is also flagged. Strip the quarantine attribute
recursively from the whole bundle:

```sh
xattr -r -d com.apple.quarantine /opt/warpkit-@VERSION@
```

## Available CLIs

- `wk-medic` — full MEDIC distortion correction pipeline
- `wk-unwrap-phase` — ROMEO multi-echo phase unwrapping
- `wk-compute-fieldmap` — compute B0 field map from unwrapped phase
- `wk-apply-warp` — apply a displacement field to an image
- `wk-convert-warp` — convert between warp field conventions
- `wk-convert-fieldmap` — convert between field-map representations
- `wk-compute-jacobian` — compute the Jacobian determinant of a warp

Run any of them with `--help` for usage.
