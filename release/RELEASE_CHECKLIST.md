# Release Checklist

This checklist follows the [ECCV 2026 artifact guidance](https://eccv.ecva.net/Conferences/2026/AuthorPractices),
the [Hugging Face model release checklist](https://huggingface.co/docs/hub/en/model-release-checklist), and
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Automated locally

- [x] Pin the six adapters to exact base-model revisions.
- [x] Publish machine-readable three-seed metrics and seed-42 raw predictions.
- [x] Verify the 7,929/5,349 split using row-level SHA-256 hashes.
- [x] Package adapters as `safetensors` with model cards, configs, manifests, and checksums.
- [x] Add CI for source compilation, release-artifact tests, lint, and the Docker source stage.
- [x] Document upstream model and dataset licensing constraints.
- [x] Attach all six adapter archives and an archive checksum file to the GitHub Release.

## Maintainer-authenticated publication

Run these only from a clean checkout of the release commit:

```bash
# Upload six prepared model directories after `hf auth login`.
python scripts/upload_hf_adapters.py \
  --artifact-root ../hf-models \
  --registry release/model_registry.json

# Publish source and immutable tag after GitHub authentication.
git push origin main
git push origin v2.0.0-eccv2026
```

- [ ] Confirm all six Hugging Face repositories are public and each checksum verifies.
- [ ] Confirm Gemma's gated-base-model notice is visible on both Gemma adapter cards.
- [x] Create a GitHub Release from `v2.0.0-eccv2026` and attach the source archive.
- [ ] Enable the GitHub-Zenodo integration, create the release, then add the minted DOI to `CITATION.cff` and README.
- [x] Run the GitHub Actions workflow on the public tag and retain the successful run URL.
- [ ] Perform one clean-machine evaluation with `./reproduce.sh evaluate` and archive the generated `metrics.json`.

Do not publish dataset images or annotations unless their redistribution license has been verified. The row-level
manifest is sufficient to validate the paper split without placing the annotations in this repository.
