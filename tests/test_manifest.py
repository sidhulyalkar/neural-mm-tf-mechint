from neural_mechint.manifest import ExperimentManifest, hash_prompts


def test_prompt_hash_is_order_sensitive_and_stable():
    assert hash_prompts(["a", "b"]) == hash_prompts(["a", "b"])
    assert hash_prompts(["a", "b"]) != hash_prompts(["b", "a"])


def test_manifest_serializes(tmp_path):
    manifest = ExperimentManifest(
        experiment="test",
        model="toy",
        metric="delta_logit",
        seed=0,
        prompt_hash=hash_prompts(["hello"]),
    )
    path = manifest.save(tmp_path / "manifest.json")
    assert path.exists()
    assert '"experiment": "test"' in path.read_text()
