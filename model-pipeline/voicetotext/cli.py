"""`voicetotext` command line: build the model files the app ships and measure them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from voicetotext.config import Config, _deep_merge, parse_override

app = typer.Typer(
    help="Model pipeline of the Greek voice-to-text app: acoustic export, n-gram LM, evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Datasets and manifests.", no_args_is_help=True)
app.add_typer(data_app, name="data")
lm_app = typer.Typer(help="n-gram language model.", no_args_is_help=True)
app.add_typer(lm_app, name="lm")
export_app = typer.Typer(help="Files for the phone.", no_args_is_help=True)
app.add_typer(export_app, name="export")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config path (default: configs/default.yaml).")
SetOpt = typer.Option(None, "--set", "-s", help="Override, e.g. decoder.kind=greedy. Repeatable.")


def _cfg(config: Optional[Path], sets: Optional[list[str]]) -> Config:
    overrides: dict = {}
    for s in sets or []:
        overrides = _deep_merge(overrides, parse_override(s))
    return Config.load(config, overrides)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="INFO logging.")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------- recognition


@app.command()
def transcribe(
    audio: list[Path] = typer.Argument(..., exists=True, readable=True, help="Audio file(s)."),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
    json_out: bool = typer.Option(False, "--json", help="Print the full Recognition as JSON."),
) -> None:
    """Transcribe one or more audio files."""
    from voicetotext.pipeline import Pipeline

    pipe = Pipeline(_cfg(config, set_))
    for p in audio:
        rec = pipe.recognize(p)
        if json_out:
            typer.echo(rec.to_json())
        else:
            typer.echo(f"{p.name}\t[{rec.final_source}]\t{rec.final_text}")
            if rec.greedy != rec.final_text:
                typer.echo(f"\tgreedy:\t{rec.greedy}")


@app.command()
def compare(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Audio file."),
    config: Optional[Path] = typer.Option("configs/omni_beam.yaml", "--config", "-c", help="Config with a beam decoder + LM."),
    set_: Optional[list[str]] = SetOpt,
    n_best: int = typer.Option(8, "--n-best", help="How many hypotheses to print."),
) -> None:
    """Show the acoustic layer and the LM layer side by side on one file.

    Prints the greedy (LM-free) view, the beam+LM result, the word-level changes the LM
    made, and the N-best list with acoustic and LM scores separated.
    """
    from voicetotext.acoustic.emissions import greedy_decode, greedy_words
    from voicetotext.audio.io import load_audio
    from voicetotext.pipeline import Pipeline

    cfg = _cfg(config, set_)
    if cfg.decoder.kind != "beam":
        typer.echo("note: decoder.kind is not 'beam'; pass -c configs/omni_beam.yaml to see the LM layer\n")
    pipe = Pipeline(cfg)
    wav, _ = load_audio(audio, cfg.audio.sample_rate)
    em = pipe.emitter.emit([wav], [audio.stem])[0]

    greedy = greedy_decode(em)
    hyps = pipe.beam.decode(em) if cfg.decoder.kind == "beam" else []
    best = hyps[0].text if hyps else greedy

    typer.echo(f"file    : {audio.name}  ({em.duration_s:.2f}s, {em.num_frames} frames of {em.frame_duration_ms:.0f}ms)")
    typer.echo(f"model   : {em.model_id}")
    typer.echo(f"LM      : {cfg.decoder.lm_path or '(none)'}  alpha={cfg.decoder.alpha} beta={cfg.decoder.beta} width={cfg.decoder.beam_width}")
    typer.echo("")
    typer.echo(f"greedy  : {greedy}")
    typer.echo(f"beam+LM : {best}")

    gw = [w.text for w in greedy_words(em)]
    bw = best.split()
    if gw == bw:
        typer.echo("\nthe LM changed nothing: the acoustic posteriors already agree with it")
    else:
        typer.echo("\nchanges made by the LM:")
        for i in range(max(len(gw), len(bw))):
            a = gw[i] if i < len(gw) else ""
            b = bw[i] if i < len(bw) else ""
            if a != b:
                typer.echo(f"  {a or '(inserted)':<22} -> {b or '(dropped)'}")

    if hyps:
        typer.echo(f"\nN-best ({len(hyps)} kept, showing {min(n_best, len(hyps))}); total = acoustic + LM:")
        typer.echo(f"  {'#':>2}  {'acoustic':>9}  {'LM':>9}  {'total':>9}  text")
        for i, h in enumerate(hyps[:n_best], 1):
            typer.echo(f"  {i:>2}  {h.acoustic_score:9.2f}  {h.lm_score:9.2f}  {h.total_score:9.2f}  {h.text}")


@app.command()
def labels(
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
    save: Optional[Path] = typer.Option(None, "--save", help="Write the ordered label list as JSON."),
) -> None:
    """Inspect the label set the configured model emits (the columns of the emission matrix)."""
    from voicetotext.decoding.vocab import inspect_labels, save_labels
    from voicetotext.pipeline import Pipeline

    cfg = _cfg(config, set_)
    labs = Pipeline(cfg).labels
    typer.echo(f"model: {cfg.model.model_id}  ({cfg.model.onnx_path})")
    typer.echo(inspect_labels(labs).format())
    if save:
        save_labels(labs, save)
        typer.echo(f"saved labels -> {save}")


# ---------------------------------------------------------------------- evaluation


@app.command("eval")
def eval_cmd(
    manifest: Path = typer.Argument(..., exists=True, help="JSONL manifest (audio_path, reference, ...)."),
    run_name: str = typer.Option(..., "--run-name", "-n", help="Results are written to <results_dir>/<run_name>/."),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N entries."),
    note: str = typer.Option("", "--note", help="Free text for the results row."),
    results_md: Optional[Path] = typer.Option(None, "--results-md", help="Append the results row to this Markdown file."),
    cache_emissions: bool = typer.Option(False, "--cache-emissions", help="Reuse/store emissions in <emissions_cache>."),
) -> None:
    """Run a manifest through the pipeline; report WER/CER with breakdowns."""
    from voicetotext.eval.harness import format_summary, results_row, run_eval

    cfg = _cfg(config, set_)
    report = run_eval(cfg, manifest, run_name, limit=limit, use_emissions_cache=cache_emissions)
    typer.echo(format_summary(report))
    row = results_row(report, note)
    typer.echo("\nresults row:\n" + row)
    if results_md:
        with open(results_md, "a", encoding="utf-8") as f:
            f.write(row + "\n")
    typer.echo(f"\nwritten: {cfg.paths.results / run_name}")


@app.command("tune")
def tune_cmd(
    manifest: Path = typer.Argument(..., exists=True, help="DEV manifest (never a test split)."),
    alphas: str = typer.Option("0.3,0.5,0.7,0.9,1.1,1.3", "--alphas"),
    betas: str = typer.Option("0.0,0.5,1.0,1.5,2.0,3.0", "--betas"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    metric: str = typer.Option("cer", "--metric", help="cer | wer"),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """Grid-search alpha/beta on cached emissions of a dev manifest; prints a table and the best."""
    import json

    from voicetotext.eval.tune import tune_alpha_beta

    cfg = _cfg(config, set_)
    res = tune_alpha_beta(cfg, manifest, [float(a) for a in alphas.split(",")], [float(b) for b in betas.split(",")], limit=limit, metric=metric)
    typer.echo(res["table"])
    typer.echo(f"best ({metric}): alpha={res['best']['alpha']} beta={res['best']['beta']} wer={res['best']['wer']:.4f} cer={res['best']['cer']:.4f}")
    out = cfg.paths.results / f"tune_{manifest.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=2), encoding="utf-8")
    typer.echo(f"written: {out}")


# ---------------------------------------------------------------------- data


@data_app.command("fleurs")
def data_fleurs(
    split: list[str] = typer.Option(["dev", "test"], "--split", help="FLEURS split(s)."),
    limit: int = typer.Option(100, "--limit", help="Clips per split (seeded sample)."),
    seed: int = typer.Option(0, "--seed"),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """Download FLEURS el_gr clips and write manifests."""
    from voicetotext.eval.datasets import build_fleurs_manifest

    cfg = _cfg(config, set_)
    for s in split:
        p = build_fleurs_manifest(cfg.paths.data, cfg.paths.manifests, s, limit=limit, seed=seed)
        typer.echo(f"manifest: {p}")


@data_app.command("common-voice")
def data_common_voice(
    cv_dir: Path = typer.Argument(..., exists=True, help="Common Voice language folder, e.g. cv-corpus-17.0/el"),
    split: str = typer.Option("test", "--split"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Seeded sample size (default: whole split)."),
    seed: int = typer.Option(0, "--seed"),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """Build a manifest from a locally downloaded Common Voice extract (decodes clips to 16 kHz wav)."""
    from voicetotext.eval.datasets import build_common_voice_manifest

    cfg = _cfg(config, set_)
    p = build_common_voice_manifest(cv_dir, cfg.paths.data, cfg.paths.manifests, split, limit=limit, seed=seed, sample_rate=cfg.audio.sample_rate)
    typer.echo(f"manifest: {p}")


@data_app.command("clean")
def data_clean(
    manifest: Path = typer.Argument(..., exists=True),
) -> None:
    """Write <manifest>_clean.jsonl: entries whose reference has no digits and no Latin letters."""
    import re

    from voicetotext.eval.harness import load_manifest, write_manifest

    entries = load_manifest(manifest)
    bad = re.compile(r"[0-9A-Za-z]")
    keep = [e for e in entries if not bad.search(e.reference)]
    out = manifest.with_name(manifest.stem + "_clean.jsonl")
    write_manifest(keep, out)
    typer.echo(f"{len(keep)}/{len(entries)} kept -> {out}")


# ---------------------------------------------------------------------- language model


@lm_app.command("prepare")
def lm_prepare(
    opensubtitles: Optional[Path] = typer.Option(None, "--opensubtitles", help="OPUS OpenSubtitles mono el (.txt or .txt.gz)."),
    wikipedia: Optional[Path] = typer.Option(None, "--wikipedia", help="wikimedia/wikipedia parquet shard (el)."),
    wiki_max_articles: Optional[int] = typer.Option(None, "--wiki-max-articles"),
    name: str = typer.Option("general_el", "--name"),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """Corpus -> cleaned, normalized training text (<lm_dir>/<name>.txt) + cleaning statistics."""
    import json

    from voicetotext.decoding.lm import prepare_corpus
    from voicetotext.pipeline import Pipeline

    cfg = _cfg(config, set_)
    sources = {k: v for k, v in {"opensubtitles": opensubtitles, "wikipedia": wikipedia}.items() if v}
    norm = Pipeline(cfg).normalizer
    cfg.paths.lm.mkdir(parents=True, exist_ok=True)
    stats = prepare_corpus(cfg, norm, sources, cfg.paths.lm / f"{name}.txt", wiki_max_articles)
    (cfg.paths.lm / f"{name}.prepare.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@lm_app.command("oov")
def lm_oov(
    manifests: list[Path] = typer.Argument(..., exists=True, help="Manifests whose references are checked."),
    sizes: str = typer.Option("70000,150000,300000,500000", "--sizes"),
    name: str = typer.Option("general_el", "--name"),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """OOV rate of top-N vocabularies (from the prepared text) on manifest references."""
    import json

    from voicetotext.decoding.lm import oov_report, word_counts_of
    from voicetotext.eval.harness import load_manifest
    from voicetotext.pipeline import Pipeline

    cfg = _cfg(config, set_)
    norm = Pipeline(cfg).normalizer
    counts = word_counts_of(cfg.paths.lm / f"{name}.txt")
    refs = {m.stem: [norm(e.reference) for e in load_manifest(m)] for m in manifests}
    rows = oov_report(counts, [int(x) for x in sizes.split(",")], refs)
    keys = [k for k in rows[0] if k not in ("types_total",)]
    typer.echo(f"corpus types: {rows[0]['types_total']}")
    typer.echo("| " + " | ".join(keys) + " |")
    typer.echo("|" + "---:|" * len(keys))
    for r in rows:
        typer.echo("| " + " | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    (cfg.paths.lm / f"{name}.oov.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


@lm_app.command("build")
def lm_build(
    opensubtitles: Optional[Path] = typer.Option(None, "--opensubtitles", help="OPUS OpenSubtitles mono el (.txt or .txt.gz)."),
    wikipedia: Optional[Path] = typer.Option(None, "--wikipedia", help="wikimedia/wikipedia parquet shard (el)."),
    wiki_max_articles: Optional[int] = typer.Option(None, "--wiki-max-articles"),
    name: str = typer.Option("general_el", "--name"),
    dev_manifest: Optional[Path] = typer.Option(None, "--dev-manifest", help="Report perplexity/OOV on these references."),
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
) -> None:
    """Build the n-gram LM (corpus -> normalize -> vocab cap -> lmplz ARPA -> KenLM binary)."""
    import json

    from voicetotext.decoding.lm import build_general_lm, load_lm, oov_rate, perplexity
    from voicetotext.pipeline import Pipeline

    cfg = _cfg(config, set_)
    sources = {}
    if opensubtitles:
        sources["opensubtitles"] = opensubtitles
    if wikipedia:
        sources["wikipedia"] = wikipedia
    if not sources:
        raise typer.BadParameter("give at least one corpus")
    norm = Pipeline(cfg).normalizer
    report = build_general_lm(cfg, norm, sources, name=name, wiki_max_articles=wiki_max_articles)
    if dev_manifest:
        from voicetotext.eval.harness import load_manifest

        refs = [norm(e.reference) for e in load_manifest(dev_manifest)]
        model = load_lm(report["bin_path"])
        vocab = set((cfg.paths.lm / f"{name}.vocab").read_text(encoding="utf-8").split())
        report["dev_perplexity"] = perplexity(model, refs)
        report["dev_oov_rate"] = oov_rate(vocab, refs)
    (cfg.paths.lm / f"{name}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@lm_app.command("export-phone")
def lm_export_phone(
    arpa: Path = typer.Argument(..., exists=True, help="ARPA written by `lm build` (order <= 3)."),
    out: Path = typer.Argument(..., help="Output file, e.g. el_3gram.gvtlm."),
) -> None:
    """Write the ARPA as the sorted-array format the app memory-maps (NgramLm.kt)."""
    import json

    from voicetotext.decoding.export_lm import export

    typer.echo(json.dumps(export(arpa, out), ensure_ascii=False, indent=2))


@app.command("homophones")
def homophones_cmd(
    out: Path = typer.Argument(..., help="Output file (the app expects el_homophones.bin)."),
    vocab: Path = typer.Option(..., "--vocab", exists=True, help="One word per line, the LM vocabulary."),
    kenlm: Optional[Path] = typer.Option(None, "--kenlm", exists=True, help="KenLM binary (the release route)."),
    arpa: Optional[Path] = typer.Option(None, "--arpa", exists=True, help="ARPA file, no kenlm module needed."),
) -> None:
    """Build the sound-key -> spellings index (layer 2b, HomophoneIndex.kt)."""
    from voicetotext.phonetics.homophones import build

    if (kenlm is None) == (arpa is None):
        raise typer.BadParameter("give exactly one of --kenlm or --arpa")
    report = build(vocab, out, kenlm=kenlm, arpa=arpa)
    typer.echo(f"{report['keys']} sound keys, {report['ambiguous']} with more than one spelling -> {out}")


# ---------------------------------------------------------------------- acoustic model


@export_app.command("omni-phone")
def export_omni_phone(
    src: Path = typer.Argument(..., exists=True, help="sherpa-onnx Omnilingual CTC 300M ONNX file."),
    out: Path = typer.Argument(..., help="Output, e.g. omni.onnx (omni.labels.json is written next to it)."),
    tokens: Optional[Path] = typer.Option(None, "--tokens", exists=True, help="sherpa tokens.txt (default: next to the ONNX file)."),
    foreign: str = typer.Option("letters", "--foreign", help="letters | drop: what happens to mass on non-Greek symbols."),
    verify: list[Path] = typer.Option([], "--verify", help="Audio files to check the phone graph against the Python emitter."),
) -> None:
    """Bake the Greek-column restriction into the ONNX graph for the app (omni.onnx + omni.labels.json)."""
    import numpy as np

    from voicetotext.acoustic.emissions import greedy_decode
    from voicetotext.acoustic.omni import OmniOnnxEmitter, export_phone_graph
    from voicetotext.audio.io import load_audio
    from voicetotext.config import ModelConfig

    dst = export_phone_graph(src, out, foreign, tokens)
    typer.echo(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB) and {dst.with_suffix('.labels.json').name}")
    if not verify:
        return
    reference = OmniOnnxEmitter(ModelConfig(onnx_path=str(src), omni_foreign=foreign))  # type: ignore[arg-type]
    phone = OmniOnnxEmitter(ModelConfig(onnx_path=str(dst)))
    for path in verify:
        wav, _ = load_audio(path)
        a = reference.emit([wav], [path.stem])[0]
        b = phone.emit([wav], [path.stem])[0]
        diff = float(np.abs(a.logprobs - b.logprobs).max())
        typer.echo(f"  {path.name}: max |dlogp| {diff:.2e}  same greedy text: {greedy_decode(a) == greedy_decode(b)}")


@export_app.command("gpt2")
def export_gpt2_cmd(
    model: str = typer.Argument(..., help="Hugging Face model id, e.g. lighteternal/gpt2-finetuned-greek."),
    out_dir: Path = typer.Argument(..., help="Writes el_gpt2.int8.onnx, vocab.json and merges.txt here."),
) -> None:
    """Export the Greek GPT-2 rescorer for the app (layer 2c)."""
    from voicetotext.decoding.gpt2_export import export

    typer.echo(export(model, out_dir))


@app.command("gpt2-probe")
def gpt2_probe(
    onnx: Path = typer.Argument(..., exists=True, help="el_gpt2.int8.onnx."),
    tokenizer: str = typer.Argument(..., help="Hugging Face id or a directory with the tokenizer."),
    sentence: str = typer.Argument(..., help="Sentence to score. The app probes one fixed sentence at load."),
) -> None:
    """Print a sentence's GPT-2 token ids and log-probability, to check the app's tokenizer."""
    from voicetotext.decoding.gpt2_rescore import Gpt2Scorer

    s = Gpt2Scorer(str(onnx), tokenizer)
    typer.echo("ids " + ",".join(map(str, s.ids(sentence))))
    typer.echo(f"logp {s.score([sentence])[0]}")


if __name__ == "__main__":
    app()
