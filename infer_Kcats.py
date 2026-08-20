from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

# if python version is <3.11 we just create a dummy logger:
if sys.version_info <= (3, 10):

    class CustomLogger:  # pragma: no cover - compatibility shim for old Python
        def __init__(self, _name: str, _log_dir: str, print_level: int = 2) -> None:
            self._print_level = print_level

        def set_print_level(self, print_level: int) -> None:
            self._print_level = print_level

        def set_log_files_location(self, _log_dir: str) -> None:
            return

        def _emit(self, prefix: str, message: str, print_level: int = 2) -> None:
            if self._print_level >= print_level:
                print(f"{prefix} - {message}")

        def info(self, message: str, print_level: int = 2, *args, **kwargs) -> None:
            self._emit("INFO", message, print_level=print_level)

        def warning(self, message: str, print_level: int = 2, *args, **kwargs) -> None:
            self._emit("WARNING", message, print_level=print_level)

        def error(self, message: str, print_level: int = 1, *args, **kwargs) -> None:
            self._emit("ERROR", message, print_level=print_level)

        def valid(self, message: str, print_level: int = 2, *args, **kwargs) -> None:
            self._emit("VALID", message, print_level=print_level)

        def starting(self, message: str, print_level: int = 2, *args, **kwargs) -> None:
            self._emit("STARTING", message, print_level=print_level)

        def finished(self, message: str, print_level: int = 2, *args, **kwargs) -> None:
            self._emit("FINISHED", message, print_level=print_level)

    def get_project_root() -> Path:
        return Path(__file__).resolve().parents[1]

else:
    from VmaxBuilder.utils.file_handling import get_project_root

    try:
        from src.VmaxBuilder.utils.custom_logging import CustomLogger
    except ModuleNotFoundError:
        from VmaxBuilder.utils.custom_logging import CustomLogger

try:
    from UniKP.compat_sklearn import safe_load_sklearn_model
except ModuleNotFoundError:
    from compat_sklearn import (  # ty: ignore[unresolved-import]
        safe_load_sklearn_model,
    )  # ty: ignore[unresolved-import] # noqa: I001


def _get_logger(logger: CustomLogger | None, print_level: int, log_dir: Path) -> CustomLogger:
    if logger is not None:
        logger.set_print_level(print_level)
        return logger
    return CustomLogger("infer_kcats", str(log_dir), print_level=print_level)


def _log_step(logger: CustomLogger, message: str, print_level: int = 2) -> None:
    logger.starting(message, print_level=print_level)


def _log_key_value(
    logger: CustomLogger, key: str, value: object, print_level: int = 2
) -> None:
    logger.info(f"{key}: {value}", print_level=print_level)


def _normalize_sequence_cache_key(sequence: str) -> str:
    """Normalize sequences to the representation actually embedded.

    Historically sequence embedding truncates sequences longer than 1000 aa to
    first 500 + last 500. Cache keys must use the same representation.
    """
    if len(sequence) > 1000:
        return sequence[:500] + sequence[-500:]
    return sequence


def _normalize_metabolite_id_for_matching(metabolite_id: str) -> str:
    """Best-effort normalization for metabolite id diagnostics."""
    normalized = metabolite_id.strip()
    normalized = re.sub(r"\[[^]]+]$", "", normalized)
    normalized = re.sub(r"_[A-Za-z0-9]+$", "", normalized)
    # Handle compact compartment suffixes like MAM01249c.
    if re.match(r"^MAM\d+[a-z]$", normalized):
        normalized = normalized[:-1]
    return normalized


def _compartment_format_variants(metabolite_id: str) -> list[str]:
    """Generate equivalent compartment-format IDs for direct lookup.

    Example: MAM01249c <-> MAM01249[c] <-> MAM01249_c
    """
    value = metabolite_id.strip()
    variants: list[str] = []

    bracket_match = re.match(r"^(.+)\[([A-Za-z0-9]+)]$", value)
    if bracket_match is not None:
        base, compartment = bracket_match.group(1), bracket_match.group(2)
        variants.extend([f"{base}{compartment}", f"{base}_{compartment}"])

    underscore_match = re.match(r"^(.+)_([A-Za-z0-9]+)$", value)
    if underscore_match is not None:
        base, compartment = underscore_match.group(1), underscore_match.group(2)
        variants.extend([f"{base}[{compartment}]", f"{base}{compartment}"])

    compact_match = re.match(r"^(MAM\d+)([a-z])$", value)
    if compact_match is not None:
        base, compartment = compact_match.group(1), compact_match.group(2)
        variants.extend([f"{base}[{compartment}]", f"{base}_{compartment}"])

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(variants))


# -----------------------------------------------------------------------------
# Embedding helpers (legacy names kept as wrappers for compatibility)
# -----------------------------------------------------------------------------


def smiles_to_embedding(
    smiles_list: list[str],
    logger: CustomLogger | None = None,
    print_level: int = 2,
    amount_of_replicates: int = 1,
    log_start: bool = True,
) -> np.ndarray | None:
    if logger is not None and log_start:
        logger.info(
            f"Building stochastic SMILES embeddings for {len(smiles_list)} SMILES "
            f"(replicates per SMILES: {amount_of_replicates})",
            print_level=print_level,
        )
    # Import lazily so package-mode imports work even if legacy modules
    # still use script-relative imports internally.
    try:
        from UniKP.build_vocab import WordVocab
        from UniKP.pretrain_trfm import TrfmSeq2seq
        from UniKP.utils import split
    except ModuleNotFoundError:
        from build_vocab import WordVocab  # ty: ignore[unresolved-import] # noqa: I001
        from pretrain_trfm import TrfmSeq2seq  # ty: ignore[unresolved-import] # noqa: I001
        from utils import split  # ty: ignore[unresolved-import] # noqa: I001

    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    try:
        vocab = WordVocab.load_vocab("vocab.pkl")
    except FileNotFoundError:
        current_dir = Path(__file__).resolve().parent
        models = current_dir
        vocab = WordVocab.load_vocab(str(models / "vocab.pkl"))

    truncation_count = 0

    def get_inputs(sm: str) -> tuple[list[int], list[int]]:
        nonlocal truncation_count
        seq_len = 220
        sm_tokens = sm.split()
        if len(sm_tokens) > 218:
            truncation_count += 1
            sm_tokens = sm_tokens[:109] + sm_tokens[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm_tokens]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def get_array(smiles_tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x_id, x_seg = [], []
        for sm in smiles_tokens:
            ids, seg = get_inputs(sm)
            x_id.append(ids)
            x_seg.append(seg)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    current_dir = Path(__file__).resolve().parent
    try:
        trfm.load_state_dict(
            torch.load("trfm_12_23000.pkl", map_location=torch.device("cpu"))
        )
    except FileNotFoundError:
        trfm.load_state_dict(
            torch.load(
                str(current_dir / "trfm_12_23000.pkl"), map_location=torch.device("cpu")
            )
        )
    trfm.eval()

    try:
        split_smiles = []
        for sm in smiles_list:
            if sm == "nan" or isinstance(sm, float):
                split_smiles.append("")
            else:
                split_smiles.append(split(sm))
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.error(f"SMILES split/encoding failed: {exc}", print_level=1)
        return None

    xid, _xseg = get_array(split_smiles)
    if logger is not None and truncation_count and print_level >= 4:
        logger.warning(
            f"Truncated {truncation_count} long SMILES in current batch",
            print_level=print_level,
        )

    # Legacy parity: build one expanded batch with repeated SMILES and encode once.
    # This matches how LEGACY_infer_Kcats generates stochastic replicate tensors.
    expanded_xid = xid.repeat_interleave(amount_of_replicates, dim=0)
    with torch.no_grad():
        encoded_expanded = np.asarray(trfm.encode(torch.t(expanded_xid)))
    encoded = encoded_expanded.reshape(len(smiles_list), amount_of_replicates, -1)
    if logger is not None and log_start:
        logger.valid("SMILES embedding generation complete", print_level=print_level)
    return encoded


def sequence_to_embedding(
    sequences: list[str],
    logger: CustomLogger | None = None,
    print_level: int = 2,
    use_tqdm: bool = True,
) -> np.ndarray:
    if logger is not None:
        logger.info(
            f"Building embeddings for {len(sequences)} unique sequences",
            print_level=print_level,
        )

    processed_sequences = [_normalize_sequence_cache_key(seq) for seq in sequences]

    sequence_examples = []
    for seq in processed_sequences:
        sequence_examples.append(" ".join(list(seq)))

    # get this files folder
    current_dir = Path(__file__).resolve().parent
    pretrained_model_dir = current_dir / "pretrained_models" / "prot_t5_xl_uniref50"
    str_pretrained_model_dir = str(pretrained_model_dir)

    tokenizer = T5Tokenizer.from_pretrained(
        str_pretrained_model_dir,
        do_lower_case=False,
    )
    model = T5EncoderModel.from_pretrained(str_pretrained_model_dir)

    gc.collect()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    features = []
    iterator = enumerate(sequence_examples)
    if use_tqdm:
        iterator = enumerate(
            tqdm(sequence_examples, desc="Embedding sequences", unit="seq", leave=False)
        )
    for idx, seq_text in iterator:
        cleaned = [seq_text]
        try:
            cleaned = [re.sub(r"[UZOB]", "X", seq_text)]
            ids = tokenizer.batch_encode_plus(cleaned, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_embed = embedding[seq_num][: seq_len - 1]
                features.append(seq_embed)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning(
                    f"Error for sequence index {idx}: {exc}",
                    print_level=print_level,
                )

    normalized = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                normalized[i][k] += features[i][j][k]
            normalized[i][k] /= len(features[i])
    if logger is not None:
        logger.valid("Sequence embedding generation complete", print_level=print_level)
    return normalized


# legacy wrappers
smiles_to_vec = smiles_to_embedding
Seq_to_vec = sequence_to_embedding


# -----------------------------------------------------------------------------
# Data preparation API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KcatPaths:
    output_dir: Path
    smiles_file: Path
    sequence_file: Path
    gene_metabolite_pairs_file: Path
    gene_metabolite_pairs_legacy_file: Path
    sequence_tensor_cache_file: Path
    smiles_tensor_cache_file: Path
    predictions_csv_file: Path
    predictions_json_file: Path
    missing_csv_file: Path


PREDICTION_COLUMNS = [
    "ensemble_id",
    "metabolite_id",
    "missing_smiles",
    "truncated_smiles",
    "min",
    "max",
    "median",
    "mean",
    "iqr",
    "sd",
    "sd_as_percent_of_mean",
]


def _resolve_output_dir(model_file: Path, kcat_root: Path | None = None) -> Path:
    model_name = model_file.parent.name
    if kcat_root is None:
        # expected model path: .../data/for_SWAMP/models/<model_name>/model_*.json
        base = model_file.parent.parent.parent
        kcat_root = base / "Kcat_predictions" / "UniKPV1"
    output_dir = kcat_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_kcat_paths(model_file: Path, kcat_root: Path | None = None) -> KcatPaths:
    output_dir = _resolve_output_dir(model_file=model_file, kcat_root=kcat_root)
    return KcatPaths(
        output_dir=output_dir,
        smiles_file=output_dir / "metabolite_smiles.csv",
        sequence_file=output_dir / "gene_or_transcript_protein_sequences.csv",
        gene_metabolite_pairs_file=output_dir / "gene_metabolite_pairs.json",
        gene_metabolite_pairs_legacy_file=output_dir / "gene_smiles_reactions_pairs.json",
        sequence_tensor_cache_file=output_dir / "sequence_embedding_cache.pkl",
        smiles_tensor_cache_file=output_dir / "smiles_embedding_cache.pkl",
        predictions_csv_file=output_dir / "kcat_gene_metabolite_predictions.csv",
        predictions_json_file=output_dir / "kcat_gene_metabolite_predictions.json",
        missing_csv_file=output_dir / "missing_genes_and_smiles.csv",
    )


def get_gene_metabolite_pairs(cobra_model) -> dict[str, list[str | list[str]]]:
    # check the reactions to see which gene interacts with which metabolites,
    # don't duplicate set, but do create reaction-association following :
    # {
    #     "0": ["ENSG00000198099", "MAM01249[c]", ["MAR03905"]],
    #     "1": ["ENSG00000198099", "MAM01796[c]", ["MAR03905"]],
    #     "2": [
    #         "ENSG00000198099",
    #         "MAM02039[c]",
    #         [
    #             "MAR03905",
    #             "MAR20008",
    #             "MAR20010",
    #         ],
    #     ],
    #     "3": [
    #         "ENSG00000198099",
    #         "MAM02552[c]",
    #         [
    #             "MAR03905",
    #             "MAR08530",
    #             "MAR20010",
    #         ],
    #     ],
    # }
    output_dictionary = {}
    pairs = set()
    for reaction in cobra_model.reactions:
        for gene in reaction.genes:
            for metabolite in reaction.metabolites:
                pair = (gene.id, metabolite.id)
                if pair not in pairs:
                    pairs.add(pair)
                    output_dictionary[str(len(pairs) - 1)] = [gene.id, metabolite.id, []]
                output_dictionary[str(len(pairs) - 1)][2].append(reaction.id)

    return output_dictionary


def write_gene_metabolite_pairs(
    cobra_model,
    output_file: Path,
    legacy_file: Path | None = None,
) -> dict[str, list[str | list[str]]]:
    payload = get_gene_metabolite_pairs(cobra_model)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if legacy_file is not None:
        legacy_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def get_gene_aa_sequences(
    cobra_model,
    *,
    level: str,
    species: str | None = None,
    provider: str = "auto",
    max_workers: int = 8,
    model_file: Path | None = None,
    gene_id_mapping_file: Path | None = None,
    use_mapping_as_primary: bool = False,
    logger: CustomLogger | None = None,
    print_level: int = 2,
) -> pd.DataFrame:
    """
    Generate protein sequence dataframe from SWaPAM sequence retrieval for any GEM.

    level='gene'       -> one sequence per gene (first canonical/available sequence)
    level='transcript' -> one row per returned sequence isoform
    """
    if level not in {"gene", "transcript"}:
        raise ValueError("level must be 'gene' or 'transcript'")

    gene_ids = sorted({gene.id for gene in cobra_model.genes})
    resolved_species = species or infer_species(gene_ids)
    if not resolved_species:
        raise ValueError("Could not infer species; pass species='human' or species='mouse'.")

    mode = SequenceMode.ALL_ISOFORMS if level == "transcript" else SequenceMode.CANONICAL_ONLY
    if logger is not None:
        logger.info(
            "Retrieving sequences "
            f"(level={level}, species={resolved_species}, provider={provider}, "
            f"max_workers={max_workers})",
            print_level=print_level,
        )
    if model_file is not None:
        results, _summary = retrieve_gem_sequences(
            model_json_path=model_file,
            species=resolved_species,
            mode=mode,
            provider=provider,
            max_workers=max_workers,
            gene_id_mapping_file=gene_id_mapping_file,
            use_mapping_as_primary=use_mapping_as_primary,
        )
    else:
        results, _summary = retrieve_gene_sequences(
            gene_ids,
            species=resolved_species,
            mode=mode,
            provider=provider,
            max_workers=max_workers,
        )
    if logger is not None:
        logger.info(
            "Sequence retrieval summary: "
            f"total_genes={_summary.total_genes}, "
            f"genes_with_sequences={_summary.genes_with_sequences}, "
            f"unresolved={len(_summary.unresolved_genes)}",
            print_level=print_level,
        )

    rows: list[dict[str, str | bool | None]] = []
    for gene_id in gene_ids:
        records = results.get(gene_id)
        if not records or not records.sequences:
            continue
        if level == "gene":
            first = records.sequences[0]
            rows.append(
                {
                    "sequence_level": "gene",
                    "ensemble_id": gene_id,
                    "transcript_or_accession": first.accession,
                    "source": first.source,
                    "is_canonical": first.is_canonical,
                    "protein_sequence": first.sequence,
                }
            )
        else:
            for entry in records.sequences:
                rows.append(
                    {
                        "sequence_level": "transcript",
                        "ensemble_id": gene_id,
                        "transcript_or_accession": entry.accession,
                        "source": entry.source,
                        "is_canonical": entry.is_canonical,
                        "protein_sequence": entry.sequence,
                    }
                )

    return pd.DataFrame(rows)


def get_metabolite_smiles(_cobra_model) -> pd.DataFrame:
    """
    Hook for future GEM-agnostic SMILES retrieval.

    Intentionally not implemented yet; caller can provide precomputed
    `metabolite_smiles.csv` with at least columns: ['id', 'isomeric SMILES'].
    """
    raise NotImplementedError(
        "SMILES retrieval hook is not implemented yet. "
        "Provide metabolite_smiles.csv manually."
    )


# -----------------------------------------------------------------------------
# Incremental cache + inference
# -----------------------------------------------------------------------------


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:  # noqa: BLE001
        try:
            return pd.read_csv(path, sep=";")
        except EmptyDataError:
            return pd.DataFrame()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        return {str(k): np.asarray(v) for k, v in data.items()}
    return {}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        pickle.dump(cache, handle)


def _load_smiles_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        return {}

    cache: dict[str, np.ndarray] = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        cache[str(key)] = array
    return cache


def _save_smiles_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        pickle.dump(cache, handle)


def _chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _update_embedding_cache(
    values: list[str],
    cache_path: Path,
    vectorizer: Callable[[list[str]], np.ndarray | None],
    key_normalizer: Callable[[str], str] | None = None,
    batch_size: int = 50,
    save_every_batches: int = 1,
    shared_cache_path: Path | None = None,
    logger: CustomLogger | None = None,
    print_level: int = 2,
) -> dict[str, np.ndarray]:
    # Load shared cache first as baseline; local run cache overrides it.
    shared_cache: dict[str, np.ndarray] = (
        _load_cache(shared_cache_path) if shared_cache_path is not None else {}
    )
    cache = _load_cache(cache_path)
    # Merge: shared provides base, local run cache takes precedence.
    merged: dict[str, np.ndarray] = {**shared_cache, **cache}

    normalizer = key_normalizer or (lambda v: v)
    normalized_values = list({normalizer(value) for value in values})
    needed = [value for value in normalized_values if value not in merged]
    shared_hits = sum(1 for v in normalized_values if v in shared_cache and v not in cache)
    local_hits = sum(1 for v in normalized_values if v in cache)
    if logger is not None:
        logger.info(
            f"Cache {cache_path.name}: cache_entries_total={len(merged)}, "
            f"request_unique={len(normalized_values)}, "
            f"local_hits={local_hits}, shared_hits={shared_hits}, "
            f"new_to_embed={len(needed)}",
            print_level=print_level,
        )
    if needed:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if save_every_batches < 1:
            raise ValueError("save_every_batches must be >= 1")

        needed_batches = _chunk_list(needed, batch_size)
        # do as tqdm
        for batch_idx, needed_batch in enumerate(
            tqdm(
                needed_batches,
                desc=f"Embedding {cache_path.name}",
                unit="batch",
                leave=False,
            ),
            start=1,
        ):
            new_vectors = vectorizer(needed_batch)
            if new_vectors is None:
                raise RuntimeError(f"Could not create embeddings for {cache_path.name}")
            for idx, key in enumerate(needed_batch):
                merged[key] = np.asarray(new_vectors[idx])

            if batch_idx % save_every_batches == 0 or batch_idx == len(needed_batches):
                # Write new entries to both local and shared cache.
                _save_cache(cache_path, merged)
                if shared_cache_path is not None:
                    shared_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_cache(shared_cache_path, merged)

            if logger is not None and print_level >= 3:
                logger.info(
                    f"{cache_path.name}: processed embedding batch {batch_idx}/"
                    f"{len(needed_batches)}",
                    print_level=print_level,
                )
    return merged


def _update_smiles_embedding_cache(
    values: list[str],
    cache_path: Path,
    amount_of_replicates: int,
    batch_size: int = 50,
    save_every_batches: int = 1,
    use_tqdm: bool = True,
    shared_cache_path: Path | None = None,
    logger: CustomLogger | None = None,
    print_level: int = 2,
) -> dict[str, np.ndarray]:
    # Load shared cache first as baseline; local run cache overrides it.
    shared_cache: dict[str, np.ndarray] = (
        _load_smiles_cache(shared_cache_path) if shared_cache_path is not None else {}
    )
    cache = _load_smiles_cache(cache_path)
    # Merge: shared provides base, local run cache takes precedence.
    merged: dict[str, np.ndarray] = {**shared_cache, **cache}

    unique_values = list(set(values))
    needed = [
        value
        for value in unique_values
        if value not in merged or merged[value].shape[0] < amount_of_replicates
    ]
    shared_hits = sum(
        1
        for v in unique_values
        if v in shared_cache
        and v not in cache
        and shared_cache[v].shape[0] >= amount_of_replicates
    )
    local_hits = sum(
        1 for v in unique_values if v in cache and cache[v].shape[0] >= amount_of_replicates
    )
    if logger is not None:
        logger.info(
            f"Cache {cache_path.name}: cache_entries_total={len(merged)}, "
            f"request_unique={len(unique_values)}, "
            f"local_hits={local_hits}, shared_hits={shared_hits}, "
            f"new_to_embed={len(needed)}, requested_replicates={amount_of_replicates}",
            print_level=print_level,
        )

    if needed:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if save_every_batches < 1:
            raise ValueError("save_every_batches must be >= 1")

        needed_batches = _chunk_list(needed, batch_size)
        progress = None
        if use_tqdm and print_level >= 2:
            progress = tqdm(
                total=len(needed),
                desc="Embedding SMILES",
                unit="smiles",
                leave=False,
            )

        if logger is not None:
            logger.info(
                f"Generating embeddings for {len(needed)} uncached SMILES "
                f"({amount_of_replicates} replicate(s) each)",
                print_level=print_level,
            )

        for batch_idx, needed_batch in enumerate(needed_batches, start=1):
            new_vectors = smiles_to_embedding(
                needed_batch,
                logger=logger,
                print_level=print_level,
                amount_of_replicates=amount_of_replicates,
                log_start=False,
            )
            if new_vectors is None:
                raise RuntimeError(f"Could not create embeddings for {cache_path.name}")
            for idx, key in enumerate(needed_batch):
                merged[key] = np.asarray(new_vectors[idx])

            if batch_idx % save_every_batches == 0 or batch_idx == len(needed_batches):
                _save_smiles_cache(cache_path, merged)
                if shared_cache_path is not None:
                    shared_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_smiles_cache(shared_cache_path, merged)

            if progress is not None:
                progress.update(len(needed_batch))

            if logger is not None and print_level >= 3:
                logger.info(
                    f"{cache_path.name}: processed embedding batch {batch_idx}/"
                    f"{len(needed_batches)}",
                    print_level=print_level,
                )

        if progress is not None:
            progress.close()

        if logger is not None:
            logger.valid("SMILES embedding generation complete", print_level=print_level)
    return merged


def _aggregate_log_predictions(log_values: np.ndarray) -> dict[str, float]:
    linear_values = np.array([10**x for x in log_values])
    mean_log = np.mean(log_values)
    return {
        "min": float(np.log10(np.min(linear_values)) if np.min(linear_values) > 0 else 0.0),
        "max": float(np.log10(np.max(linear_values)) if np.max(linear_values) > 0 else 0.0),
        "median": float(
            np.log10(np.median(linear_values)) if np.median(linear_values) > 0 else 0.0
        ),
        "mean": float(
            np.log10(np.mean(linear_values)) if np.mean(linear_values) > 0 else 0.0
        ),
        "iqr": float(np.percentile(log_values, 75) - np.percentile(log_values, 25)),
        "sd": float(np.std(log_values)),
        "sd_as_percent_of_mean": float(
            np.std(log_values) / mean_log if mean_log != 0 else 0.0
        ),
    }


def _build_missing_report_df(
    missing_genes: set[str],
    missing_smiles: set[str],
) -> pd.DataFrame:
    sorted_genes = sorted(missing_genes)
    sorted_smiles = sorted(missing_smiles)
    max_len = max(len(sorted_genes), len(sorted_smiles))
    if max_len == 0:
        return pd.DataFrame(columns=["missing_genes", "missing_smiles"])

    return pd.DataFrame(
        {
            "missing_genes": sorted_genes + [None] * (max_len - len(sorted_genes)),
            "missing_smiles": sorted_smiles + [None] * (max_len - len(sorted_smiles)),
        }
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _resolve_lean_model_root() -> Path:
    model_root = Path(__file__).resolve().parent / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root


def _resolve_lean_model_pickle(model_root: Path) -> Path:
    model_pickle = model_root / "kcat" / "UniKP for kcat.pkl"
    if model_pickle.exists():
        return model_pickle

    available_pickles = sorted(model_root.glob("*.pkl"))
    if len(available_pickles) == 1:
        return available_pickles[0]
    raise FileNotFoundError(
        "Could not resolve installed model pickle in models/. "
        "Expected models/UniKP20kcat.pkl or exactly one *.pkl file."
    )


def _build_lean_kcat_paths(output_dir: Path, base_cache_dir: Path | None = None) -> KcatPaths:
    if base_cache_dir is None:
        current_dir = Path(__file__).resolve().parent
        base_cache_dir = current_dir
    lean_output_dir = output_dir / "lean_kcat_inference"
    lean_output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = base_cache_dir / ".lookup_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using lookup cache directory: {cache_dir}")

    return KcatPaths(
        output_dir=output_dir,
        smiles_file=lean_output_dir / "metabolite_smiles.csv",
        sequence_file=lean_output_dir / "gene_or_transcript_protein_sequences.csv",
        gene_metabolite_pairs_file=lean_output_dir / "gene_metabolite_pairs.json",
        gene_metabolite_pairs_legacy_file=lean_output_dir
        / "gene_smiles_reactions_pairs.json",
        sequence_tensor_cache_file=cache_dir / "sequence_embedding_cache.pkl",
        smiles_tensor_cache_file=cache_dir / "smiles_embedding_cache.pkl",
        predictions_csv_file=lean_output_dir / "kcat_gene_metabolite_predictions.csv",
        predictions_json_file=lean_output_dir / "kcat_gene_metabolite_predictions.json",
        missing_csv_file=lean_output_dir / "missing_genes_and_smiles.csv",
    )


def _flatten_lean_pairs(gene_substrate_pairs: dict[str, set[str]]) -> list[tuple[str, str]]:
    all_pairs: list[tuple[str, str]] = []
    for gene_id, metabolite_ids in gene_substrate_pairs.items():
        for metabolite_id in metabolite_ids:
            all_pairs.append((str(gene_id), str(metabolite_id)))
    return sorted(set(all_pairs))


def _prepare_lean_sequence_map(
    transcript_df: pd.DataFrame,
    required_gene_ids: set[str],
) -> dict[str, str]:
    if "peptide_seq" not in transcript_df.columns:
        raise ValueError("transcript_df must contain 'peptide_seq' column")
    if transcript_df.index.hasnans:
        raise ValueError("transcript_df index must contain gene ids (no NaN index values)")

    seq_pairs = transcript_df[["peptide_seq"]].copy()
    seq_pairs = seq_pairs[seq_pairs["peptide_seq"].notna()]
    seq_pairs["peptide_seq"] = seq_pairs["peptide_seq"].astype(str)
    seq_pairs = seq_pairs[seq_pairs["peptide_seq"] != ""]
    seq_pairs.index = seq_pairs.index.astype(str)
    seq_pairs = seq_pairs[~seq_pairs.index.duplicated(keep="first")]
    if required_gene_ids:
        seq_pairs = seq_pairs.loc[seq_pairs.index.isin(required_gene_ids)]
    return seq_pairs["peptide_seq"].to_dict()


def _prepare_lean_smiles_map(
    smiles_df: pd.DataFrame,
    required_metabolite_ids: set[str],
    type_of_smiles: str,
) -> dict[str, str]:
    if type_of_smiles not in smiles_df.columns:
        raise ValueError(f"smiles_df must contain '{type_of_smiles}' column")
    if smiles_df.index.hasnans:
        raise ValueError("smiles_df index must contain metabolite ids (no NaN index values)")

    smiles_pairs = smiles_df[[type_of_smiles]].copy()
    smiles_pairs = smiles_pairs[smiles_pairs[type_of_smiles].notna()]
    smiles_pairs[type_of_smiles] = smiles_pairs[type_of_smiles].astype(str)
    smiles_pairs = smiles_pairs[smiles_pairs[type_of_smiles] != ""]
    smiles_pairs.index = smiles_pairs.index.astype(str)
    smiles_pairs = smiles_pairs[~smiles_pairs.index.duplicated(keep="first")]
    if required_metabolite_ids:
        smiles_pairs = smiles_pairs.loc[smiles_pairs.index.isin(required_metabolite_ids)]
    return smiles_pairs[type_of_smiles].to_dict()


def _get_truncated_smiles_ids(smiles_by_id: dict[str, str]) -> set[str]:
    truncated_smiles_metabolite_ids: set[str] = set()
    try:
        try:
            from UniKP.utils import split as _split_smiles
        except ModuleNotFoundError:
            from utils import split as _split_smiles  # ty: ignore[unresolved-import]

        for met_id, smiles in smiles_by_id.items():
            tokenized = _split_smiles(smiles)
            if len(tokenized.split()) > 218:
                truncated_smiles_metabolite_ids.add(met_id)
    except Exception:  # noqa: BLE001
        return set()
    return truncated_smiles_metabolite_ids


def _load_lean_prediction_cache(cache_file: Path) -> pd.DataFrame:
    if cache_file.exists():
        return _read_csv_flexible(cache_file)
    return pd.DataFrame()


def run_kcat_inference_lean(
    smiles_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    gene_substrate_pairs: dict[str, set[str]],
    output_path: Path,
    model_path: Path,
    chunk_size: int = 200,
    embedding_batch_size: int = 50,
    embedding_cache_save_every_batches: int = 1,
    prediction_checkpoint_every_chunks: int = 10,
    amount_of_smiles_replicates: int = 50,
    type_of_smiles: str = "isomeric_SMILES",
) -> tuple[KcatPaths, pd.DataFrame]:
    # print(f"Running lean kcat inference with model: {model_path}")
    model_root = _resolve_lean_model_root()
    model_pickle = _resolve_lean_model_pickle(model_root)
    paths = _build_lean_kcat_paths(output_dir=output_path, base_cache_dir=None)

    all_pairs = _flatten_lean_pairs(gene_substrate_pairs)
    required_gene_ids = {gene_id for gene_id, _ in all_pairs}
    required_metabolite_ids = {metabolite_id for _, metabolite_id in all_pairs}

    seq_by_gene = _prepare_lean_sequence_map(transcript_df, required_gene_ids)
    smiles_by_id = _prepare_lean_smiles_map(
        smiles_df, required_metabolite_ids, type_of_smiles
    )
    truncated_smiles_metabolite_ids = _get_truncated_smiles_ids(smiles_by_id)

    print(f"Total gene-substrate pairs: {len(all_pairs)}")
    sequence_cache = _update_embedding_cache(
        list(set(seq_by_gene.values())),
        paths.sequence_tensor_cache_file,
        lambda values: sequence_to_embedding(values, logger=None, use_tqdm=False),
        key_normalizer=_normalize_sequence_cache_key,
        batch_size=embedding_batch_size,
        save_every_batches=embedding_cache_save_every_batches,
        shared_cache_path=None,
        logger=None,
    )
    print(f"Total unique sequences: {len(sequence_cache)}")
    smiles_cache = _update_smiles_embedding_cache(
        list(set(smiles_by_id.values())),
        paths.smiles_tensor_cache_file,
        amount_of_replicates=amount_of_smiles_replicates,
        batch_size=embedding_batch_size,
        save_every_batches=embedding_cache_save_every_batches,
        use_tqdm=True,
        shared_cache_path=None,
        logger=None,
    )

    cache_file = paths.output_dir / "kcat_gene_metabolite_predictions_cache.csv"
    cache_scope_columns = ["cache_type_of_smiles", "cache_amount_of_smiles_replicates"]
    cache_df = _load_lean_prediction_cache(cache_file)
    for col in PREDICTION_COLUMNS + cache_scope_columns:
        if col not in cache_df.columns:
            cache_df[col] = np.nan

    cache_replicates = pd.to_numeric(
        cache_df["cache_amount_of_smiles_replicates"],
        errors="coerce",
    ).fillna(-1)
    active_cache = cache_df[
        (cache_df["cache_type_of_smiles"].astype(str) == str(type_of_smiles))
        & (cache_replicates.astype(int).eq(int(amount_of_smiles_replicates)))
    ].copy()
    if not active_cache.empty:
        active_cache = active_cache.drop_duplicates(
            subset=["ensemble_id", "metabolite_id"],
            keep="last",
        )

    existing_rows_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in active_cache.iterrows():
        gene_id = str(row.get("ensemble_id", ""))
        metabolite_id = str(row.get("metabolite_id", ""))
        if not gene_id or not metabolite_id:
            continue
        existing_rows_by_pair[(gene_id, metabolite_id)] = {
            str(column): value for column, value in row.to_dict().items()
        }

    done_pairs: set[tuple[str, str]] = set()
    pending_pairs: list[tuple[str, str, bool]] = []
    missing_genes: set[str] = set()
    missing_smiles: set[str] = set()
    for gene_id, metabolite_id in all_pairs:
        existing_row = existing_rows_by_pair.get((gene_id, metabolite_id))
        if existing_row is not None:
            # Missing-SMILES rows are never reusable cache entries.
            if not _as_bool(existing_row.get("missing_smiles", False)):
                done_pairs.add((gene_id, metabolite_id))
                continue
        if gene_id not in seq_by_gene:
            missing_genes.add(gene_id)
            continue
        if metabolite_id not in smiles_by_id:
            missing_smiles.add(metabolite_id)
            pending_pairs.append((gene_id, metabolite_id, True))
            continue
        pending_pairs.append((gene_id, metabolite_id, False))

    reused_rows_df = pd.DataFrame(
        [
            existing_rows_by_pair[pair]
            for pair in sorted(done_pairs)
            if pair in existing_rows_by_pair
        ]
    )

    new_rows: list[dict[str, object]] = []
    print(f"Total pending pairs for inference: {len(pending_pairs)}")

    if pending_pairs:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        model = safe_load_sklearn_model(model_pickle)
        print(f"Loaded model from {model_pickle}, starting inference...")
        for chunk_index, chunk_start in enumerate(
            range(0, len(pending_pairs), chunk_size), start=1
        ):
            print(
                f"Processing chunk {chunk_index} (pairs {chunk_start} to {chunk_start + chunk_size})"
            )
            pair_chunk = pending_pairs[chunk_start : chunk_start + chunk_size]
            chunk_feature_batches = []
            chunk_pair_meta: list[tuple[str, str, bool, bool, int]] = []

            if (
                any(is_missing_smi for _, _, is_missing_smi in pair_chunk)
                and "" not in smiles_cache
            ):
                empty_embedding = smiles_to_embedding(
                    [""],
                    logger=None,
                    print_level=0,
                    amount_of_replicates=1,
                    log_start=False,
                )
                if empty_embedding is None:
                    raise RuntimeError("Could not create empty-SMILES fallback embedding")
                smiles_cache[""] = np.asarray(empty_embedding[0])
                _save_smiles_cache(paths.smiles_tensor_cache_file, smiles_cache)

            for gene_id, metabolite_id, is_missing_smi in pair_chunk:
                sequence_key = _normalize_sequence_cache_key(seq_by_gene[gene_id])
                seq_vec = sequence_cache[sequence_key]
                if is_missing_smi:
                    smi_vecs = smiles_cache[""]
                    replicate_count = 1
                else:
                    smi_vecs = smiles_cache[smiles_by_id[metabolite_id]]
                    replicate_count = amount_of_smiles_replicates
                    if smi_vecs.shape[0] < amount_of_smiles_replicates:
                        raise RuntimeError(
                            f"Cached SMILES embeddings for {metabolite_id} have only "
                            f"{smi_vecs.shape[0]} replicate(s), expected "
                            f"{amount_of_smiles_replicates}"
                        )

                for rep_idx in range(replicate_count):
                    chunk_feature_batches.append(np.concatenate([smi_vecs[rep_idx], seq_vec]))

                chunk_pair_meta.append(
                    (
                        gene_id,
                        metabolite_id,
                        is_missing_smi,
                        (metabolite_id in truncated_smiles_metabolite_ids)
                        if not is_missing_smi
                        else False,
                        replicate_count,
                    )
                )

            predicted_chunk = model.predict(np.asarray(chunk_feature_batches))

            offset = 0
            for (
                gene_id,
                metabolite_id,
                is_missing_smi,
                is_truncated_smi,
                replicate_count,
            ) in chunk_pair_meta:
                replicate_slice = predicted_chunk[offset : offset + replicate_count]
                stats = _aggregate_log_predictions(np.asarray(replicate_slice))
                new_rows.append(
                    {
                        "ensemble_id": gene_id,
                        "metabolite_id": metabolite_id,
                        "missing_smiles": is_missing_smi,
                        "truncated_smiles": is_truncated_smi,
                        "cache_type_of_smiles": type_of_smiles,
                        "cache_amount_of_smiles_replicates": amount_of_smiles_replicates,
                        **stats,
                    }
                )
                offset += replicate_count

            if (
                prediction_checkpoint_every_chunks > 0
                and chunk_index % prediction_checkpoint_every_chunks == 0
            ):
                checkpoint_df = pd.concat(
                    [reused_rows_df, pd.DataFrame(new_rows)],
                    ignore_index=True,
                ).reindex(columns=PREDICTION_COLUMNS)
                checkpoint_df.to_csv(paths.predictions_csv_file, index=False)
                paths.predictions_json_file.write_text(
                    checkpoint_df.to_json(orient="records", indent=2),
                    encoding="utf-8",
                )

    new_rows_df = pd.DataFrame(new_rows)
    if new_rows_df.empty:
        new_rows_df = pd.DataFrame(columns=PREDICTION_COLUMNS + cache_scope_columns)

    predictions_df = pd.concat([reused_rows_df, new_rows_df], ignore_index=True)
    if predictions_df.empty:
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS + cache_scope_columns)
    predictions_df = predictions_df.drop_duplicates(
        subset=["ensemble_id", "metabolite_id"],
        keep="last",
    )

    active_cache_updated = predictions_df.reindex(
        columns=PREDICTION_COLUMNS + cache_scope_columns
    )
    other_scopes = cache_df[
        ~(
            (cache_df["cache_type_of_smiles"].astype(str) == str(type_of_smiles))
            & (cache_replicates.astype(int).eq(int(amount_of_smiles_replicates)))
        )
    ]
    full_cache_df = pd.concat([other_scopes, active_cache_updated], ignore_index=True)
    full_cache_df.to_csv(cache_file, index=False)

    output_predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    output_predictions_df = output_predictions_df.sort_values(
        by=["ensemble_id", "metabolite_id"]
    ).reset_index(drop=True)

    output_predictions_df.to_csv(paths.predictions_csv_file, index=False)
    paths.predictions_json_file.write_text(
        output_predictions_df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    missing_df = _build_missing_report_df(missing_genes, missing_smiles)
    missing_df.to_csv(paths.missing_csv_file, index=False)
    if model_path is not None:
        output_predictions_df.to_csv(
            model_path / "kcat_gene_metabolite_predictions.csv", index=False
        )
        missing_df.to_csv(model_path / "missing_genes_and_smiles.csv", index=False)

    return paths, output_predictions_df


def run_kcat_inference(
    *,
    model_file: Path,
    smiles_csv_file: Path | None = None,
    sequence_csv_file: Path | None = None,
    sequence_level: str = "gene",
    species: str | None = None,
    amount_of_smiles_replicates: int = 50,
    type_of_smiles: str = "isomeric SMILES",
    model_pickle: Path = Path("UniKP20kcat.pkl"),
    gene_id_mapping_file: Path | None = None,
    use_mapping_as_primary: bool = False,
    chunk_size: int = 200,
    embedding_batch_size: int = 50,
    embedding_cache_save_every_batches: int = 1,
    prediction_checkpoint_every_chunks: int = 10,
    target_pairs: list[tuple[str, str]] | None = None,
    kcat_root: Path | None = None,
    shared_embedding_cache_dir: Path | None = None,
    print_level: int = 2,
    logger: CustomLogger | None = None,
    use_tqdm: bool = True,
) -> KcatPaths:
    model_file = model_file.resolve()
    bootstrap_log_dir = model_file.parent / "logs"
    logger = _get_logger(logger, print_level=print_level, log_dir=bootstrap_log_dir)

    logger.starting("Starting UniKP Kcat inference", print_level=2)
    _log_key_value(logger, "Model file", model_file, print_level=2)
    _log_key_value(logger, "Sequence level", sequence_level, print_level=2)
    _log_key_value(logger, "Species", species if species else "auto", print_level=2)
    _log_key_value(
        logger,
        "Gene id mapping",
        gene_id_mapping_file if gene_id_mapping_file else "auto (model dir)",
        print_level=2,
    )
    _log_key_value(
        logger,
        "Use mapping as primary",
        use_mapping_as_primary,
        print_level=2,
    )
    _log_key_value(
        logger,
        "Shared embedding cache dir",
        shared_embedding_cache_dir if shared_embedding_cache_dir else "disabled",
        print_level=2,
    )
    _log_key_value(logger, "SMILES replicates", amount_of_smiles_replicates, print_level=2)
    _log_key_value(logger, "Embedding batch size", embedding_batch_size, print_level=2)
    _log_key_value(
        logger,
        "Prediction checkpoint every chunks",
        prediction_checkpoint_every_chunks,
        print_level=2,
    )
    _log_key_value(
        logger,
        "SMILES source",
        smiles_csv_file if smiles_csv_file else "hook/file",
        print_level=2,
    )
    _log_key_value(
        logger,
        "Sequence source",
        sequence_csv_file if sequence_csv_file else "output/retrieval",
        print_level=2,
    )
    _log_key_value(
        logger,
        "Target pairs",
        len(target_pairs) if target_pairs else "all model pairs",
        print_level=2,
    )

    _log_step(logger, "Loading COBRA model", print_level=2)
    cobra_model = load_json_model(model_file)
    _log_key_value(logger, "Genes in model", len(cobra_model.genes), print_level=2)
    _log_key_value(
        logger, "Metabolites in model", len(cobra_model.metabolites), print_level=2
    )

    _log_step(logger, "Resolving output paths", print_level=2)
    paths = build_kcat_paths(model_file, kcat_root=kcat_root)
    logger.set_log_files_location(str(paths.output_dir / "logs"))
    _log_key_value(logger, "Output dir", paths.output_dir, print_level=2)
    _log_key_value(logger, "Sequence file", paths.sequence_file, print_level=3)
    _log_key_value(logger, "SMILES file", paths.smiles_file, print_level=3)
    _log_key_value(logger, "Pairs file", paths.gene_metabolite_pairs_file, print_level=3)
    _log_key_value(logger, "Predictions CSV", paths.predictions_csv_file, print_level=3)

    # 1) Gene-metabolite pair file (new name + legacy mirror)
    _log_step(logger, "Writing gene-metabolite pairs", print_level=2)
    write_gene_metabolite_pairs(
        cobra_model,
        output_file=paths.gene_metabolite_pairs_file,
        legacy_file=paths.gene_metabolite_pairs_legacy_file,
    )
    pair_payload = json.loads(paths.gene_metabolite_pairs_file.read_text(encoding="utf-8"))
    all_pairs = [(str(v[0]), str(v[1])) for v in pair_payload.values()]
    _log_key_value(
        logger, "Total gene-metabolite pairs (model)", len(all_pairs), print_level=2
    )

    if target_pairs:
        requested_pairs = [(str(g), str(m)) for g, m in target_pairs]
        requested_set = set(requested_pairs)
        model_pair_set = set(all_pairs)
        missing_requested_pairs = sorted(requested_set - model_pair_set)
        all_pairs = [pair for pair in requested_pairs if pair in model_pair_set]
        _log_key_value(logger, "Requested target pairs", len(requested_pairs), print_level=2)
        _log_key_value(logger, "Valid target pairs in model", len(all_pairs), print_level=2)
        if missing_requested_pairs:
            logger.warning(
                f"Requested target pairs not found in model: {missing_requested_pairs}",
                print_level=2,
            )

    required_gene_ids = {gene_id for gene_id, _ in all_pairs}
    required_metabolite_ids = {metabolite_id for _, metabolite_id in all_pairs}
    _log_key_value(logger, "Pairs selected for processing", len(all_pairs), print_level=2)

    # 2) Sequence file: append only missing genes
    _log_step(logger, "Preparing gene/transcript sequence table", print_level=2)
    if sequence_csv_file is not None:
        sequence_df = _read_csv_flexible(sequence_csv_file)
        sequence_df.to_csv(paths.sequence_file, index=False)
        _log_key_value(
            logger,
            "Sequence source",
            f"Provided CSV ({sequence_csv_file})",
            print_level=2,
        )
    elif paths.sequence_file.exists():
        sequence_df = _read_csv_flexible(paths.sequence_file)
        _log_key_value(logger, "Existing sequence rows", len(sequence_df), print_level=2)
    else:
        sequence_df = pd.DataFrame(columns=["ensemble_id", "protein_sequence"])
        _log_key_value(logger, "Existing sequence rows", 0, print_level=2)

    existing_genes = set(
        sequence_df.get("ensemble_id", pd.Series(dtype=str)).astype(str).tolist()
    )
    missing_genes = sorted(required_gene_ids - existing_genes)
    _log_key_value(logger, "Genes missing sequence", len(missing_genes), print_level=2)
    if missing_genes:
        logger.info(f"Genes missing from sequence CSV: {missing_genes}", print_level=2)
    if missing_genes:
        # retrieve all and filter to missing genes for stable provider behavior
        generated = get_gene_aa_sequences(
            cobra_model,
            level=sequence_level,
            species=species,
            model_file=model_file,
            gene_id_mapping_file=gene_id_mapping_file,
            use_mapping_as_primary=use_mapping_as_primary,
            logger=logger,
            print_level=print_level,
        )
        _log_key_value(logger, "Retrieved rows", len(generated), print_level=2)
        generated = generated[generated["ensemble_id"].astype(str).isin(missing_genes)]
        sequence_df = pd.concat([sequence_df, generated], ignore_index=True)
        if sequence_level == "gene":
            sequence_df = sequence_df.drop_duplicates(subset=["ensemble_id"], keep="first")
        else:
            dedup_cols = ["ensemble_id", "transcript_or_accession"]
            dedup_cols = [col for col in dedup_cols if col in sequence_df.columns]
            sequence_df = sequence_df.drop_duplicates(subset=dedup_cols, keep="first")
        sequence_df.to_csv(paths.sequence_file, index=False)
    _log_key_value(logger, "Final sequence rows", len(sequence_df), print_level=2)
    genes_with_empty_sequence = sequence_df[
        sequence_df["protein_sequence"].isna() | (sequence_df["protein_sequence"] == "")
    ]["ensemble_id"].tolist()
    if genes_with_empty_sequence:
        logger.info(
            f"Genes in CSV with empty/null sequences: {genes_with_empty_sequence}",
            print_level=2,
        )

    # compatibility mirror expected by old scripts
    legacy_sequence_file = paths.output_dir / "final_transcript_sequence_df.csv"
    sequence_df.to_csv(legacy_sequence_file, index=False)

    # 3) SMILES file (manual for now unless hook implemented)
    _log_step(logger, "Preparing metabolite SMILES table", print_level=2)
    if smiles_csv_file is not None:
        smiles_df = _read_csv_flexible(smiles_csv_file)
        smiles_df.to_csv(paths.smiles_file, index=False)
        _log_key_value(
            logger, "SMILES source", f"Provided CSV ({smiles_csv_file})", print_level=2
        )
    elif paths.smiles_file.exists():
        smiles_df = _read_csv_flexible(paths.smiles_file)
        _log_key_value(logger, "SMILES source", "Existing output file", print_level=2)
    else:
        smiles_df = get_metabolite_smiles(cobra_model)
        smiles_df.to_csv(paths.smiles_file, index=False)
        _log_key_value(logger, "SMILES source", "SMILES hook implementation", print_level=2)
    _log_key_value(logger, "SMILES rows", len(smiles_df), print_level=2)

    # 4) Incremental embedding caches
    _log_step(logger, "Building/Updating embedding caches", print_level=2)
    seq_pairs: pd.DataFrame = (
        sequence_df.loc[:, ["ensemble_id", "protein_sequence"]].dropna().copy()
    )
    if required_gene_ids:
        seq_pairs = seq_pairs[seq_pairs["ensemble_id"].astype(str).isin(required_gene_ids)]
    if "is_canonical" in sequence_df.columns:
        seq_pairs["is_canonical"] = (
            sequence_df.loc[seq_pairs.index, "is_canonical"].fillna(False).astype(bool)
        )
        seq_pairs = seq_pairs.sort_values(  # type: ignore[call-overload]
            by=["ensemble_id", "is_canonical"],
            ascending=[True, False],
        )
    seq_pairs = seq_pairs.drop_duplicates(  # type: ignore[call-overload]
        subset=["ensemble_id"], keep="first"
    )
    seq_by_gene = dict(
        zip(seq_pairs["ensemble_id"].astype(str), seq_pairs["protein_sequence"].astype(str))
    )
    _log_key_value(logger, "Genes with usable sequences", len(seq_by_gene), print_level=2)
    smiles_pairs = smiles_df[["id", type_of_smiles]].dropna()
    smiles_by_id_exact = dict(
        zip(smiles_pairs["id"].astype(str), smiles_pairs[type_of_smiles].astype(str))
    )
    normalized_smiles_id_to_ids: dict[str, list[str]] = {}
    for smiles_met_id in smiles_by_id_exact:
        norm_id = _normalize_metabolite_id_for_matching(smiles_met_id)
        normalized_smiles_id_to_ids.setdefault(norm_id, []).append(smiles_met_id)

    # Resolve model metabolite ids to available SMILES ids.
    # Priority: exact id match -> unique normalized-id match.
    smiles_by_id: dict[str, str] = {}
    smiles_source_id_by_model_metabolite: dict[str, str] = {}
    resolved_by_exact_id = 0
    resolved_by_compartment_variant = 0
    resolved_by_normalized_id = 0
    ambiguous_normalized_ids: set[str] = set()
    if required_metabolite_ids:
        metabolite_ids_to_resolve = required_metabolite_ids
    else:
        metabolite_ids_to_resolve = set(smiles_by_id_exact.keys())

    for model_metabolite_id in metabolite_ids_to_resolve:
        if model_metabolite_id in smiles_by_id_exact:
            smiles_by_id[model_metabolite_id] = smiles_by_id_exact[model_metabolite_id]
            smiles_source_id_by_model_metabolite[model_metabolite_id] = model_metabolite_id
            resolved_by_exact_id += 1
            continue

        resolved_variant = False
        for variant_id in _compartment_format_variants(model_metabolite_id):
            if variant_id in smiles_by_id_exact:
                smiles_by_id[model_metabolite_id] = smiles_by_id_exact[variant_id]
                smiles_source_id_by_model_metabolite[model_metabolite_id] = variant_id
                resolved_by_compartment_variant += 1
                resolved_variant = True
                break
        if resolved_variant:
            continue

        normalized_model_id = _normalize_metabolite_id_for_matching(model_metabolite_id)
        candidate_smiles_ids = normalized_smiles_id_to_ids.get(normalized_model_id, [])
        if len(candidate_smiles_ids) == 1:
            source_smiles_id = candidate_smiles_ids[0]
            smiles_by_id[model_metabolite_id] = smiles_by_id_exact[source_smiles_id]
            smiles_source_id_by_model_metabolite[model_metabolite_id] = source_smiles_id
            resolved_by_normalized_id += 1
            continue

        if len(candidate_smiles_ids) > 1:
            candidate_smiles_values = {
                smiles_by_id_exact[candidate_id] for candidate_id in candidate_smiles_ids
            }
            # Deterministic fallback if multiple IDs normalize to same key but
            # still map to one identical SMILES value.
            if len(candidate_smiles_values) == 1:
                source_smiles_id = sorted(candidate_smiles_ids)[0]
                smiles_by_id[model_metabolite_id] = smiles_by_id_exact[source_smiles_id]
                smiles_source_id_by_model_metabolite[model_metabolite_id] = source_smiles_id
                resolved_by_normalized_id += 1
                continue
            ambiguous_normalized_ids.add(normalized_model_id)

    # Track which metabolite ids would be truncated during SMILES tokenization.
    truncated_smiles_metabolite_ids: set[str] = set()
    try:
        try:
            from UniKP.utils import split as _split_smiles
        except ModuleNotFoundError:
            # noqa:  I001# ty: ignore[unresolved-import]
            from utils import split as _split_smiles

        for met_id, smiles in zip(
            smiles_pairs["id"].astype(str),
            smiles_pairs[type_of_smiles].astype(str),
        ):
            tokenized = _split_smiles(smiles)
            if len(tokenized.split()) > 218:
                truncated_smiles_metabolite_ids.add(met_id)
    except Exception:  # noqa: BLE001
        # If tokenization helper import fails, continue without this annotation.
        truncated_smiles_metabolite_ids = set()

    # Remap truncation annotations from source SMILES ids to model metabolite ids.
    truncated_smiles_metabolite_ids_resolved: set[str] = set()
    for model_metabolite_id, source_smiles_id in smiles_source_id_by_model_metabolite.items():
        if source_smiles_id in truncated_smiles_metabolite_ids:
            truncated_smiles_metabolite_ids_resolved.add(model_metabolite_id)

    _log_key_value(logger, "Metabolites with usable SMILES", len(smiles_by_id), print_level=2)
    _log_key_value(
        logger,
        "SMILES id resolution (exact)",
        resolved_by_exact_id,
        print_level=2,
    )
    _log_key_value(
        logger,
        "SMILES id resolution (compartment format fallback)",
        resolved_by_compartment_variant,
        print_level=2,
    )
    _log_key_value(
        logger,
        "SMILES id resolution (normalized fallback)",
        resolved_by_normalized_id,
        print_level=2,
    )
    _log_key_value(
        logger,
        "SMILES id resolution (ambiguous normalized ids)",
        len(ambiguous_normalized_ids),
        print_level=2,
    )
    model_met_ids = {str(m.id) for m in cobra_model.metabolites}
    smiles_met_ids = set(smiles_by_id.keys())
    raw_met_overlap = len(model_met_ids & smiles_met_ids)
    norm_model_met_ids = {_normalize_metabolite_id_for_matching(x) for x in model_met_ids}
    norm_smiles_met_ids = {_normalize_metabolite_id_for_matching(x) for x in smiles_met_ids}
    norm_met_overlap = len(norm_model_met_ids & norm_smiles_met_ids)
    _log_key_value(
        logger, "Model-vs-SMILES metabolite id overlap (raw)", raw_met_overlap, print_level=2
    )
    _log_key_value(
        logger,
        "Model-vs-SMILES metabolite id overlap (normalized)",
        norm_met_overlap,
        print_level=2,
    )

    sequence_cache = _update_embedding_cache(
        list(set(seq_by_gene.values())),
        paths.sequence_tensor_cache_file,
        lambda values: sequence_to_embedding(
            values,
            logger=logger,
            print_level=print_level,
            use_tqdm=use_tqdm,
        ),
        key_normalizer=_normalize_sequence_cache_key,
        batch_size=embedding_batch_size,
        save_every_batches=embedding_cache_save_every_batches,
        shared_cache_path=(
            shared_embedding_cache_dir / "sequence_embedding_cache.pkl"
            if shared_embedding_cache_dir is not None
            else None
        ),
        logger=logger,
        print_level=print_level,
    )
    # SMILES shared cache is keyed per smiles-type so isomeric and canonical don't mix.
    _smiles_type_slug = type_of_smiles.lower().replace(" ", "_")
    smiles_cache = _update_smiles_embedding_cache(
        list(set(smiles_by_id.values())),
        paths.smiles_tensor_cache_file,
        amount_of_replicates=amount_of_smiles_replicates,
        batch_size=embedding_batch_size,
        save_every_batches=embedding_cache_save_every_batches,
        use_tqdm=use_tqdm,
        shared_cache_path=(
            shared_embedding_cache_dir / f"smiles_embedding_cache_{_smiles_type_slug}.pkl"
            if shared_embedding_cache_dir is not None
            else None
        ),
        logger=logger,
        print_level=print_level,
    )
    _log_key_value(logger, "Sequence cache size", len(sequence_cache), print_level=2)
    _log_key_value(logger, "SMILES cache size", len(smiles_cache), print_level=2)

    # 5) Load previous predictions and determine pending pairs
    _log_step(logger, "Resolving pending predictions", print_level=2)
    if paths.predictions_csv_file.exists():
        predictions_df = _read_csv_flexible(paths.predictions_csv_file)
    else:
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    if predictions_df.empty:
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    _log_key_value(logger, "Existing prediction rows", len(predictions_df), print_level=2)

    done_pairs: set[tuple[str, str]] = set()
    existing_rows_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            gene_id = str(row.get("ensemble_id", ""))
            metabolite_id = str(row.get("metabolite_id", ""))
            if not gene_id or not metabolite_id:
                continue
            pair_key: tuple[str, str] = (gene_id, metabolite_id)
            row_dict: dict[str, object] = {
                str(column): value for column, value in row.to_dict().items()
            }
            existing_rows_by_pair[pair_key] = row_dict

    _log_key_value(logger, "Total gene-metabolite pairs", len(all_pairs), print_level=2)

    pending_pairs: list[tuple[str, str, bool]] = []
    missing_genes: set[str] = set()
    missing_smiles: set[str] = set()
    cached_pairs: set[tuple[str, str]] = set()
    stale_fallback_pairs: set[tuple[str, str]] = set()
    missing_gene_pairs_count = 0
    missing_smiles_pairs_count = 0
    for gene_id, metabolite_id in all_pairs:
        existing_row = existing_rows_by_pair.get((gene_id, metabolite_id))
        if existing_row is not None:
            cached_missing_smiles = _as_bool(existing_row.get("missing_smiles", False))
            # If an older run used missing-SMILES fallback but this run now has
            # SMILES, recompute this pair with full stochastic replicates.
            if cached_missing_smiles and metabolite_id in smiles_by_id:
                stale_fallback_pairs.add((gene_id, metabolite_id))
            else:
                done_pairs.add((gene_id, metabolite_id))
                cached_pairs.add((gene_id, metabolite_id))
                continue
        if gene_id not in seq_by_gene:
            missing_genes.add(gene_id)
            missing_gene_pairs_count += 1
            continue
        if metabolite_id not in smiles_by_id:
            missing_smiles.add(metabolite_id)
            missing_smiles_pairs_count += 1
            pending_pairs.append((gene_id, metabolite_id, True))
            continue
        pending_pairs.append((gene_id, metabolite_id, False))

    _log_key_value(logger, "Pending pairs", len(pending_pairs), print_level=2)
    _log_key_value(logger, "Pairs already cached", len(cached_pairs), print_level=2)
    _log_key_value(
        logger,
        "Stale cached fallback pairs to recompute",
        len(stale_fallback_pairs),
        print_level=2,
    )
    _log_key_value(logger, "Pairs missing genes", missing_gene_pairs_count, print_level=2)
    _log_key_value(logger, "Unique missing gene IDs", len(missing_genes), print_level=2)
    if missing_genes:
        logger.info(f"Missing gene IDs: {sorted(missing_genes)}", print_level=2)
    _log_key_value(logger, "Pairs missing SMILES", missing_smiles_pairs_count, print_level=2)
    _log_key_value(
        logger, "Unique missing metabolite IDs", len(missing_smiles), print_level=2
    )

    # Summary of pair filtering
    total_accounted = len(cached_pairs) + len(pending_pairs) + missing_gene_pairs_count
    logger.info(
        f"Pair summary: "
        f"total={len(all_pairs)}, "
        f"cached={len(cached_pairs)}, "
        f"stale_cached_fallback_recompute={len(stale_fallback_pairs)}, "
        f"ready_for_inference={len(pending_pairs)}, "
        f"missing_gene={missing_gene_pairs_count}, "
        f"missing_smiles_fallback_in_ready={missing_smiles_pairs_count}, "
        f"accounted={total_accounted}",
        print_level=2,
    )

    if stale_fallback_pairs and not predictions_df.empty:
        stale_keys = {
            f"{gene_id}|||{metabolite_id}" for gene_id, metabolite_id in stale_fallback_pairs
        }
        prediction_keys = (
            predictions_df["ensemble_id"].astype(str)
            + "|||"
            + predictions_df["metabolite_id"].astype(str)
        )
        stale_mask = prediction_keys.isin(stale_keys)
        stale_rows_removed = int(stale_mask.sum())
        predictions_df = predictions_df.loc[~stale_mask].copy()
        if stale_rows_removed and logger is not None:
            logger.info(
                f"Dropped stale cached fallback prediction rows: {stale_rows_removed}",
                print_level=2,
            )

    if pending_pairs:
        _log_step(logger, "Running model inference for pending pairs", print_level=2)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        model = safe_load_sklearn_model(model_pickle)
        # Missing-SMILES pairs use a single empty-SMILES embedding replicate.
        total_feature_rows = sum(
            1 for _, _, is_missing_smi in pending_pairs if is_missing_smi
        ) + sum(
            amount_of_smiles_replicates
            for _, _, is_missing_smi in pending_pairs
            if not is_missing_smi
        )
        total_chunks = (len(pending_pairs) + chunk_size - 1) // chunk_size
        _log_key_value(logger, "Inference chunk size (pairs)", chunk_size, print_level=2)
        _log_key_value(logger, "Total chunks", total_chunks, print_level=2)
        _log_key_value(logger, "Total feature rows", total_feature_rows, print_level=2)

        rows = []
        chunk_starts = range(0, len(pending_pairs), chunk_size)
        if use_tqdm:
            chunk_starts = tqdm(
                chunk_starts,
                total=total_chunks,
                desc="Predicting chunks",
                unit="chunk",
                leave=False,
            )

        for chunk_index, chunk_start in enumerate(chunk_starts, start=1):
            pair_chunk = pending_pairs[chunk_start : chunk_start + chunk_size]
            chunk_feature_batches = []
            chunk_pair_meta: list[tuple[str, str, bool, bool, int]] = []

            # Lazily create and persist one empty-SMILES embedding when first needed.
            if (
                any(is_missing_smi for _, _, is_missing_smi in pair_chunk)
                and "" not in smiles_cache
            ):
                empty_embedding = smiles_to_embedding(
                    [""],
                    logger=logger,
                    print_level=print_level,
                    amount_of_replicates=1,
                    log_start=False,
                )
                if empty_embedding is None:
                    raise RuntimeError("Could not create empty-SMILES fallback embedding")
                smiles_cache[""] = np.asarray(empty_embedding[0])
                _save_smiles_cache(paths.smiles_tensor_cache_file, smiles_cache)

            for gene_id, metabolite_id, is_missing_smi in pair_chunk:
                sequence_key = _normalize_sequence_cache_key(seq_by_gene[gene_id])
                seq_vec = sequence_cache[sequence_key]
                if is_missing_smi:
                    smi_vecs = smiles_cache[""]
                    replicate_count = 1
                else:
                    smi_vecs = smiles_cache[smiles_by_id[metabolite_id]]
                    replicate_count = amount_of_smiles_replicates
                    if smi_vecs.shape[0] < amount_of_smiles_replicates:
                        raise RuntimeError(
                            f"Cached SMILES embeddings for {metabolite_id} have only "
                            f"{smi_vecs.shape[0]} replicate(s), expected "
                            f"{amount_of_smiles_replicates}"
                        )

                for rep_idx in range(replicate_count):
                    chunk_feature_batches.append(np.concatenate([smi_vecs[rep_idx], seq_vec]))

                chunk_pair_meta.append(
                    (
                        gene_id,
                        metabolite_id,
                        is_missing_smi,
                        (metabolite_id in truncated_smiles_metabolite_ids_resolved)
                        if not is_missing_smi
                        else False,
                        replicate_count,
                    )
                )

            predicted_chunk = model.predict(np.asarray(chunk_feature_batches))

            offset = 0
            for (
                gene_id,
                metabolite_id,
                is_missing_smi,
                is_truncated_smi,
                replicate_count,
            ) in chunk_pair_meta:
                replicate_slice = predicted_chunk[offset : offset + replicate_count]
                stats = _aggregate_log_predictions(np.asarray(replicate_slice))
                rows.append(
                    {
                        "ensemble_id": gene_id,
                        "metabolite_id": metabolite_id,
                        "missing_smiles": is_missing_smi,
                        "truncated_smiles": is_truncated_smi,
                        **stats,
                    }
                )
                offset += replicate_count

            if (
                prediction_checkpoint_every_chunks > 0
                and chunk_index % prediction_checkpoint_every_chunks == 0
            ):
                checkpoint_df = pd.concat(
                    [predictions_df, pd.DataFrame(rows)], ignore_index=True
                )
                checkpoint_df = checkpoint_df.reindex(columns=PREDICTION_COLUMNS)
                checkpoint_df.to_csv(paths.predictions_csv_file, index=False)
                paths.predictions_json_file.write_text(
                    checkpoint_df.to_json(orient="records", indent=2), encoding="utf-8"
                )
                if logger is not None:
                    logger.info(
                        f"Checkpoint written after chunk {chunk_index}/{total_chunks}",
                        print_level=3,
                    )

        logger.valid("Model prediction completed", print_level=2)
        if rows:
            predictions_df = pd.concat(
                [predictions_df, pd.DataFrame(rows)], ignore_index=True
            )
    else:
        logger.info("No pending pairs to run; using cached predictions only", print_level=2)

    _log_step(logger, "Writing outputs", print_level=2)
    predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    predictions_df.to_csv(paths.predictions_csv_file, index=False)
    paths.predictions_json_file.write_text(
        predictions_df.to_json(orient="records", indent=2), encoding="utf-8"
    )
    _log_key_value(logger, "Final prediction rows", len(predictions_df), print_level=2)

    missing_df = _build_missing_report_df(missing_genes, missing_smiles)
    missing_df.to_csv(paths.missing_csv_file, index=False)
    _log_key_value(logger, "Missing report rows", len(missing_df), print_level=2)

    logger.finished("UniKP Kcat inference completed", print_level=2)
    _log_key_value(logger, "Predictions CSV", paths.predictions_csv_file, print_level=2)
    _log_key_value(logger, "Predictions JSON", paths.predictions_json_file, print_level=2)
    _log_key_value(logger, "Missing report", paths.missing_csv_file, print_level=2)

    return paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _find_first_model_json(model_dir: Path) -> Path:
    for file in sorted(model_dir.iterdir()):
        if file.name.startswith("model_") and file.suffix == ".json":
            return file
    raise FileNotFoundError(f"No model_*.json found in {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental UniKP kcat inference runner")
    parser.add_argument("--model-file", type=Path, required=False)
    parser.add_argument("--model-dir", type=Path, required=False)
    parser.add_argument("--smiles-csv", type=Path, required=False)
    parser.add_argument("--sequence-level", choices=["gene", "transcript"], default="gene")
    parser.add_argument("--species", type=str, required=False)
    parser.add_argument("--gene-id-mapping-file", type=Path, required=False)
    parser.add_argument("--use-mapping-as-primary", action="store_true")
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--embedding-batch-size", type=int, default=50)
    parser.add_argument("--embedding-cache-save-every-batches", type=int, default=1)
    parser.add_argument("--prediction-checkpoint-every-chunks", type=int, default=10)
    parser.add_argument("--print-level", type=int, default=2)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    if args.model_file is None and args.model_dir is None:
        raise ValueError("Provide either --model-file or --model-dir")

    model_file = (
        args.model_file.resolve()
        if args.model_file
        else _find_first_model_json(args.model_dir.resolve())
    )
    run_kcat_inference(
        model_file=model_file,
        smiles_csv_file=args.smiles_csv.resolve() if args.smiles_csv else None,
        sequence_level=args.sequence_level,
        species=args.species,
        gene_id_mapping_file=(
            args.gene_id_mapping_file.resolve() if args.gene_id_mapping_file else None
        ),
        use_mapping_as_primary=args.use_mapping_as_primary,
        amount_of_smiles_replicates=args.replicates,
        chunk_size=args.chunk_size,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache_save_every_batches=args.embedding_cache_save_every_batches,
        prediction_checkpoint_every_chunks=args.prediction_checkpoint_every_chunks,
        print_level=args.print_level,
        use_tqdm=not args.no_tqdm,
    )


def run_small_test():
    project_root = get_project_root()
    model_dir = project_root / "data" / "for_SWAMP" / "models" / "model_inhouse_v9_human"
    model_file = _find_first_model_json(model_dir)

    reference_pair = ("ENSG00000172955", "MAM01249[c]")
    smiles_source = model_dir / "final_SMILES_metabolite_df.csv"
    sequence_source = model_dir / "final_transcript_sequence_df.csv"
    model_pickle = project_root / "UniKP" / "UniKP20kcat.pkl"

    required_files = [model_file, smiles_source, sequence_source, model_pickle]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"run_small_test missing required file(s): {missing_list}")

    smiles_df = _read_csv_flexible(smiles_source)
    smiles_df = smiles_df[
        (smiles_df["id"].astype(str) == reference_pair[1])
        & smiles_df["isomeric SMILES"].notna()
        & (smiles_df["isomeric SMILES"].astype(str) != "")
    ][["id", "isomeric SMILES"]]
    if smiles_df.empty:
        raise ValueError(
            f"Small test metabolite not found or missing SMILES: {reference_pair[1]}"
        )

    sequence_df = _read_csv_flexible(sequence_source)
    sequence_df = sequence_df[
        (sequence_df["ensemble_id"].astype(str) == reference_pair[0])
        & sequence_df["protein_sequence"].notna()
        & (sequence_df["protein_sequence"].astype(str) != "")
    ][["ensemble_id", "protein_sequence"]]
    if sequence_df.empty:
        raise ValueError(
            f"Small test gene not found or missing sequence: {reference_pair[0]}"
        )

    small_test_dir = project_root / "data" / "for_SWAMP" / "Kcat_predictions" / "_small_test"
    input_dir = small_test_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    smiles_input = input_dir / "small_test_smiles.csv"
    sequence_input = input_dir / "small_test_sequence.csv"
    smiles_df.to_csv(smiles_input, index=False)
    sequence_df.to_csv(sequence_input, index=False)

    return run_kcat_inference(
        model_file=model_file,
        smiles_csv_file=smiles_input,
        sequence_csv_file=sequence_input,
        species="human",
        sequence_level="gene",
        amount_of_smiles_replicates=50,
        type_of_smiles="isomeric SMILES",
        model_pickle=model_pickle,
        target_pairs=[reference_pair],
        kcat_root=small_test_dir,
        chunk_size=1,
        embedding_batch_size=10,
        embedding_cache_save_every_batches=1,
        prediction_checkpoint_every_chunks=1,
        print_level=2,
        use_tqdm=False,
    )


if __name__ == "__main__":
    # main()
    run_mode = "full"  # options: "small_test", "full"

    if run_mode == "small_test":
        run_small_test()
        raise SystemExit(0)

    # run using model
    project_root = get_project_root()
    data_dir = project_root / "data"
    models_dir = data_dir / "for_SWAMP" / "models"
    model_name = "mouseGEM_1_8_mouse_inhouse_v9"
    model_dir = models_dir / model_name
    model_file = _find_first_model_json(model_dir)

    ############# user input #############
    species = "mouse"
    sequence_level = "gene"
    amount_of_smiles_replicates = 200
    chunk_size = 200
    embedding_batch_size = 200
    embedding_cache_save_every_batches = 1
    prediction_checkpoint_every_chunks = 10
    print_level = 2

    alt_model_dir = data_dir / "for_SWAMP" / "models" / "model_inhouse_v9_human"
    smiles_csv = alt_model_dir / "final_SMILES_metabolite_df.csv"
    # additional_mapping_file = model_dir / "MouseGEM_1_8_MGI_gene_ID_mapping.csv"

    # test case, only do 50 pairs for now

    run_kcat_inference(
        model_file=model_file,
        smiles_csv_file=smiles_csv,
        sequence_level=sequence_level,
        species=species,
        amount_of_smiles_replicates=amount_of_smiles_replicates,
        chunk_size=chunk_size,
        embedding_batch_size=embedding_batch_size,
        embedding_cache_save_every_batches=embedding_cache_save_every_batches,
        prediction_checkpoint_every_chunks=prediction_checkpoint_every_chunks,
        # gene_id_mapping_file=additional_mapping_file,
        print_level=print_level,
    )
