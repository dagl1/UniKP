from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from SWAMP.utils.file_handling import get_project_root

try:
    from src.SWAMP.utils.custom_logging import CustomLogger
except ModuleNotFoundError:
    from SWAMP.utils.custom_logging import CustomLogger

try:
    from UniKP.compat_sklearn import safe_load_sklearn_model
except ModuleNotFoundError:
    from compat_sklearn import safe_load_sklearn_model  # ty: ignore[unresolved-import] # noqa: I001

from src.cobrapy_fork.io import load_json_model
from src.SWAMP.sequence_retrieval import (
    SequenceMode,
    infer_species,
    retrieve_gem_sequences,
    retrieve_gene_sequences,
)


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
    normalized = re.sub(r"\[[^]]+]$", "", metabolite_id)
    normalized = re.sub(r"_[a-zA-Z]$", "", normalized)
    return normalized


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
    vocab = WordVocab.load_vocab("vocab.pkl")

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
    trfm.load_state_dict(
        torch.load(
            "trfm_12_23000.pkl",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    trfm.train()

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

    replicate_embeddings = []
    replicate_iterator = range(amount_of_replicates)
    with torch.no_grad():
        for _ in replicate_iterator:
            encoded = trfm.encode(torch.t(xid))
            replicate_embeddings.append(np.asarray(encoded))

    encoded = np.stack(replicate_embeddings, axis=1)
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

    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    features = []
    iterator = enumerate(sequence_examples)
    if use_tqdm and print_level >= 2:
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
    logger: CustomLogger | None = None,
    print_level: int = 2,
) -> dict[str, np.ndarray]:
    cache = _load_cache(cache_path)
    normalizer = key_normalizer or (lambda v: v)
    normalized_values = list({normalizer(value) for value in values})
    needed = [value for value in normalized_values if value not in cache]
    request_hits = len(normalized_values) - len(needed)
    if logger is not None:
        logger.info(
            f"Cache {cache_path.name}: cache_entries_total={len(cache)}, "
            f"request_unique={len(normalized_values)}, request_hits={request_hits}, "
            f"new_to_embed={len(needed)}",
            print_level=print_level,
        )
    if needed:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if save_every_batches < 1:
            raise ValueError("save_every_batches must be >= 1")

        needed_batches = _chunk_list(needed, batch_size)
        for batch_idx, needed_batch in enumerate(needed_batches, start=1):
            new_vectors = vectorizer(needed_batch)
            if new_vectors is None:
                raise RuntimeError(f"Could not create embeddings for {cache_path.name}")
            for idx, key in enumerate(needed_batch):
                cache[key] = np.asarray(new_vectors[idx])

            if batch_idx % save_every_batches == 0 or batch_idx == len(needed_batches):
                _save_cache(cache_path, cache)

            if logger is not None and print_level >= 3:
                logger.info(
                    f"{cache_path.name}: processed embedding batch {batch_idx}/"
                    f"{len(needed_batches)}",
                    print_level=print_level,
                )
    return cache


def _update_smiles_embedding_cache(
    values: list[str],
    cache_path: Path,
    amount_of_replicates: int,
    batch_size: int = 50,
    save_every_batches: int = 1,
    use_tqdm: bool = True,
    logger: CustomLogger | None = None,
    print_level: int = 2,
) -> dict[str, np.ndarray]:
    cache = _load_smiles_cache(cache_path)
    unique_values = list(set(values))
    needed = [
        value
        for value in unique_values
        if value not in cache or cache[value].shape[0] < amount_of_replicates
    ]
    request_hits = len(unique_values) - len(needed)
    if logger is not None:
        logger.info(
            f"Cache {cache_path.name}: cache_entries_total={len(cache)}, "
            f"request_unique={len(unique_values)}, request_hits={request_hits}, "
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
                cache[key] = np.asarray(new_vectors[idx])

            if batch_idx % save_every_batches == 0 or batch_idx == len(needed_batches):
                _save_smiles_cache(cache_path, cache)

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
    return cache


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


def run_kcat_inference(
    *,
    model_file: Path,
    smiles_csv_file: Path | None = None,
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

    _log_step(logger, "Loading COBRA model", print_level=2)
    cobra_model = load_json_model(model_file)
    _log_key_value(logger, "Genes in model", len(cobra_model.genes), print_level=2)
    _log_key_value(
        logger, "Metabolites in model", len(cobra_model.metabolites), print_level=2
    )

    _log_step(logger, "Resolving output paths", print_level=2)
    paths = build_kcat_paths(model_file)
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
    _log_key_value(logger, "Total gene-metabolite pairs", len(pair_payload), print_level=2)

    # 2) Sequence file: append only missing genes
    _log_step(logger, "Preparing gene/transcript sequence table", print_level=2)
    if paths.sequence_file.exists():
        sequence_df = _read_csv_flexible(paths.sequence_file)
        _log_key_value(logger, "Existing sequence rows", len(sequence_df), print_level=2)
    else:
        sequence_df = pd.DataFrame(columns=["ensemble_id", "protein_sequence"])
        _log_key_value(logger, "Existing sequence rows", 0, print_level=2)

    model_genes = {gene.id for gene in cobra_model.genes}
    existing_genes = set(
        sequence_df.get("ensemble_id", pd.Series(dtype=str)).astype(str).tolist()
    )
    missing_genes = sorted(model_genes - existing_genes)
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
    smiles_by_id = dict(
        zip(smiles_pairs["id"].astype(str), smiles_pairs[type_of_smiles].astype(str))
    )
    # Track which metabolite ids would be truncated during SMILES tokenization.
    truncated_smiles_metabolite_ids: set[str] = set()
    try:
        try:
            from UniKP.utils import split as _split_smiles
        except ModuleNotFoundError:
            from utils import split as _split_smiles  # noqa:  I001# ty: ignore[unresolved-import]

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

    _log_key_value(logger, "Metabolites with usable SMILES", len(smiles_by_id), print_level=2)
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
        logger=logger,
        print_level=print_level,
    )
    smiles_cache = _update_smiles_embedding_cache(
        list(set(smiles_by_id.values())),
        paths.smiles_tensor_cache_file,
        amount_of_replicates=amount_of_smiles_replicates,
        batch_size=embedding_batch_size,
        save_every_batches=embedding_cache_save_every_batches,
        use_tqdm=use_tqdm,
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
    if not predictions_df.empty:
        done_pairs = set(
            zip(
                predictions_df["ensemble_id"].astype(str),
                predictions_df["metabolite_id"].astype(str),
            )
        )

    all_pairs = [(str(v[0]), str(v[1])) for v in pair_payload.values()]
    _log_key_value(logger, "Total gene-metabolite pairs", len(all_pairs), print_level=2)

    pending_pairs: list[tuple[str, str, bool]] = []
    missing_genes: set[str] = set()
    missing_smiles: set[str] = set()
    cached_pairs: set[tuple[str, str]] = set()
    missing_gene_pairs_count = 0
    missing_smiles_pairs_count = 0
    for gene_id, metabolite_id in all_pairs:
        if (gene_id, metabolite_id) in done_pairs:
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
        f"ready_for_inference={len(pending_pairs)}, "
        f"missing_gene={missing_gene_pairs_count}, "
        f"missing_smiles_fallback_in_ready={missing_smiles_pairs_count}, "
        f"accounted={total_accounted}",
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
        if use_tqdm and print_level >= 2:
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
                        print_level=print_level,
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

    missing_df = pd.DataFrame(
        {
            "missing_genes": list(sorted(missing_genes))
            + [None] * (len(missing_smiles) - len(missing_genes)),
            "missing_smiles": sorted(missing_smiles),
        }
    )
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


if __name__ == "__main__":
    # main()
    # run using model
    project_root = get_project_root()
    data_dir = project_root / "data"
    models_dir = data_dir / "for_SWAMP" / "models"
    model_name = "MouseGEM_1_8_mouse"
    model_dir = models_dir / model_name
    model_file = _find_first_model_json(model_dir)

    ############# user input #############
    species = "mouse"
    sequence_level = "gene"
    amount_of_smiles_replicates = 50
    chunk_size = 200
    embedding_batch_size = 50
    embedding_cache_save_every_batches = 1
    prediction_checkpoint_every_chunks = 10
    print_level = 2

    smiles_csv = model_dir / "final_SMILES_metabolite_df.csv"
    additional_mapping_file = model_dir / "MouseGEM_1_8_MGI_gene_ID_mapping.csv"
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
        gene_id_mapping_file=additional_mapping_file,
        print_level=print_level,
    )
