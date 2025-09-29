import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
import pickle
import json
import os


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    # trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    # Modified by JELLE BONTHUIS on 2025-03-20 as the original code was not working
        # for GPU only
    trfm.load_state_dict(
        torch.load('trfm_12_23000.pkl', map_location=torch.device('cpu')))
    trfm.eval()
    try:
        x_split = []
        for idx, sm in enumerate(Smiles):
            if sm == 'nan' or isinstance(sm, float):
                x_split.append("")
                continue
            x_split.append(split(sm))
        # x_split = [split(sm) for sm in Smiles]
    except Exception as e:
        print(f"Error: {e}")
        return None

    xid, xseg = get_array(x_split)
    print(f"shape of xid: {xid.shape}")
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    ###### you should place downloaded model into this directory.
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i + 1))
        sequences_Example_i = sequences_Example[i]
        try:
            sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
            ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True,
                                              padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][:seq_len - 1]
                features.append(seq_emd)
        except Exception as e:
            print(f'Error: for sequence number {i}: {sequences_Example_i}\n', e)

    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


if __name__ == '__main__':
    # todo add checks that this runs correctly and all files exist etc.
    # todo make this production-grade code

    #### IMPORTANT ####
    # todo clean up, comment code, integrate into SWaPAM pipeline
    # todo rename files to represent what they do (gene-smiles-reaction pairs do not have SMILES)
    #    AA and smiles sequences.py do not do either , but only create json files with gene-metabolite pairs
    # This script requires specific scikit version which only works (verified with) python 3.8

    # In addition, it requires three files to be available:
    #   - final_SMILES_metabolite_df.csv (contains the metabolites with SMILES)
    #   - final_transcript_sequence_df.csv (contains the genes with sequences)
    #   - gene_smiles_reactions_pairs.json (contains the gene-metabolite pairs to infer kcats for)

    # These files can be created with some of the metabolic task score preprocessing scripts
    # and will be made available/adapted for SWaPAM > specifically the files necessary are:
    #  - smiles_getters.py (creates the final_SMILES_metabolite_df.csv)
    #  - transcript_getters.py (creates the final_transcript_sequence_df.csv)
    #  - preprocess_AA_and_SMILES_sequences.py (creates the gene_smiles_reactions_pairs.json)

    # Running this script will create the following files:
    #   - protein_sequence_tensors.pkl (contains the tensors for the sequences and can be reused)
    #   - SMILES_tensors.pkl (contains the tensors for the SMILES and can be reused)
    #   - per_gene_combination_results_*.json/csv (intermediate results, can be deleted after running)
    #   - final_per_gene_combination_results.json/csv (final results with statistics)
    #   - missing_genes_and_SMILES.csv (list of genes and SMILES that were not found in the input files)
    ####################
    
    ### Options and file paths
    # Multiple predictions are made by multiplying SMILES, later on the statistics are created
    # 50 seems to be fine, but can be adjusted (most computation time is not spend on this)
    amount_of_times_to_multiply_smiles = 50
    combinations_output_location = r"C:\Users\MACSBIO-metabolic\git\SWAMP\data\for_SWaPAM\combinations\model_inhouse_v7_DCM_test_metabolic_tasks_2024_v1_01"
    type_of_SMILES = 'isomeric SMILES' # seems to be what UniKP uses

    ### File loading
    try:
        SMILES_df = pd.read_csv(
            os.path.join(combinations_output_location, "final_SMILES_metabolite_df.csv"))
    except:
        SMILES_df = pd.read_csv(
            os.path.join(combinations_output_location, "final_SMILES_metabolite_df.csv"), sep =";")
    sequence_df = pd.read_csv(
        os.path.join(combinations_output_location, "final_transcript_sequence_df.csv"))
    if len(sequence_df.columns) <2:
        sequence_df = pd.read_csv(
            os.path.join(combinations_output_location, "final_transcript_sequence_df.csv"), ";")
    with open(os.path.join(combinations_output_location, "gene_smiles_reactions_pairs.json"), "r") as f:
        gene_smiles_reactions_dict = json.load(f)


    # Create or load tensors
    sequences = sequence_df["protein_sequence"].values.tolist()
    sequences = sequences
    if not os.path.exists(os.path.join(combinations_output_location, f"protein_sequence_tensors.pkl")):
        print("Creating protein sequence tensors")
        seq_vec = Seq_to_vec(sequences)
        with open(os.path.join(combinations_output_location, f"protein_sequence_tensors.pkl"),
                "wb") as f:
            pickle.dump(seq_vec, f)
    else:
        with open(os.path.join(combinations_output_location, f"protein_sequence_tensors.pkl"),
                "rb") as f:
            seq_vec = pickle.load(f)

    # create duplicates of the sequence tensors if needed
    seq_vec = np.array([seq.copy() for seq in seq_vec for i in range(amount_of_times_to_multiply_smiles)])
    smiles_ = SMILES_df[type_of_SMILES].values.tolist()
    smiles_unique = []
    for smiles in smiles_:
        if smiles not in smiles_unique:
            smiles_unique.append(smiles)

    smiles_unique = smiles_unique
    smiles = [item for item in smiles_unique for i in range(amount_of_times_to_multiply_smiles)]
    if not os.path.exists(os.path.join(combinations_output_location, f"SMILES_tensors.pkl")):
        print("Creating SMILES tensors")
        smiles_vec = smiles_to_vec(smiles)
        with open(os.path.join(combinations_output_location, f"SMILES_tensors.pkl"),
                "wb") as f:
            pickle.dump(smiles_vec, f)
    else:
        with open(os.path.join(combinations_output_location, f"SMILES_tensors.pkl"),
                "rb") as f:
            smiles_vec = pickle.load(f)

    unique_smiles_positions = {smiles: (i*amount_of_times_to_multiply_smiles, ((i + 1)* amount_of_times_to_multiply_smiles)) for i, smiles in enumerate(smiles_unique)}
    unique_sequences_positions = {sequence: (i*amount_of_times_to_multiply_smiles, ((i + 1)* amount_of_times_to_multiply_smiles)) for i, sequence in enumerate(sequences)}

    not_in_sequence_df = []
    not_in_smiles_df = []

    fused_vectors = np.array([])
    second_dim = seq_vec.shape[1]
    chunk_amount = 200
    total_size = len(gene_smiles_reactions_dict)
    amount_of_things_to_check = total_size * amount_of_times_to_multiply_smiles
    # set chunk size to be the minimally a full amount_of_times_to_multiply_smiles
    chunk_size = amount_of_things_to_check // chunk_amount
    if chunk_size % amount_of_times_to_multiply_smiles != 0:
        chunk_size = (chunk_size // amount_of_times_to_multiply_smiles) * amount_of_times_to_multiply_smiles
        chunk_amount += 1
    with open('UniKP20kcat.pkl', "rb") as f:
        model = pickle.load(f)

    final_per_gene_combination_results = {}
    for chunk in range(chunk_amount):
        print(f"Chunk: {chunk}")
        start = chunk * chunk_size // amount_of_times_to_multiply_smiles
        end = (chunk + 1) * chunk_size // amount_of_times_to_multiply_smiles
        seq_tensors_large = np.ndarray((chunk_size, second_dim))
        smiles_tensors_large = np.ndarray((chunk_size, second_dim))
        for idx, (key, value) in enumerate(list(gene_smiles_reactions_dict.items())[start:end]):
            gene_id = value[0]
            metabolite = value[1]
            if gene_id not in sequence_df["ensemble_id"].values:
                print(f"gene not in sequence_df: {gene_id}")
                not_in_sequence_df.append(gene_id)
                continue
            gene_index_in_seq_df = sequence_df[sequence_df["ensemble_id"] == gene_id].index[0]
            if metabolite not in SMILES_df["id"].values:
                print(f"smiles not in SMILES_df: {metabolite}")
                not_in_smiles_df.append(metabolite)
                continue
            smiles_index_in_smiles_df = SMILES_df[SMILES_df["id"] ==metabolite].index[0]
            smiles = SMILES_df[SMILES_df["id"] == metabolite][type_of_SMILES].values[0]
            gene = sequence_df[sequence_df["ensemble_id"] == gene_id]["protein_sequence"].values[0]
            # now use the unique positions to grab the correct tensors
            gene_seq_tensor = seq_vec[gene_index_in_seq_df*amount_of_times_to_multiply_smiles:((gene_index_in_seq_df + 1)*amount_of_times_to_multiply_smiles)]
            smiles_unique_positions = unique_smiles_positions[smiles]
            smiles_tensor = smiles_vec[smiles_unique_positions[0]:smiles_unique_positions[1]]

            seq_tensors_large[idx*amount_of_times_to_multiply_smiles:((idx + 1)*amount_of_times_to_multiply_smiles)] = gene_seq_tensor
            smiles_tensors_large[idx*amount_of_times_to_multiply_smiles:((idx + 1)*amount_of_times_to_multiply_smiles)] = smiles_tensor

        fused_vectors = np.concatenate((smiles_tensors_large, seq_tensors_large), axis=1)
        Pre_label = model.predict(fused_vectors)
        per_gene_combination_results = {}
        for idx, value in enumerate(list(gene_smiles_reactions_dict.values())[start:end]):
            # in sets of amounht_of_times_to_multiply_smiles
            log_results = Pre_label[
                          idx*amount_of_times_to_multiply_smiles:
                          ((idx + 1)*amount_of_times_to_multiply_smiles)
                          ]
            # since data is in log scale, we need to convert it back to normal scale for statistics
            # then convert those back to log scale
            linear_results = np.array([10**x for x in log_results])

            dict_ = {}
            dict_["min"] = np.log10(np.min(linear_results)) if np.min(linear_results) > 0 else 0
            dict_["max"] = np.log10(np.max(linear_results)) if np.max(linear_results) > 0 else 0
            dict_["median"] = np.log10(np.median(linear_results)) if np.median(linear_results) > 0 else 0
            dict_["mean"] = np.log10(np.mean(linear_results)) if np.mean(linear_results) > 0 else 0

            dict_["iqr"] = np.percentile(log_results, 75) - np.percentile(log_results, 25)
            dict_["sd"] = np.std(log_results)
            dict_["sd_as_percent_of_mean"] = dict_["sd"] / np.mean(log_results) if np.mean(log_results) != 0 else 0
            dict_["ensemble_id"] = value[0]
            dict_["metabolite_id"] = value[1]

            new_key = (value[0], value[1])
            per_gene_combination_results[new_key] = dict_

        # save temporary results
        per_gene_combination_results = {str(key): value for key, value in per_gene_combination_results.items()}
        with open(os.path.join(combinations_output_location, f"per_gene_combination_results_{chunk}.json"), "w") as f:
            json.dump(per_gene_combination_results, f)
        # create df
        df = pd.DataFrame(per_gene_combination_results).T
        df.to_csv(os.path.join(combinations_output_location, f"per_gene_combination_results_{chunk}.csv"))

        final_per_gene_combination_results.update(per_gene_combination_results)

    final_per_gene_combination_results = {str(key): value for key, value in final_per_gene_combination_results.items()}
    with open(os.path.join(combinations_output_location, "final_per_gene_combination_results.json"), "w") as f:
        json.dump(final_per_gene_combination_results , f)


    unique_not_in_sequence_df = list(set(not_in_sequence_df))
    unique_not_in_smiles_df = list(set(not_in_smiles_df))

    df = pd.DataFrame(final_per_gene_combination_results).T
    df["missing"] = False
    df["smiles_longer_than_218"] = False
    for idx, value in df.iterrows():
        if value["ensemble_id"] in unique_not_in_sequence_df or value["metabolite_id"] in unique_not_in_smiles_df:
            df.at[idx, "missing"] = True
        smiles = SMILES_df[SMILES_df["id"] == value["metabolite_id"]][type_of_SMILES].values
        if len(smiles) == 0:
            continue
        else:
            smiles = smiles[0]
        if not isinstance(smiles, float) and len(smiles.split()) > 218:
            df.at[idx, "smiles_longer_than_218"] = True

    df.to_csv(os.path.join(combinations_output_location, "final_per_gene_combination_results.csv"))
    #make same length
    if len(unique_not_in_sequence_df) < len(unique_not_in_smiles_df):
        unique_not_in_sequence_df.extend([""]*(len(unique_not_in_smiles_df) - len(unique_not_in_sequence_df)))
    else:
        unique_not_in_smiles_df.extend([""]*(len(unique_not_in_sequence_df) - len(unique_not_in_smiles_df)))

    missing_df = pd.DataFrame({"missing_genes": unique_not_in_sequence_df, "missing_SMILES": unique_not_in_smiles_df})
    missing_df.to_csv(os.path.join(combinations_output_location, "missing_genes_and_SMILES.csv"))

