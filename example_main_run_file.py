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
    x_split = [split(sm) for sm in Smiles]
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
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i + 1))
        sequences_Example_i = sequences_Example[i]
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
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


if __name__ == '__main__':
    combinations_output_location = r"C:\Git\Metabolic_Task_Score\Data\Main_files\For_running\combinations\HumanGem17_DCM_test_metabolic_tasks_2024_v1_01"
    # todo add checks that this runs correctly and all files exist etc.
    SMILES_df = pd.read_csv(
        os.path.join(combinations_output_location, "final_SMILES_metabolite_df.csv"))
    sequence_df = pd.read_csv(
        os.path.join(combinations_output_location, "final_transcript_sequence_df.csv"))
    # gene_smiles_reactions_pairs.json
    with open(os.path.join(combinations_output_location, "gene_smiles_reactions_pairs.json"), "r") as f:
        gene_smiles_reactions_dict = json.load(f)

    n = 20
    combinations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    ]

    duplicate_smiles_tensor = False
    duplicate_seq_tensor = True

    # for duplicate_smiles_tensor, duplicate_seq_tensor in combinations:
    for i in range(1):
        duplicate_smiles_tensor = False
        duplicate_seq_tensor = True
        smiles = [
            'CC(C)C1=CC=C(C=C1)C(=O)O'
        ]
        sequences = [
            'MEDIPDTSRPPLKYVKGIPLIKYFAEALESLQDFQAQPDDLLISTYPKSGTTWVSEILDMIYQDGDVEKCRRAPVFIRVPFLEFKA'
        ]
        smiles_vec = smiles_to_vec(smiles)
        seq_vec = Seq_to_vec(sequences)
        if duplicate_smiles_tensor:
            smiles_vec = np.array([smiles_vec[0].copy() for i in range(n)])
            smiles = [smiles[0] for i in range(n)]
        if duplicate_seq_tensor:
            seq_vec = np.array([seq_vec[0].copy() for i in range(n)])
            sequences = [sequences[0] for i in range(n)]

        fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)

        ###### you should place downloaded model into this directory.
        # For kcat
        with open('UniKP20kcat.pkl', "rb") as f:
            model = pickle.load(f)
        # For Km
        # with open('UniKP/UniKP for Km.pkl', "rb") as f:
        #     model = pickle.load(f)
        # For kcat/Km
        # with open('UniKP/UniKP for kcat_Km.pkl', "rb") as f:
        #     model = pickle.load(f)

        # TODO ask Kiki how she managed to get rid of the issue as I cannot seem
            # to get rid fo the problem with it saying that nested tensor and batch effect
            # should not be true/false respectively
        # also results are different when using the double nested approach for the smiles, which is weird
        # turning on batch effects removes this issue, but changes the output

        Pre_label = model.predict(fused_vector)
        res = pd.DataFrame(
            {'sequences': sequences, 'Smiles': smiles, 'Pre_label': Pre_label})
        settings = f"smiles_tensor_{duplicate_smiles_tensor}_seq_tensor_{duplicate_seq_tensor}"
        res.to_csv(f'{settings}.csv', index=False)
