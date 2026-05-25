# Converted from frame_level.ipynb

# 1. Pipeline up to step2 — needed for stim indices
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
pipeline.step1_load_dutch30_data(patient_range=(21, 30))
pipeline.step2_split_by_instances(train_fraction=0.8)
pipeline.step3_load_channel_exclusions('channel_exclusions.json') 
pipeline.apply_channel_exclusions()

# (skip step3-5; no collapsed features)

# 2. Build minimal saved_models from MFA + your manner table
from run_pipeline import load_mfa_alignments
from collections import Counter
import numpy as np

MANNER = {
    # ---- Vowels (manner=0) ----
    # short
    'ɪ': 0, 'ɛ': 0, 'ɑ': 0, 'ɔ': 0, 'ʏ': 0, 'ə': 0, 'a': 0,
    # long / tense
    'i':  0, 'iː': 0, 'eː': 0, 'aː': 0, 'oː': 0,
    'uː': 0, 'yː': 0, 'øː': 0, 'u': 0, 'y': 0, 'e': 0, 'o': 0, 'ɔ̈':  0, 'ɛ̈':  0,
    # diphthongs
    'ɛi': 0, 'ɛj': 0, 'œy': 0, 'ɔu': 0, 'ɑu': 0, 'ɑi': 0, 'au': 0,
    'ui': 0, 'oi': 0, 'œ':  0,

    # ---- Stops / plosives (manner=1) ----
    'p': 1, 'b': 1, 't': 1, 'd': 1, 'k': 1, 'g': 1, 'ɡ': 1,
    'ʔ': 1, 'c':  1,   # palatal stop

    # ---- Fricatives (manner=2) ----
    'f': 2, 'v': 2, 's': 2, 'z': 2, 'x': 2, 'ɣ': 2, 'h': 2,
    'ʃ': 2, 'ʒ': 2, 'ç': 2, 'χ': 2,

    # ---- Nasals (manner=3) ----
    'm': 3, 'n': 3, 'ŋ': 3, 'ɲ': 3,

    # ---- Approximants / liquids (manner=4) ----
    'l': 4, 'r': 4, 'ʁ': 4, 'ɹ': 4, 'j': 4, 'ʋ': 4, 'w': 4, 'ɥ':  4,

    # ---- Affricates (treated as stops for manner) ----
    'ts': 1, 'tʃ': 1, 'dʒ': 1,
}

PLACE = {
    # ---- All vowels = 0 ----
    'ɪ': 0, 'ɛ': 0, 'ɑ': 0, 'ɔ': 0, 'ʏ': 0, 'ə': 0, 'a': 0,
    'i':  0, 'iː': 0, 'eː': 0, 'aː': 0, 'oː': 0,
    'uː': 0, 'yː': 0, 'øː': 0, 'u': 0, 'y': 0, 'e': 0, 'o': 0,
    'ɛi': 0, 'ɛj': 0, 'œy': 0, 'ɔu': 0, 'ɑu': 0, 'ɑi': 0, 'au': 0,
    'ui': 0, 'oi': 0, 'œ':  0, 'ɔ̈':  0, 'ɛ̈':  0,

    # ---- Bilabial (1): p b m ----
    'p': 1, 'b': 1, 'm': 1,

    # ---- Labiodental (2): f v ʋ ----
    'f': 2, 'v': 2, 'ʋ': 2, 'w': 2,

    # ---- Alveolar (3): t d s z n l r ts ----
    't': 3, 'd': 3, 's': 3, 'z': 3, 'n': 3, 'l': 3,
    'r': 3, 'ɹ': 3, 'ts': 3,

    # ---- Postalveolar (4): ʃ ʒ tʃ dʒ ----
    'ʃ': 4, 'ʒ': 4, 'tʃ': 4, 'dʒ': 4,

    # ---- Velar (5): k g x ɣ ŋ ----
    'k': 5, 'g': 5, 'ɡ': 5, 'x': 5, 'ɣ': 5, 'ŋ': 5,
    'χ': 5, 'ʁ': 5,

    # ---- Glottal (6): h ʔ ----
    'h': 6, 'ʔ': 6,

    # ---- Palatal (7): j ɲ ç ----
    'j': 7, 'ɲ': 7, 'ç': 7, 'c':  7, 'ɥ':  7,
}


def build_saved_state(pid, pipeline):
    wd = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    train_sent_ids = set()
    if 'train' in pipeline.split_result:
        for inst in pipeline.split_result['train'].get(pid, []):
            if isinstance(inst, dict) and 'sentence_idx' in inst:
                train_sent_ids.add(inst['sentence_idx'])
    if not train_sent_ids:
        all_real = [i for i, s in enumerate(wd['sentence_list'])
                    if isinstance(s, dict) and s.get('text')]
        train_sent_ids = set(i for i in all_real if i not in all_real[::6])

    # phoneme inventory from train sentences only (leak-safe)
    phons = []
    for sidx in train_sent_ids:
        if sidx in mfa:
            phons.extend(p['phone'] for p in mfa[sidx])
    inv = sorted(set(phons))
    cls_to_i = {ph: i for i, ph in enumerate(inv)}

    # bigram log-probs from train
    n = len(inv)
    bg = np.ones((n, n), dtype=np.float32)   # Laplace smoothing
    for sidx in train_sent_ids:
        if sidx not in mfa: continue
        seq = [cls_to_i[p['phone']] for p in mfa[sidx]
               if p['phone'] in cls_to_i]
        for a, b in zip(seq[:-1], seq[1:]):
            bg[a, b] += 1
    bg_lp = np.log(bg / bg.sum(axis=1, keepdims=True))

    # phone_to_manner array
    pm_arr = np.array([MANNER.get(ph, 0) for ph in inv], dtype=int)

    return {'cls_to_i': cls_to_i, 'bg_lp': bg_lp,
            'phone_to_manner': pm_arr}

# ---- Coverage check against the actual phonemes in your MFA output ----
def check_coverage(saved_models, manner=MANNER, place=PLACE):
    missing_manner = set()
    missing_place = set()
    for pid, st in saved_models.items():
        for ph in st['cls_to_i'].keys():
            if ph not in manner: missing_manner.add(ph)
            if ph not in place: missing_place.add(ph)
    if missing_manner:
        print(f"MISSING from MANNER: {sorted(missing_manner)}")
    if missing_place:
        print(f"MISSING from PLACE: {sorted(missing_place)}")
    if not missing_manner and not missing_place:
        print("All phonemes covered.")

saved_models = {pid: build_saved_state(pid, pipeline)
                for pid in ['P22', 'P23', 'P26', 'P29']}

check_coverage(saved_models) 

# # Load speech vs non speech detector
# import torch.nn as nn

# class SpeechDetector(nn.Module):
#     """Per-patient speech-vs-non-speech model."""
#     def __init__(self, n_in, lstm_hidden=128, lstm_layers=2, dropout=0.2):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(n_in, lstm_hidden * 2), nn.GELU(), nn.Dropout(dropout))
#         self.lstm = nn.LSTM(lstm_hidden * 2, lstm_hidden,
#                             num_layers=lstm_layers,
#                             dropout=dropout if lstm_layers > 1 else 0.0,
#                             bidirectional=True, batch_first=False)
#         self.head = nn.Linear(lstm_hidden * 2, 2)

#     def forward(self, x):
#         h = self.proj(x).unsqueeze(1)
#         h, _ = self.lstm(h)
#         return self.head(h.squeeze(1))


# class CrossPatientSpeechDetector(nn.Module):
#     """Per-patient input projection feeds into a shared BiLSTM + head."""
#     def __init__(self, n_in_per_pid, embed_dim=128,
#                  lstm_hidden=128, lstm_layers=2, dropout=0.2):
#         super().__init__()
#         self.projs = nn.ModuleDict({
#             pid: nn.Sequential(nn.Linear(n_in, embed_dim), nn.GELU(),
#                                nn.Dropout(dropout))
#             for pid, n_in in n_in_per_pid.items()
#         })
#         self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=lstm_layers,
#                             dropout=dropout if lstm_layers > 1 else 0.0,
#                             bidirectional=True, batch_first=False)
#         self.head = nn.Linear(lstm_hidden * 2, 2)

#     def forward(self, x, pid):
#         h = self.projs[pid](x).unsqueeze(1)
#         h, _ = self.lstm(h)
#         return self.head(h.squeeze(1))
        
# ckpt = torch.load('bio_models/speech_detector_cross_patient.pt',
#                   map_location=DEVICE, weights_only=False)
# sd_model = CrossPatientSpeechDetector(
#     n_in_per_pid=ckpt['n_in_per_pid'],
#     **ckpt['arch']).to(DEVICE)
# sd_model.load_state_dict(ckpt['state_dict'])
# sd_model.eval()
# # remember per-patient mu/sd for standardizing the input
# mu_sd_speech = ckpt['mu_sd']

# ============================================================
# Path B v3 — joint dataset: keep all frames, two label streams
# ============================================================
import os, numpy as np, torch
from collections import defaultdict, Counter
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

PRE_ONSET_MS     = 200
PRE_ONSET_FRAMES = int(PRE_ONSET_MS / SHIFT_MS)

def build_speech_label_full(mfa_phones, n_frames):
    """Per-frame binary speech label (with pre-onset extension)."""
    label = np.zeros(n_frames, dtype=np.int64)
    for ph in mfa_phones:
        k_start = int(np.ceil(ph['start_s']  * FRAME_HZ))
        k_end   = int(np.floor(ph['end_s']   * FRAME_HZ))
        k_pre   = max(0, k_start - PRE_ONSET_FRAMES)
        k_start = max(0, k_start); k_end = min(n_frames - 1, k_end)
        if k_end < k_pre: continue
        label[k_pre:k_end + 1] = 1
    return label

def build_joint_dataset_v3(pid, pipeline, channel_mask=None):
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    if channel_mask is not None:
        raw_eeg = raw_eeg[:, channel_mask]

    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[::6])

    manner_map = get_manner_table(saved_models, pid)

    sents = {'train': [], 'test': []}
    n_used = n_skip = 0
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]:
            n_skip += 1; continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: n_skip += 1; continue

        X = extract_hg_frames(raw_eeg[s0:s1])
        if X.shape[0] < 30: n_skip += 1; continue
        Xs = stack_context(X, K=5)

        # Two label streams
        speech_full = build_speech_label_full(mfa[sent_idx], X.shape[0])
        bio, phon, keep = build_speech_only_labels(mfa[sent_idx], X.shape[0])
        if keep.sum() < 11:
            n_skip += 1; continue

        bio_idx = np.where(keep)[0]
        bio_arr = np.array(bio)[bio_idx]
        phon_arr = np.array(phon)[bio_idx]
        manner_arr = np.array([manner_map.get(p, -1) for p in phon_arr], dtype=np.int64)
        place_arr  = np.array([PLACE_TABLE.get(p, -1) for p in phon_arr], dtype=np.int64)

        split = 'test' if sent_idx in test_sent_ids else 'train'
        sents[split].append({
            'X':        torch.from_numpy(Xs).float(),
            'speech':   torch.from_numpy(speech_full),
            'bio_idx':  torch.from_numpy(bio_idx).long(),
            'bio_strs': bio_arr,                          # for OOV filter + tag lookup
            'manner':   torch.from_numpy(manner_arr),
            'place':    torch.from_numpy(place_arr),
            'sent_idx': sent_idx,
        })
        n_used += 1

    n_in = sents['train'][0]['X'].shape[1] if sents['train'] else 0
    print(f"  [{pid}] used={n_used} skipped={n_skip}  n_in={n_in}  "
          f"train={len(sents['train'])} test={len(sents['test'])}")
    return sents


TARGET_PIDS = ['P22', 'P23', 'P26', 'P29']
joint_datasets = {}
for pid in TARGET_PIDS:
    print(f"\nBuilding {pid}...")
    joint_datasets[pid] = build_joint_dataset_v3(pid, pipeline)

#BiLSTM emission backbone + CRF (no O)

class JointBiLSTM_BIO_CRF(nn.Module):
    """Same as BiLSTM_BIO_CRF but with an additional speech_head trained on
       ALL frames. BIO/manner/place heads only see frames inside phoneme
       intervals (selected via bio_idx)."""
    def __init__(self, n_in, n_phon, n_manner=5, n_place=8,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 transition_mask=None, transition_init=None, n_tags=None):
        super().__init__()
        self.n_tags = n_tags
        self.proj = nn.Sequential(
            nn.Linear(n_in, lstm_hidden * 2), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=True, batch_first=False)
        self.drop = nn.Dropout(dropout)
        self.bio_head     = nn.Linear(lstm_hidden * 2, n_tags)
        self.bio_aux_head = nn.Linear(lstm_hidden * 2, n_tags)
        self.manner_head  = nn.Linear(lstm_hidden * 2, n_manner)
        self.place_head   = nn.Linear(lstm_hidden * 2, n_place)
        self.speech_head  = nn.Linear(lstm_hidden * 2, 2)        # NEW
        self.crf = LinearChainCRF(n_tags, transition_mask, transition_init)

    def encode(self, x):
        h, _ = self.lstm(self.proj(x).unsqueeze(1))
        return self.drop(h.squeeze(1))                            # (T_full, 2H)

    def loss(self, x_full, speech_labels, bio_idx, tags, manner, place,
             lam_speech=0.5, lam_bio_ce=0.5, lam_manner=0.3, lam_place=0.1,
             ce_weights=None, speech_weights=None):
        h_full   = self.encode(x_full)                            # (T_full, 2H)
        speech_l = F.cross_entropy(self.speech_head(h_full),
                                   speech_labels, weight=speech_weights)

        h_bio = h_full[bio_idx]                                   # (T_bio, 2H)
        bio_em = self.bio_head(h_bio)
        crf_nll = self.crf.neg_log_likelihood(bio_em, tags) / max(h_bio.size(0), 1)
        bio_ce = F.cross_entropy(self.bio_aux_head(h_bio), tags, weight=ce_weights)

        valid_m = manner >= 0
        mn_loss = (F.cross_entropy(self.manner_head(h_bio)[valid_m], manner[valid_m])
                   if valid_m.sum() > 0 else torch.tensor(0., device=x_full.device))
        valid_p = place >= 0
        pl_loss = (F.cross_entropy(self.place_head(h_bio)[valid_p], place[valid_p])
                   if valid_p.sum() > 0 else torch.tensor(0., device=x_full.device))

        total = (crf_nll + lam_bio_ce * bio_ce + lam_manner * mn_loss
                 + lam_place * pl_loss + lam_speech * speech_l)
        return total, {
            'crf':    float(crf_nll.item()),
            'bio_ce': float(bio_ce.item()),
            'mn':     float(mn_loss.item()),
            'pl':     float(pl_loss.item()),
            'speech': float(speech_l.item()),
        }

    @torch.no_grad()
    def decode(self, x_full, bio_idx):
        """At inference, return Viterbi path over the speech frames
           (selected via bio_idx — gold speech mask in eval, predicted at deploy)."""
        h_full = self.encode(x_full)
        h_bio  = h_full[bio_idx]
        return self.crf.viterbi(self.bio_head(h_bio))

    @torch.no_grad()
    def predict_speech(self, x_full):
        """Per-frame speech probability for deployment use."""
        return F.softmax(self.speech_head(self.encode(x_full)), dim=-1)


def make_joint_model(pid, joint_datasets, saved_models,
                     lstm_hidden=128, lstm_layers=2, dropout=0.3):
    fd = joint_datasets[pid]
    state = saved_models[pid]
    cls_to_i = state['cls_to_i']
    bg_lp = state['bg_lp']
    pm_arr = np.asarray(state['phone_to_manner']).astype(int)
    n_in = fd['train'][0]['X'].shape[1]
    n_phon = len(cls_to_i)
    n_manner = int(pm_arr.max()) + 1
    n_place = 8

    tag_to_idx, idx_to_tag, n_tags = build_tag_index_noO(cls_to_i)
    trans_mask = build_transition_mask_noO(idx_to_tag)
    trans_init = init_transition_matrix_noO(idx_to_tag, bg_lp, cls_to_i)

    model = JointBiLSTM_BIO_CRF(
        n_in=n_in, n_phon=n_phon, n_manner=n_manner, n_place=n_place,
        lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout,
        transition_mask=trans_mask, transition_init=trans_init, n_tags=n_tags)
    return model, tag_to_idx, idx_to_tag


# Quick sanity check
for pid in TARGET_PIDS:
    if pid not in joint_datasets: continue
    m, _, _ = make_joint_model(pid, joint_datasets, saved_models)
    print(f"{pid}: params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  n_tags={m.n_tags}")

# # GPU training + mid-phoneme mixup

# import time
# import torch.optim as optim

# EPOCHS         = 40
# LR             = 2.5e-4
# WEIGHT_DECAY   = 1e-3
# EARLY_STOP_PATIENCE = 6      # stop if no improvement for 8 epochs
# LAM_MANNER     = 0.3
# LAM_PLACE      = 0.1
# LAM_BIO_CE     = 0.5
# B_BOOST        = 0.0      # O is gone; usually unneeded
# GRAD_CLIP      = 5.0
# MIN_SENT_FRAMES = 30

# # augmentation
# P_AUG          = 0.5      # prob of augmenting a sentence in a step
# AUG_FRAC       = 0.2      # fraction of mid-phoneme frames perturbed
# MIX_RATIO      = 0.8      # anchor weight
# MIN_PHON_LEN_FOR_AUG = 6  # frames; phoneme must be at least this long


# def split_into_sentences_v2(split_dict, tag_to_idx):
#     sents = []
#     sidx = split_dict['sent_idx']
#     boundaries = np.where(np.diff(sidx, prepend=sidx[0] - 1) != 0)[0].tolist() \
#                  + [len(sidx)]
#     n_oov_total = 0
#     for k in range(len(boundaries) - 1):
#         s, e = boundaries[k], boundaries[k + 1]
#         if e - s < MIN_SENT_FRAMES: continue
#         bio_str_full = split_dict['bio'][s:e]
#         # filter frames whose tag is OOV (test phoneme not in train vocab)
#         keep_local = np.array([t in tag_to_idx for t in bio_str_full])
#         n_oov_total += int((~keep_local).sum())
#         if keep_local.sum() < MIN_SENT_FRAMES: continue

#         bio_str = bio_str_full[keep_local]
#         tags = np.array([tag_to_idx[t] for t in bio_str], dtype=np.int64)

#         # mid_phoneme mask (same as before, on the kept slice)
#         mid_mask = np.zeros(len(tags), dtype=bool)
#         i = 0
#         while i < len(bio_str):
#             if bio_str[i].startswith('B-'):
#                 ph = bio_str[i][2:]
#                 j = i + 1
#                 while j < len(bio_str) and bio_str[j] == f'I-{ph}':
#                     j += 1
#                 if (j - i) >= MIN_PHON_LEN_FOR_AUG:
#                     margin = 2
#                     mid_mask[i + margin: j - margin] = True
#                 i = j
#             else:
#                 i += 1

#         X_full      = split_dict['X'][s:e]
#         manner_full = split_dict['manner'][s:e]
#         place_full  = split_dict['place'][s:e]
#         sents.append({
#             'X':      torch.from_numpy(X_full[keep_local]).float(),
#             'tags':   torch.from_numpy(tags),
#             'manner': torch.from_numpy(manner_full[keep_local]),
#             'place':  torch.from_numpy(place_full[keep_local]),
#             'mid':    torch.from_numpy(mid_mask),
#             'sent_idx': int(sidx[s]),
#         })
#     if n_oov_total > 0:
#         print(f"  dropped {n_oov_total} frames with OOV phonemes")
#     return sents

# def fit_mu_sd(sents):
#     Xall = torch.cat([s['X'] for s in sents], dim=0).numpy()
#     return Xall.mean(0), Xall.std(0)

# def standardize(sents, mu, sd):
#     sd_safe = np.where(sd < 1e-6, 1.0, sd)
#     mu_t = torch.from_numpy(mu).float()
#     sd_t = torch.from_numpy(sd_safe).float()
#     for s in sents:
#         s['X'] = (s['X'] - mu_t) / sd_t

# def to_device(sents, device):
#     for s in sents:
#         for k in ('X', 'tags', 'manner', 'place', 'mid'):
#             s[k] = s[k].to(device)

# def make_ce_weights(sents_tr, n_tags, device):
#     """Class weights for the BIO direct-CE loss: inverse frequency, clamped."""
#     cnt = torch.zeros(n_tags)
#     for s in sents_tr:
#         for t in s['tags'].cpu().tolist():
#             cnt[t] += 1
#     cnt = cnt.clamp(min=1.0)
#     w = (cnt.sum() / (n_tags * cnt))
#     w = w.clamp(min=0.2, max=5.0)
#     return w.to(device)

# def augment_sentence(s_anchor, partner_pool, p_aug, aug_frac, mix_ratio, rng):
#     """Returns (X_maybe_augmented, tags_unchanged)."""
#     if rng.random() > p_aug or s_anchor['mid'].sum() == 0:
#         return s_anchor['X']
#     mid_idx = s_anchor['mid'].nonzero(as_tuple=False).flatten()
#     n_perturb = max(1, int(len(mid_idx) * aug_frac))
#     perm = mid_idx[torch.randperm(len(mid_idx), device=mid_idx.device)[:n_perturb]]
#     partner = partner_pool[rng.integers(0, len(partner_pool))]
#     # pick random frames from partner (any frames, length-matched via wrap)
#     if partner['X'].size(0) == 0:
#         return s_anchor['X']
#     p_idx = torch.randint(0, partner['X'].size(0), (n_perturb,),
#                           device=partner['X'].device)
#     X = s_anchor['X'].clone()
#     X[perm] = mix_ratio * X[perm] + (1.0 - mix_ratio) * partner['X'][p_idx]
#     return X

# def evaluate_for_selection(model, sents_te, idx_to_tag):
#     model.eval()
#     correct = total = 0
#     pred_symbols = set()
#     n_pred = 0
#     with torch.no_grad():
#         for s in sents_te:
#             path = model.decode(s['X'])
#             tg = s['tags'].tolist()
#             for p, t in zip(path, tg):
#                 correct += (p == t); total += 1
#             for p in path:
#                 tag = idx_to_tag[int(p)]
#                 if tag.startswith('B-'):
#                     pred_symbols.add(tag[2:])
#                     n_pred += 1
#     tag_acc = correct / max(total, 1)
#     # Relaxed: 3 unique phonemes → factor 1.0 (was 5)
#     diversity = min(1.0, len(pred_symbols) / 3.0)
#     # Relaxed: 0.5× gold coverage already counts as full
#     coverage_ratio = min(1.0, n_pred / max(1, total / 34))   # 34 = 2× phoneme len
#     composite = tag_acc * diversity * coverage_ratio
#     return composite, tag_acc, len(pred_symbols), n_pred

# def train_v2(pid, frame_datasets_v2, saved_models, epochs=EPOCHS, **arch):
#     fd = frame_datasets_v2[pid]
#     if fd['train'] is None or fd['test'] is None:
#         return None
#     model, tag_to_idx, idx_to_tag = make_model_v2(
#         pid, frame_datasets_v2, saved_models, **arch)
#     model = model.to(DEVICE)

#     sents_tr = split_into_sentences_v2(fd['train'], tag_to_idx)
#     sents_te = split_into_sentences_v2(fd['test'],  tag_to_idx)
#     if not sents_tr:
#         return None
#     mu, sd = fit_mu_sd(sents_tr)
#     standardize(sents_tr, mu, sd); standardize(sents_te, mu, sd)
#     to_device(sents_tr, DEVICE); to_device(sents_te, DEVICE)

#     ce_weights = make_ce_weights(sents_tr, model.n_tags, DEVICE)

#     opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#     sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

#     rng = np.random.default_rng(0)
#     best_acc, best_state = 0.0, None
#     no_improve = 0
#     stopped_early = False

#     for ep in range(epochs):
#         model.train()
#         perm = rng.permutation(len(sents_tr))
#         losses = {'total': 0, 'crf': 0, 'bio_ce': 0, 'mn': 0, 'pl': 0}
#         for idx in perm:
#             s = sents_tr[idx]
#             X_aug = augment_sentence(s, sents_tr, P_AUG, AUG_FRAC, MIX_RATIO, rng)
#             opt.zero_grad()
#             loss, parts = model.loss(
#                 X_aug, s['tags'], s['manner'], s['place'],
#                 lam_manner=LAM_MANNER, lam_place=LAM_PLACE,
#                 lam_bio_ce=LAM_BIO_CE, b_boost=B_BOOST,
#                 ce_weights=ce_weights)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#             opt.step()
#             losses['total'] += float(loss.item())
#             for k in ('crf', 'bio_ce', 'mn', 'pl'):
#                 losses[k] += parts[k]
#         sched.step()

#         if (ep + 1) % 5 == 0 or ep == 0:
#             score, tag_acc, n_uniq, n_pred = evaluate_for_selection(
#                 model, sents_te, idx_to_tag)
#             if score > best_acc and ep >= 5:
#                 best_acc = score
#                 best_state = {k: v.detach().cpu().clone()
#                               for k, v in model.state_dict().items()}
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             n = len(sents_tr)
#             print(f"  [{pid}] ep{ep+1:3d} "
#                   f"loss={losses['total']/n:6.2f} crf={losses['crf']/n:5.2f} "
#                   f"bio_ce={losses['bio_ce']/n:5.2f} "
#                   f"mn={losses['mn']/n:.2f} pl={losses['pl']/n:.2f}  "
#                   f"score={score:.3f} tag_acc={tag_acc:.3f} "
#                   f"n_uniq={n_uniq} n_pred={n_pred} "
#                   f"(best_score={best_acc:.3f})")

#             if no_improve >= EARLY_STOP_PATIENCE:
#                 print(f"  [{pid}] early stopping at ep {ep+1}")
#                 stopped_early = True
#                 break

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     # Save to disk for inspection
#     import os, pickle
#     os.makedirs('bio_models', exist_ok=True)
#     save_path = f'bio_models/{pid}_biocrf_v2.pkl'
#     with open(save_path, 'wb') as f:
#         pickle.dump({
#             'state_dict': model.state_dict(),
#             'mu': mu, 'sd': sd,
#             'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
#             'arch': dict(lstm_hidden=128, lstm_layers=2,
#                          lstm_dropout=0.3, head_dropout=0.3),
#             'best_acc': best_acc,
#             'pid': pid,
#         }, f)
#     print(f"  [{pid}] saved to {save_path}")

#     return {
#         'model': model, 'mu': mu, 'sd': sd,
#         'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
#         'sents_tr': sents_tr, 'sents_te': sents_te,
#         'best_acc': best_acc,
#     }
    
# bio_results_v2 = {}
# for pid in TARGET_PIDS:
#     if pid not in frame_datasets_v2 or frame_datasets_v2[pid]['train'] is None:
#         continue
#     print(f"\n=== Training {pid} ===")
#     t0 = time.time()
#     bio_results_v2[pid] = train_v2(
#         pid, frame_datasets_v2, saved_models, epochs=EPOCHS,
#         lstm_hidden=128, lstm_layers=2, lstm_dropout=0.3, head_dropout=0.3)
#     if bio_results_v2[pid]:
#         print(f"  [{pid}] done in {time.time() - t0:.1f}s")

# ============================================================
# Joint training
# Make sure: joint_datasets is populated, JointBiLSTM_BIO_CRF is defined,
# build_tag_index_noO / build_transition_mask_noO / init_transition_matrix_noO
# / LinearChainCRF are defined (from the v2 cells), and make_joint_model exists.
# ============================================================
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

EPOCHS              = 40
LR                  = 3e-4
WEIGHT_DECAY        = 1e-3
LAM_SPEECH          = 0.5
LAM_MANNER          = 0.3
LAM_PLACE           = 0.1
LAM_BIO_CE          = 0.5
EARLY_STOP_PATIENCE = 6
MIN_SENT_FRAMES     = 30


def prepare_joint_sentences(sents, tag_to_idx):
    out = []
    n_oov = 0
    for s in sents:
        if s['X'].size(0) < MIN_SENT_FRAMES: continue
        bio_strs = s['bio_strs']
        keep_local = np.array([t in tag_to_idx for t in bio_strs])
        n_oov += int((~keep_local).sum())
        if keep_local.sum() < 11: continue
        bio_idx_filt = s['bio_idx'][torch.from_numpy(keep_local)]
        tags = np.array([tag_to_idx[t] for t in bio_strs[keep_local]],
                        dtype=np.int64)
        out.append({
            'X':        s['X'],
            'speech':   s['speech'],
            'bio_idx':  bio_idx_filt,
            'tags':     torch.from_numpy(tags),
            'manner':   s['manner'][torch.from_numpy(keep_local)],
            'place':    s['place'][torch.from_numpy(keep_local)],
            'sent_idx': s['sent_idx'],
        })
    if n_oov: print(f"  dropped {n_oov} OOV-tag frames")
    return out


def fit_mu_sd_joint(sents):
    Xall = torch.cat([s['X'].cpu() for s in sents], dim=0).numpy()
    return Xall.mean(0), Xall.std(0)

def standardize_joint(sents, mu, sd):
    sd_safe = np.where(sd < 1e-6, 1.0, sd)
    for s in sents:
        device = s['X'].device
        mu_t = torch.from_numpy(mu).float().to(device)
        sd_t = torch.from_numpy(sd_safe).float().to(device)
        s['X'] = (s['X'] - mu_t) / sd_t

def to_device_joint(sents, device):
    for s in sents:
        for k in ('X', 'speech', 'bio_idx', 'tags', 'manner', 'place'):
            s[k] = s[k].to(device)


def make_bio_ce_weights(sents_tr, n_tags, device):
    cnt = torch.zeros(n_tags)
    for s in sents_tr:
        for t in s['tags'].cpu().tolist(): cnt[t] += 1
    cnt = cnt.clamp(min=1.0)
    w = (cnt.sum() / (n_tags * cnt)).clamp(min=0.2, max=5.0)
    return w.to(device)

def make_speech_weights(sents_tr, device):
    all_sp = torch.cat([s['speech'].cpu() for s in sents_tr])
    n0 = (all_sp == 0).sum().item(); n1 = (all_sp == 1).sum().item()
    return torch.tensor([n1/(n0+n1), n0/(n0+n1)],
                        dtype=torch.float32, device=device)


def evaluate_joint(model, sents_te, idx_to_tag):
    model.eval()
    correct = total = 0
    pred_symbols = set(); n_pred = 0
    sp_correct = sp_total = 0
    with torch.no_grad():
        for s in sents_te:
            sp_pred = model.predict_speech(s['X']).argmax(-1)
            sp_correct += (sp_pred == s['speech']).sum().item()
            sp_total   += s['speech'].numel()

            path = model.decode(s['X'], s['bio_idx'])
            tg = s['tags'].tolist()
            for p, t in zip(path, tg):
                correct += (p == t); total += 1
            for p in path:
                tag = idx_to_tag[int(p)]
                if tag.startswith('B-'):
                    pred_symbols.add(tag[2:]); n_pred += 1
    tag_acc = correct / max(total, 1)
    sp_acc  = sp_correct / max(sp_total, 1)
    diversity = min(1.0, len(pred_symbols) / 3.0)
    coverage  = min(1.0, n_pred / max(1, total / 34))
    composite = tag_acc * diversity * coverage
    return {'composite': composite, 'tag_acc': tag_acc, 'speech_acc': sp_acc,
            'n_uniq': len(pred_symbols), 'n_pred': n_pred}


def train_joint(pid, joint_datasets, saved_models, epochs=EPOCHS, **arch):
    model, tag_to_idx, idx_to_tag = make_joint_model(
        pid, joint_datasets, saved_models, **arch)
    model = model.to(DEVICE)

    sents_tr = prepare_joint_sentences(joint_datasets[pid]['train'], tag_to_idx)
    sents_te = prepare_joint_sentences(joint_datasets[pid]['test'],  tag_to_idx)
    if not sents_tr: return None
    mu, sd = fit_mu_sd_joint(sents_tr)
    standardize_joint(sents_tr, mu, sd); standardize_joint(sents_te, mu, sd)
    to_device_joint(sents_tr, DEVICE);   to_device_joint(sents_te, DEVICE)

    ce_w = make_bio_ce_weights(sents_tr, model.n_tags, DEVICE)
    sp_w = make_speech_weights(sents_tr, DEVICE)
    print(f"  [{pid}] speech weights: nonsp={sp_w[0]:.3f} sp={sp_w[1]:.3f}")

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_score, best_state, no_improve = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr))
        losses = {'total': 0, 'crf': 0, 'bio_ce': 0, 'mn': 0, 'pl': 0, 'speech': 0}
        for idx in perm:
            s = sents_tr[idx]
            opt.zero_grad()
            loss, parts = model.loss(
                s['X'], s['speech'], s['bio_idx'],
                s['tags'], s['manner'], s['place'],
                lam_speech=LAM_SPEECH, lam_bio_ce=LAM_BIO_CE,
                lam_manner=LAM_MANNER, lam_place=LAM_PLACE,
                ce_weights=ce_w, speech_weights=sp_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses['total'] += float(loss.item())
            for k in ('crf', 'bio_ce', 'mn', 'pl', 'speech'):
                losses[k] += parts[k]
        sched.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            ev = evaluate_joint(model, sents_te, idx_to_tag)
            if ev['composite'] > best_score and ep >= 2:
                best_score = ev['composite']
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            n = len(sents_tr)
            print(f"  [{pid}] ep{ep+1:3d}  "
                  f"crf={losses['crf']/n:.2f} bio_ce={losses['bio_ce']/n:.2f} "
                  f"sp={losses['speech']/n:.2f}  "
                  f"score={ev['composite']:.3f} tag={ev['tag_acc']:.3f} "
                  f"speech={ev['speech_acc']:.3f} "
                  f"n_uniq={ev['n_uniq']} (best={best_score:.3f})")
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  [{pid}] early stop at ep {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    import os, pickle
    os.makedirs('bio_models', exist_ok=True)
    save_path = f'bio_models/{pid}_biocrf_joint_v3.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'state_dict': model.state_dict(),
            'mu': mu, 'sd': sd,
            'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'arch': dict(lstm_hidden=128, lstm_layers=2, dropout=0.3),
            'best_score': best_score,
            'pid': pid,
        }, f)
    print(f"  [{pid}] saved to {save_path}")

    return {'model': model, 'mu': mu, 'sd': sd,
            'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'sents_tr': sents_tr, 'sents_te': sents_te,
            'best_score': best_score}


# Actually run joint training
joint_results = {}
for pid in TARGET_PIDS:
    if pid not in joint_datasets: continue
    print(f"\n=== Joint training {pid} ===")
    t0 = time.time()
    joint_results[pid] = train_joint(
        pid, joint_datasets, saved_models, epochs=EPOCHS,
        lstm_hidden=128, lstm_layers=2, dropout=0.3)
    if joint_results[pid]:
        print(f"  [{pid}] done in {time.time() - t0:.1f}s")

# ============================================================
# Joint v3 — larger model
# ============================================================
EPOCHS              = 50          # was 40
LAM_SPEECH_BIG      = 0.3         # was 0.5 — less capacity to speech task
LR_BIG              = 2e-4        # was 3e-4 — bigger model, more careful
WEIGHT_DECAY_BIG    = 2e-3        # was 1e-3 — more regularization

def train_joint_big(pid, joint_datasets, saved_models, epochs=EPOCHS,
                    lstm_hidden=256, lstm_layers=3, dropout=0.4):
    """Same as train_joint but with bigger model + retuned regularization."""
    model, tag_to_idx, idx_to_tag = make_joint_model(
        pid, joint_datasets, saved_models,
        lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout)
    model = model.to(DEVICE)
    print(f"  [{pid}] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    sents_tr = prepare_joint_sentences(joint_datasets[pid]['train'], tag_to_idx)
    sents_te = prepare_joint_sentences(joint_datasets[pid]['test'],  tag_to_idx)
    if not sents_tr: return None
    mu, sd = fit_mu_sd_joint(sents_tr)
    standardize_joint(sents_tr, mu, sd); standardize_joint(sents_te, mu, sd)
    to_device_joint(sents_tr, DEVICE);   to_device_joint(sents_te, DEVICE)

    ce_w = make_bio_ce_weights(sents_tr, model.n_tags, DEVICE)
    sp_w = make_speech_weights(sents_tr, DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR_BIG,
                            weight_decay=WEIGHT_DECAY_BIG)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_score, best_state, no_improve = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr))
        losses = {'total': 0, 'crf': 0, 'bio_ce': 0, 'mn': 0, 'pl': 0, 'speech': 0}
        for idx in perm:
            s = sents_tr[idx]
            opt.zero_grad()
            loss, parts = model.loss(
                s['X'], s['speech'], s['bio_idx'],
                s['tags'], s['manner'], s['place'],
                lam_speech=LAM_SPEECH_BIG, lam_bio_ce=LAM_BIO_CE,
                lam_manner=LAM_MANNER, lam_place=LAM_PLACE,
                ce_weights=ce_w, speech_weights=sp_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses['total'] += float(loss.item())
            for k in ('crf', 'bio_ce', 'mn', 'pl', 'speech'):
                losses[k] += parts[k]
        sched.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            ev = evaluate_joint(model, sents_te, idx_to_tag)
            if ev['composite'] > best_score and ep >= 2:
                best_score = ev['composite']
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            n = len(sents_tr)
            print(f"  [{pid}] ep{ep+1:3d}  "
                  f"crf={losses['crf']/n:.2f} bio_ce={losses['bio_ce']/n:.2f} "
                  f"sp={losses['speech']/n:.2f}  "
                  f"score={ev['composite']:.3f} tag={ev['tag_acc']:.3f} "
                  f"speech={ev['speech_acc']:.3f} "
                  f"n_uniq={ev['n_uniq']} (best={best_score:.3f})")
            if no_improve >= 6:
                print(f"  [{pid}] early stop at ep {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    import os, pickle
    os.makedirs('bio_models', exist_ok=True)
    save_path = f'bio_models/{pid}_biocrf_joint_v3_big.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'state_dict': model.state_dict(),
            'mu': mu, 'sd': sd,
            'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'arch': dict(lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                         dropout=dropout),
            'best_score': best_score,
            'pid': pid,
        }, f)
    print(f"  [{pid}] saved to {save_path}")

    return {'model': model, 'mu': mu, 'sd': sd,
            'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'sents_tr': sents_tr, 'sents_te': sents_te,
            'best_score': best_score}


# Run on all four patients
joint_big_results = {}
for pid in TARGET_PIDS:
    if pid not in joint_datasets: continue
    print(f"\n=== Joint BIG training {pid} ===")
    t0 = time.time()
    joint_big_results[pid] = train_joint_big(
        pid, joint_datasets, saved_models, epochs=EPOCHS,
        lstm_hidden=256, lstm_layers=3, dropout=0.4)
    if joint_big_results[pid]:
        print(f"  [{pid}] done in {time.time() - t0:.1f}s")

# Score the big-joint results
pipeline.patient_results = {}
for pid in TARGET_PIDS:
    if pid not in joint_big_results or joint_big_results[pid] is None: continue
    pipeline.patient_results[pid] = joint_results_to_patient_results(pid, joint_big_results)

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    pr = pipeline.patient_results[pid]
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pmask = pr['pred_sentence_ids'] == sid
        gmask = pr['true_sentence_ids'] == sid
        pred_sents.append(list(pr['predictions'][pmask]))
        gold_sents.append(list(pr['true_labels'][gmask]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    print(f"\n=== {pid} ===")
    score_run(pred_sents, gold_sents, f"{pid} joint big", gold_lp)

# Are joint helpers defined?
print('train_joint:', 'train_joint' in dir())
print('JointBiLSTM_BIO_CRF:', 'JointBiLSTM_BIO_CRF' in dir())
print('joint_datasets:', 'joint_datasets' in dir())
print('joint_results:', 'joint_results' in dir())

# Score joint results with surprise z
for pid in TARGET_PIDS:
    if pid not in joint_results or joint_results[pid] is None: continue
    pipeline.patient_results[pid] = joint_results_to_patient_results(pid, joint_results)

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    pr = pipeline.patient_results[pid]
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pmask = pr['pred_sentence_ids'] == sid
        gmask = pr['true_sentence_ids'] == sid
        pred_sents.append(list(pr['predictions'][pmask]))
        gold_sents.append(list(pr['true_labels'][gmask]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    print(f"\n=== {pid} ===")
    score_run(pred_sents, gold_sents, f"{pid} joint v3", gold_lp)

def joint_results_to_patient_results(pid, joint_results, frame_shift_s=FRAME_SHIFT_S):
    res = joint_results[pid]
    model = res['model']; model.eval()
    idx_to_tag = res['idx_to_tag']
    sents_te = res['sents_te']

    true_labels, predictions = [], []
    true_sentence_ids, pred_sentence_ids = [], []
    true_segments, pred_segments = [], []

    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'], s['bio_idx'])         # <-- two args
            pred_tag_strs = [idx_to_tag[int(t)] for t in path]
            gold_tag_strs = [idx_to_tag[int(t)] for t in s['tags'].tolist()]
            pp, ps, pe = collapse_bio_to_segments(pred_tag_strs)
            gp, gs, ge = collapse_bio_to_segments(gold_tag_strs)
            sid = s['sent_idx']
            for ph, fa, fb in zip(pp, ps, pe):
                predictions.append(ph)
                pred_sentence_ids.append(sid)
                pred_segments.append((fa * frame_shift_s, fb * frame_shift_s))
            for ph, fa, fb in zip(gp, gs, ge):
                true_labels.append(ph)
                true_sentence_ids.append(sid)
                true_segments.append((fa * frame_shift_s, fb * frame_shift_s))

    from e2e_brain_decoder import edit_distance
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    acc = float((true_arr[:min(len(true_arr), len(pred_arr))]
                 == pred_arr[:min(len(true_arr), len(pred_arr))]).mean()) \
          if len(true_arr) and len(pred_arr) else 0.0
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'accuracy': acc, 'edit_distance': ed, 'per': per,
        'n_test': len(true_labels), 'n_pred': len(predictions),
    }


# Build patient_results and score
if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}
for pid in joint_results:
    if joint_results[pid] is None: continue
    pipeline.patient_results[pid] = joint_results_to_patient_results(pid, joint_results)
    
# Score
for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    pr = pipeline.patient_results[pid]
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pmask = pr['pred_sentence_ids'] == sid
        gmask = pr['true_sentence_ids'] == sid
        pred_sents.append(list(pr['predictions'][pmask]))
        gold_sents.append(list(pr['true_labels'][gmask]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    print(f"\n=== {pid} ===")
    score_run(pred_sents, gold_sents, f"{pid} joint", gold_lp)

# ============================================================
# Adapter: BIO-CRF results → pipeline.patient_results format
# ============================================================
from e2e_brain_decoder import show_matched_sequences_with_times

FRAME_SHIFT_S = 5e-3   # 200 Hz

def collapse_bio_to_segments(bio_tag_strs):
    """Given a list of BIO tag strings (no O in v2), return parallel lists
    of (phoneme_symbol, start_frame_idx, end_frame_idx_exclusive)."""
    phons, starts, ends = [], [], []
    i = 0
    n = len(bio_tag_strs)
    while i < n:
        t = bio_tag_strs[i]
        if t.startswith('B-'):
            ph = t[2:]
            j = i + 1
            while j < n and bio_tag_strs[j] == f'I-{ph}':
                j += 1
            phons.append(ph); starts.append(i); ends.append(j)
            i = j
        elif t.startswith('I-'):
            # orphan I (shouldn't happen with structural mask, but be defensive)
            ph = t[2:]
            j = i + 1
            while j < n and bio_tag_strs[j] == f'I-{ph}':
                j += 1
            phons.append(ph); starts.append(i); ends.append(j)
            i = j
        else:
            i += 1
    return phons, starts, ends


def bio_results_to_patient_results(pid, bio_results_v2,
                                    sd_model=None, mu_sd_speech=None,
                                    frame_shift_s=FRAME_SHIFT_S):
    """Build the dict structure that show_matched_sequences_with_times reads.
       If sd_model and mu_sd_speech are provided, gate phoneme predictions
       by the speech detector: only keep segments whose majority of frames
       are flagged as speech."""
    res = bio_results_v2[pid]
    model = res['model']; model.eval()
    idx_to_tag = res['idx_to_tag']
    sents_te = res['sents_te']

    gating = sd_model is not None and mu_sd_speech is not None
    if gating:
        sd_model.eval()
        mu_bio = torch.from_numpy(np.asarray(res['mu'])).float().to(DEVICE)
        sd_bio = torch.from_numpy(
            np.where(np.asarray(res['sd']) < 1e-6, 1.0, np.asarray(res['sd']))
        ).float().to(DEVICE)
        mu_sp_np, sd_sp_np = mu_sd_speech[pid]
        mu_sp = torch.from_numpy(mu_sp_np).float().to(DEVICE)
        sd_sp = torch.from_numpy(
            np.where(sd_sp_np < 1e-6, 1.0, sd_sp_np)
        ).float().to(DEVICE)
        is_cross = hasattr(sd_model, 'projs')

    true_labels = []
    predictions = []
    true_sentence_ids = []
    pred_sentence_ids = []
    true_segments = []
    pred_segments = []
    n_dropped_total = 0
    n_kept_total = 0

    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'])
            pred_tag_strs = [idx_to_tag[int(t)] for t in path]
            gold_tag_strs = [idx_to_tag[int(t)] for t in s['tags'].tolist()]

            # Compute per-frame speech mask for this sentence (if gating)
            if gating:
                X_raw     = s['X'] * sd_bio + mu_bio          # un-standardize
                X_for_sd  = (X_raw - mu_sp) / sd_sp           # re-standardize
                if is_cross:
                    speech_logits = sd_model(X_for_sd, pid)
                else:
                    speech_logits = sd_model(X_for_sd)
                is_speech = (speech_logits.argmax(-1) == 1).cpu().numpy()
            else:
                is_speech = np.ones(len(path), dtype=bool)

            pp, ps, pe = collapse_bio_to_segments(pred_tag_strs)
            gp, gs, ge = collapse_bio_to_segments(gold_tag_strs)

            sid = s['sent_idx']
            for ph, fa, fb in zip(pp, ps, pe):
                # gate: drop predicted phoneme if majority of its frames are non-speech
                if gating:
                    span = is_speech[fa:fb]
                    if len(span) == 0 or span.mean() < 0.5:
                        n_dropped_total += 1
                        continue
                    n_kept_total += 1
                predictions.append(ph)
                pred_sentence_ids.append(sid)
                pred_segments.append((fa * frame_shift_s, fb * frame_shift_s))
            for ph, fa, fb in zip(gp, gs, ge):
                true_labels.append(ph)
                true_sentence_ids.append(sid)
                true_segments.append((fa * frame_shift_s, fb * frame_shift_s))

    if gating:
        print(f"  [{pid}] speech gating: kept {n_kept_total} segments, "
              f"dropped {n_dropped_total} (={100*n_dropped_total/max(n_kept_total+n_dropped_total,1):.1f}%)")

    from e2e_brain_decoder import edit_distance
    true_arr = np.array(true_labels)
    pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    acc = float((true_arr[:min(len(true_arr), len(pred_arr))]
                 == pred_arr[:min(len(true_arr), len(pred_arr))]).mean()) \
          if len(true_arr) and len(pred_arr) else 0.0

    return {
        'true_labels':       true_arr,
        'predictions':       pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments':     true_segments,
        'pred_segments':     pred_segments,
        'accuracy':          acc,
        'edit_distance':     ed,
        'per':               per,
        'n_test':            len(true_labels),
        'n_pred':            len(predictions),
    }

# Build sentence_texts dict for nicer display
def build_sentence_texts(pipeline, pid):
    wd = pipeline.split_result['word_segments_dict'][pid]
    out = {}
    for i, s in enumerate(wd['sentence_list']):
        if isinstance(s, dict) and s.get('text'):
            out[i] = s['text']
    return out


# Wire into pipeline.patient_results and call the viewer
if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

for pid in bio_results_v2:
    # pipeline.patient_results[pid] = bio_results_to_patient_results(pid, bio_results_v2) call without speech gating
    pipeline.patient_results[pid] = bio_results_to_patient_results(  # call with speech gating
    pid, bio_results_v2,
    sd_model=sd_model,
    mu_sd_speech=mu_sd_speech,
) 


import importlib, e2e_brain_decoder
importlib.reload(e2e_brain_decoder)
from e2e_brain_decoder import show_matched_sequences_with_times

# Now call the viewer
for pid in sorted(bio_results_v2.keys()):
    print(f"\n========== {pid} ==========")
    show_matched_sequences_with_times(
        pipeline, pid,
        max_per_line=30,
        # time_align_tol_s=0.05,         # 50 ms tolerance for column alignment
        collapse_repeats=False,         # BIO already segmented; don't collapse twice
    )

# Permutation of labels to get surprise-z + max_run scorer
from collections import Counter

N_PERM    = 2000
SHIFT_MAX = 3
MIN_MATCH = 3

def longest_run_with_shift(pred, gold, shift_max=SHIFT_MAX):
    best, best_span = 0, None
    P, G = len(pred), len(gold)
    if P == 0 or G == 0: return 0, None
    for i in range(P):
        for j in range(max(0, i - shift_max), min(G, i + shift_max + 1)):
            k = 0
            while i + k < P and j + k < G and pred[i + k] == gold[j + k]:
                k += 1
            if k > best:
                best, best_span = k, (i, j, k)
    return best, best_span

def collect_matches(pred_sents, gold_sents, min_match=MIN_MATCH):
    matches = []
    for p, g in zip(pred_sents, gold_sents):
        L, span = longest_run_with_shift(p, g)
        if L >= min_match and span is not None:
            i, j, k = span
            matches.append(tuple(p[i:i + k]))
    return matches

def surprise_score(matches, marginal_logp):
    fallback = -np.log(1e-6)
    return sum(-marginal_logp.get(ph, fallback) for m in matches for ph in m)

def perm_null(pred_sents, gold_sents, marginal_logp, n_perm=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    nulls = np.zeros(n_perm)
    for b in range(n_perm):
        shuf = []
        for p in pred_sents:
            if len(p) == 0:
                shuf.append(p); continue
            idx = rng.permutation(len(p))
            shuf.append([p[k] for k in idx])
        nulls[b] = surprise_score(collect_matches(shuf, gold_sents), marginal_logp)
    return nulls

def score_run(pred_sents, gold_sents, label, gold_marginal_logp):
    if not any(len(s) for s in pred_sents):
        print(f"[{label}] empty"); return None
    runs = [longest_run_with_shift(p, g)[0] for p, g in zip(pred_sents, gold_sents)]
    max_run = max(runs) if runs else 0
    matches = collect_matches(pred_sents, gold_sents)
    obs = surprise_score(matches, gold_marginal_logp)
    nulls = perm_null(pred_sents, gold_sents, gold_marginal_logp)
    mu, sd = nulls.mean(), nulls.std() + 1e-9
    z = (obs - mu) / sd
    print(f"[{label}] sents={len(pred_sents)} max_run={max_run} "
          f"n_matches(>={MIN_MATCH})={len(matches)} "
          f"obs={obs:.2f} null mu={mu:.2f} sd={sd:.2f}  z={z:+.2f}")
    return {"max_run": max_run, "z": z}

# Score gated vs ungated predictions
for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results:
        continue
    pr = pipeline.patient_results[pid]
    # split into per-sentence sequences
    pred_sents = []
    gold_sents = []
    for sid in np.unique(pr['true_sentence_ids']):
        pmask = pr['pred_sentence_ids'] == sid
        gmask = pr['true_sentence_ids'] == sid
        pred_sents.append(list(pr['predictions'][pmask]))
        gold_sents.append(list(pr['true_labels'][gmask]))

    # gold marginal
    all_gold = [ph for s in gold_sents for ph in s]
    from collections import Counter
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}

    print(f"\n=== {pid} ===")
    score_run(pred_sents, gold_sents, f"{pid} gated", gold_lp)

for pid in bio_results_v2:
    pipeline.patient_results[pid] = bio_results_to_patient_results(pid, bio_results_v2) 

# Now call the viewer
for pid in sorted(bio_results_v2.keys()):
    print(f"\n========== {pid} ==========")
    show_matched_sequences_with_times(
        pipeline, pid,
        max_per_line=30,
        time_align_tol_s=0.05,         # 50 ms tolerance for column alignment
        collapse_repeats=False,         # BIO already segmented; don't collapse twice
    )

# Score gated vs ungated predictions
for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results:
        continue
    pr = pipeline.patient_results[pid]
    # split into per-sentence sequences
    pred_sents = []
    gold_sents = []
    for sid in np.unique(pr['true_sentence_ids']):
        pmask = pr['pred_sentence_ids'] == sid
        gmask = pr['true_sentence_ids'] == sid
        pred_sents.append(list(pr['predictions'][pmask]))
        gold_sents.append(list(pr['true_labels'][gmask]))

    # gold marginal
    all_gold = [ph for s in gold_sents for ph in s]
    from collections import Counter
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}

    print(f"\n=== {pid} ===")
    score_run(pred_sents, gold_sents, f"{pid} gated", gold_lp)
