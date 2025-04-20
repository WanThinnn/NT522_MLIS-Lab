import tensorflow.compat.v1 as tf
# Disable eager to use K.function as in TF1.x
tf.disable_v2_behavior()
from MalGAN_utils import fgsm 
from sklearn.neighbors import NearestNeighbors
from tensorflow.compat.v1.keras import backend as K
import MalGAN_utils
from MalGAN_preprocess import preprocess
import numpy as np
import os



def gen_adv_samples(model, fn_list, pad_percent=0.1, step_size=0.001, thres=0.5):
    """
    Generate adversarial samples using FGSM and neighbor embedding search.
    """
    # Helper: search nearest neighbor in embedding space
    def emb_search(org, adv_emb, pad_idx, pad_len, neigh):
        out = org.copy()
        for idx in range(pad_idx, pad_idx + pad_len):
            target = adv_emb[idx].reshape(1, -1)
            best_idx = neigh.kneighbors(target, n_neighbors=1, return_distance=False)[0][0]
            out[0][idx] = best_idx
        return out

    max_len = int(model.input.shape[1])
    emb_layer = model.layers[1]
    emb_weight = emb_layer.get_weights()[0]
    # Build function: input -> embedding output
    inp2emb = K.function([model.input, K.learning_phase()], [emb_layer.output])

    # Build neighbor searcher with correct parameter
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(emb_weight)

    log = MalGAN_utils.logger()
    adv_samples = []
    
    for e, fn in enumerate(fn_list):
        if not os.path.exists(fn):
            print(f"[!] File not found: {fn}")
            continue
        # Preprocess file, get input sequence and original length
        inp, len_list = preprocess([fn], max_len)
        if not len_list:
            print(f"[!] Skipping missing or invalid file: {fn}")
            continue
        # Get embedding of input sequence
        inp_emb = np.squeeze(np.array(inp2emb([inp, 0])), 0)  # 0 for test mode

        pad_idx = len_list[0]
        pad_len = max(min(int(len_list[0] * pad_percent), max_len - pad_idx), 0)
        org_score = model.predict(inp)[0][0]

        loss, pred = float('nan'), float('nan')
        if pad_len > 0 and org_score < thres:
            adv_emb, gradient, loss = fgsm(model, inp_emb, pad_idx, pad_len, e, step_size)
            adv_seq = emb_search(inp, adv_emb[0], pad_idx, pad_len, neigh)
            pred = model.predict(adv_seq)[0][0]
            final_adv = adv_seq[0][: pad_idx + pad_len]
        else:
            final_adv = inp[0][:pad_idx]

        log.write(fn, org_score, pad_idx, pad_len, loss, pred)

        # Convert sequence back to bytes
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

    return adv_samples, log


