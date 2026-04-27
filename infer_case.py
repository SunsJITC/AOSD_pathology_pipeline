import argparse, os, json, numpy as np, torch, cv2, pandas as pd
from joblib import load
from utils.io import load_config, ensure_dir
from wsi.tiling import load_rgb, tile_image, make_thumbnail
from wsi.stain_norm import macenko_normalize
from wsi.heatmap import render_attention_heatmap
from models.encoders import build_encoder
from models.mil import build_mil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wsi_path", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights_path = cfg["encoder"]["weights_path"] or os.path.join(cfg["encoder"]["ckpt_root"], cfg["encoder"]["kind"], cfg["encoder"].get("ckpt_filename","pytorch_model.bin"))
    wd = cfg["run"]["workdir"]; ensure_dir(wd)
    out_dir = os.path.join(wd,"infer"); ensure_dir(out_dir)

    scaler = load(os.path.join(wd,"models","scaler.joblib"))
    pca    = load(os.path.join(wd,"models","pca.joblib"))
    cox    = load(os.path.join(wd,"models","cox.joblib"))
    cutoff = json.load(open(os.path.join(wd,"models","cutoff.json")))["cutoff"]
    feat_dim = int(cfg["encoder"]["embed_dim"])
    mil = build_mil(cfg["mil"].get("arch","gated_attn"), in_dim=feat_dim, hidden=cfg["mil"]["hidden"], dropout=cfg["mil"]["dropout"])
    mil.load_state_dict(torch.load(os.path.join(wd,"models","attn_mil.pt"), map_location="cpu"))
    mil.eval()

    enc = build_encoder(cfg["encoder"]["kind"], weights_path); enc.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); enc.to(dev)

    mean = np.array(cfg["encoder"].get("norm_mean", [0.5,0.5,0.5]), dtype="float32")
    std  = np.array(cfg["encoder"].get("norm_std",  [0.5,0.5,0.5]), dtype="float32")
    wsi_path = args.wsi_path; name = os.path.splitext(os.path.basename(wsi_path))[0]
    rgb = load_rgb(wsi_path, level=int(cfg["tiling"].get("read_level", 0)))
    tiles, coords = tile_image(rgb,
                               patch_size=cfg["tiling"]["patch_size"],
                               overlap=cfg["tiling"]["overlap"],
                               min_tissue_percent=cfg["tiling"]["min_tissue_percent"],
                               blur_var_th=cfg["tiling"]["blur_var_th"],
                               max_tiles=cfg["tiling"]["max_tiles_per_wsi"])
    if cfg["stain"]["method"]=="macenko":
        tiles=[macenko_normalize(t) for t in tiles]
    thumb, scale = make_thumbnail(rgb, max_size=cfg["tiling"]["thumbnail_size"])

    inp = int(cfg["encoder"]["input_size"]); bs=int(cfg["encoder"]["batch_size"]); D=int(cfg["encoder"]["embed_dim"])
    def pre(img):
        x = cv2.resize(img, (inp,inp), interpolation=cv2.INTER_AREA).astype("float32")/255.0
        x = np.transpose(x,(2,0,1))
        x = (x - mean[:,None,None]) / std[:,None,None]
        return x
    if len(tiles) == 0:
        raise RuntimeError("No valid tissue tiles found for the provided WSI path.")
    embs = np.zeros((len(tiles), D), dtype="float32")
    pos=0; batch=[]
    for i,t in enumerate(tiles):
        batch.append(pre(t))
        if len(batch)==bs or i==len(tiles)-1:
            x = torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)
            with torch.no_grad():
                f = enc(x).detach().cpu().numpy()
            embs[pos:pos+len(batch)] = f; pos += len(batch); batch=[]

    with torch.no_grad():
        z, w = mil(torch.tensor(embs, dtype=torch.float32))
    z = z.numpy(); w=w.numpy()

    z_std = scaler.transform(z.reshape(1, -1))
    z_pca = pca.transform(z_std)
    row = pd.DataFrame(z_pca)
    risk = float(cox.predict_partial_hazard(row).values[0])
    label = "High-risk" if risk >= cutoff else "Low-risk"

    w_norm = (w - w.min())/(w.max()-w.min()+1e-6)
    heat_path = os.path.join(out_dir, f"{name}_attn.png")
    render_attention_heatmap(thumb, coords, cfg["tiling"]["patch_size"], scale, w_norm, heat_path)

    out = {"ID": name, "risk": risk, "cutoff": cutoff, "class": label, "attn_heatmap": heat_path}
    with open(os.path.join(out_dir, f"{name}_report.json"),"w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__=="__main__":
    main()
