import argparse, os, glob, numpy as np, cv2, torch, logging
from utils.io import load_config, ensure_dir
from wsi.tiling import load_rgb, tile_image, make_thumbnail
from wsi.stain_norm import macenko_normalize
from models.encoders import build_encoder
import tifffile as tiff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    weights_path = cfg["encoder"]["weights_path"] or os.path.join(cfg["encoder"]["ckpt_root"], cfg["encoder"]["kind"], cfg["encoder"].get("ckpt_filename","pytorch_model.bin"))
    wd = cfg["run"]["workdir"]; ensure_dir(wd)
    log_dir = os.path.join(wd, "logs"); ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "extract.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    feats_dir = os.path.join(wd,"feats"); thumbs_dir=os.path.join(wd,"thumbs")
    ensure_dir(feats_dir); ensure_dir(thumbs_dir)

    glob_patterns = cfg["data"]["wsi_glob"]
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    paths=[]
    for pat in glob_patterns:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(paths)
    logging.info(f"[FOUND] {len(paths)} WSIs")
    enc = build_encoder(cfg["encoder"]["kind"], weights_path); enc.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); enc.to(dev)
    bs = int(cfg["encoder"]["batch_size"]); inp = int(cfg["encoder"]["input_size"]); D=int(cfg["encoder"]["embed_dim"])

    mean = np.array(cfg["encoder"].get("norm_mean", [0.5,0.5,0.5]), dtype="float32")
    std  = np.array(cfg["encoder"].get("norm_std",  [0.5,0.5,0.5]), dtype="float32")
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        out_npz = os.path.join(feats_dir, f"{name}.npz")
        out_thumb = os.path.join(thumbs_dir, f"{name}.png")
        if os.path.exists(out_npz):
            logging.info(f"[SKIP] {out_npz} already exists, resume mode")
            continue
        rgb = load_rgb(p, level=int(cfg["tiling"].get("read_level", 0)))
        tiles, coords = tile_image(rgb,
                                   patch_size=cfg["tiling"]["patch_size"],
                                   overlap=cfg["tiling"]["overlap"],
                                   min_tissue_percent=cfg["tiling"]["min_tissue_percent"],
                                   blur_var_th=cfg["tiling"]["blur_var_th"],
                                   max_tiles=cfg["tiling"]["max_tiles_per_wsi"])
        if cfg["stain"]["method"]=="macenko":
            tiles = [macenko_normalize(t) for t in tiles]
        th, scale = make_thumbnail(rgb, max_size=cfg["tiling"]["thumbnail_size"])
        cv2.imwrite(out_thumb, cv2.cvtColor(th, cv2.COLOR_RGB2BGR))

        def pre(img):
            x = cv2.resize(img, (inp, inp), interpolation=cv2.INTER_AREA).astype("float32")/255.0
            x = np.transpose(x,(2,0,1))
            x = (x - mean[:,None,None]) / std[:,None,None]
            return x

        embs = np.zeros((len(tiles), D), dtype="float32")
        batch=[]; pos=0
        for i,t in enumerate(tiles):
            batch.append(pre(t))
            if len(batch)==bs or i==len(tiles)-1:
                x = torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)
                with torch.no_grad():
                    f = enc(x).detach().cpu().numpy()
                embs[pos:pos+len(batch)] = f
                pos += len(batch); batch=[]
        np.savez_compressed(out_npz,
                            feats=embs, coords=np.array(coords, dtype="int32"),
                            tile_size=int(cfg["tiling"]["patch_size"]), thumb_scale=float(scale))
        logging.info(f"[SAVE] {out_npz} (tiles={len(tiles)})")

if __name__=="__main__":
    main()
