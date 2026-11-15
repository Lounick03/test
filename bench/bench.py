"""Banc d'essai v2 : précision de CLIP ViT-B/32 (le modèle de l'app) sur de vraies photos de plats Food-101,
en classant parmi TOUS les aliments de l'app (+ arrière-plans).

Deux jeux d'images distincts :
  - DEV  : images du split « validation » de Food-101, sur lesquelles les réglages ont été examinés ;
  - TEST : images du split « train » de Food-101, jamais regardées pendant les réglages (chiffres à retenir).
Compare : formulations de requêtes (seules / en ensemble) x descriptions d'aliments (v1 d'origine / v2 corrigée).
"""
import io, json, sys, time
import numpy as np
import torch
from datasets import load_dataset, Image as HFImage
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

N_DEV, N_TEST = 24, 40
MODEL = 'openai/clip-vit-base-patch32'

MAP = {
    'chicken_curry': ['Curry de poulet (lait de coco)', 'Poulet tikka masala'],
    'dumplings': ['Raviolis chinois (vapeur)'], 'gyoza': ['Raviolis chinois (vapeur)'],
    'edamame': ['Edamame cuits'], 'falafel': ['Falafel'], 'french_fries': ['Frites'],
    'fried_rice': ['Riz sauté (œuf, légumes)'], 'guacamole': ['Guacamole'], 'hamburger': ['Hamburger (fast-food)'],
    'hummus': ['Houmous'], 'ice_cream': ['Glace (crème glacée)'], 'lasagna': ['Lasagnes'],
    'miso_soup': ['Soupe miso'], 'pad_thai': ['Pad thaï'], 'pizza': ['Pizza (margherita)'],
    'samosa': ['Samoussa (viande)'], 'spaghetti_bolognese': ['Spaghetti bolognaise'],
    'spring_rolls': ['Nems (rouleau frit)', 'Rouleaux de printemps (frais)'], 'sushi': ['Sushi / maki'],
    'grilled_salmon': ['Saumon cuit'], 'pork_chop': ['Porc cuit (filet, côte)'], 'steak': ['Bœuf grillé (steak, rumsteck)'],
}
TEMPLATES = {
    'T0': 'a photo of {}, a type of food.',
    'T1': 'a photo of {}.',
    'T2': 'a close-up photo of {} on a plate.',
    'T3': 'a photo of a dish of {}.',
    'T4': 'a top-down photo of {}, food photography.',
    'T5': 'a photo of {}, a type of food, served on a plate.',
}
CONFIGS = {
    'T0 seul (version actuelle)': ['T0'],
    'T2 seul': ['T2'],
    'T5 seul': ['T5'],
    'T2+T5': ['T2', 'T5'],
    'T0+T2+T5': ['T0', 'T2', 'T5'],
    'T0+T2+T3+T5': ['T0', 'T2', 'T3', 'T5'],
    'T0+T2+T4+T5': ['T0', 'T2', 'T4', 'T5'],
}

cls = json.load(open('bench/classes.json', encoding='utf-8'))
v2 = json.load(open('bench/desc_v2.json', encoding='utf-8'))
foods = cls['foods']; bg = cls['bg']
names = [f['name'] for f in foods]
DESCS = {
    'v1': [f['desc'] for f in foods] + bg,
    'v2': [v2.get(f['name'], f['desc']) for f in foods] + bg,
}
N = len(names) + len(bg)
ALL_NAMES = names + bg

torch.set_grad_enabled(False)
model = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

def text_emb(template_ids, descs):
    out = []
    for i in range(0, N, 64):
        chunk = descs[i:i + 64]
        per_t = []
        for t in template_ids:
            tok = proc(text=[TEMPLATES[t].format(d) for d in chunk], return_tensors='pt', padding=True, truncation=True)
            e = model.text_projection(model.text_model(**tok).pooler_output)
            per_t.append(e / e.norm(dim=-1, keepdim=True))
        m = torch.stack(per_t).mean(0)
        out.append(m / m.norm(dim=-1, keepdim=True))
    return torch.cat(out)

def collect(split, per_class):
    want = {k: [] for k in MAP}
    ds = load_dataset('ethz/food101', split=split, streaming=True).cast_column('image', HFImage(decode=False))
    labels = ds.features['label'].names
    seen = 0
    for ex in ds:
        seen += 1
        lab = labels[ex['label']]
        if lab in want and len(want[lab]) < per_class:
            want[lab].append(ex['image']['bytes'])
        if all(len(v) >= per_class for v in want.values()):
            break
    print('%s : %d images parcourues, %d retenues' % (split, seen, sum(len(v) for v in want.values())), flush=True)
    feats, truth = [], []
    for lab, lst in want.items():
        for b in lst:
            img = Image.open(io.BytesIO(b)).convert('RGB')
            inp = proc(images=img, return_tensors='pt')
            e = model.visual_projection(model.vision_model(pixel_values=inp['pixel_values']).pooler_output)
            feats.append((e / e.norm(dim=-1, keepdim=True))[0])
            truth.append({names.index(n) for n in MAP[lab]})
    return torch.stack(feats), truth

SETS = {'DEV': collect('validation', N_DEV), 'TEST': collect('train', N_TEST)}

def evaluate(img, truth, txt):
    order = (100 * img @ txt.T).argsort(dim=1, descending=True).tolist()
    t1 = np.array([o[0] in tr for o, tr in zip(order, truth)])
    t3 = np.array([any(x in tr for x in o[:3]) for o, tr in zip(order, truth)])
    return t1, t3

R = {}   # (jeu, version, config) -> (top1[], top3[])
for ver in ('v1', 'v2'):
    for cname, ts in CONFIGS.items():
        txt = text_emb(ts, DESCS[ver])
        for sname, (img, truth) in SETS.items():
            R[(sname, ver, cname)] = evaluate(img, truth, txt)

def pct(a): return 100 * a.mean()
print('\n===== RÉSULTATS =====')
for sname in ('DEV', 'TEST'):
    n = len(SETS[sname][1])
    print('\n--- %s : %d images, 22 classes, classement parmi %d requêtes (aliments + arrière-plans) ---' % (sname, n, N))
    print('%-30s %-3s %7s %7s' % ('configuration', 'desc', 'top-1', 'top-3'))
    rows = [(k, v) for k, v in R.items() if k[0] == sname]
    for (s, ver, c), (t1, t3) in sorted(rows, key=lambda kv: -kv[1][0].mean()):
        print('%-30s %-3s %6.1f%% %6.1f%%' % (c, ver, pct(t1), pct(t3)))

# choix de la configuration sur DEV uniquement, puis comparaison à la version actuelle sur TEST (bootstrap apparié)
cand = [(k[1], k[2]) for k in R if k[0] == 'DEV']
best = max(cand, key=lambda vc: (R[('DEV',) + vc][0].mean() + R[('DEV',) + vc][1].mean()))
base = ('v1', 'T0 seul (version actuelle)')
print('\nConfiguration retenue sur DEV (top-1 + top-3) :', best)
rng = np.random.default_rng(0)
for k, name in ((0, 'top-1'), (1, 'top-3')):
    a = R[('TEST',) + best][k].astype(float); b = R[('TEST',) + base][k].astype(float)
    d = a - b
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print('TEST %s : version actuelle %.1f%% -> retenue %.1f%%  (écart %+.1f pts, IC 95 %% [%+.1f ; %+.1f])' % (name, 100 * b.mean(), 100 * a.mean(), 100 * d.mean(), 100 * lo, 100 * hi))

# erreurs de la configuration retenue sur TEST, par classe
img, truth = SETS['TEST']
txt = text_emb(CONFIGS[best[1]], DESCS[best[0]])
pred = (100 * img @ txt.T).argmax(dim=1).tolist()
txt0 = text_emb(CONFIGS[base[1]], DESCS[base[0]])
pred0 = (100 * img @ txt0.T).argmax(dim=1).tolist()
print('\nTEST, par classe (retenue / actuelle) :')
idx = 0
for lab, ex in MAP.items():
    ok = ok0 = 0; conf = {}
    for _ in range(N_TEST):
        if pred[idx] in truth[idx]: ok += 1
        else: conf[ALL_NAMES[pred[idx]]] = conf.get(ALL_NAMES[pred[idx]], 0) + 1
        if pred0[idx] in truth[idx]: ok0 += 1
        idx += 1
    top = sorted(conf.items(), key=lambda x: -x[1])[:2]
    print('  %-20s %2d/%d  (actuelle %2d)  confusions: %s' % (lab, ok, N_TEST, ok0, ', '.join('%s×%d' % c for c in top)))

out = {'best': list(best), 'base': list(base)}
for sname in ('DEV', 'TEST'):
    out[sname] = {'%s | %s' % (k[1], k[2]): [float(v[0].mean()), float(v[1].mean())] for k, v in R.items() if k[0] == sname}
json.dump(out, open('bench/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nFIN')
