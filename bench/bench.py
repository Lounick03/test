"""Banc d'essai : précision de CLIP ViT-B/32 (le modèle utilisé par l'app) sur de vraies photos de plats Food-101,
en classant parmi TOUS les aliments de l'app (+ arrière-plans). Compare plusieurs formulations de requêtes.

Ce n'est pas le modèle quantifié exact de l'app (ici : PyTorch, précision complète), mais la même architecture et les mêmes poids.
Seuls les plats de Food-101 qui existent aussi dans la liste de l'app sont testés (une vingtaine de classes).
"""
import io, json, itertools, random, sys, time
import numpy as np
import torch
from datasets import load_dataset, Image as HFImage
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PER_CLASS = 24
MODEL = 'openai/clip-vit-base-patch32'

# étiquette Food-101 -> noms d'aliments acceptés dans l'app
MAP = {
    'chicken_curry': ['Curry de poulet (lait de coco)', 'Poulet tikka masala'],
    'dumplings': ['Raviolis chinois (vapeur)'],
    'gyoza': ['Raviolis chinois (vapeur)'],
    'edamame': ['Edamame cuits'],
    'falafel': ['Falafel'],
    'french_fries': ['Frites'],
    'fried_rice': ['Riz sauté (œuf, légumes)'],
    'guacamole': ['Guacamole'],
    'hamburger': ['Hamburger (fast-food)'],
    'hummus': ['Houmous'],
    'ice_cream': ['Glace (crème glacée)'],
    'lasagna': ['Lasagnes'],
    'miso_soup': ['Soupe miso'],
    'pad_thai': ['Pad thaï'],
    'pizza': ['Pizza (margherita)'],
    'samosa': ['Samoussa (viande)'],
    'spaghetti_bolognese': ['Spaghetti bolognaise'],
    'spring_rolls': ['Nems (rouleau frit)', 'Rouleaux de printemps (frais)'],
    'sushi': ['Sushi / maki'],
    'grilled_salmon': ['Saumon cuit'],
    'pork_chop': ['Porc cuit (filet, côte)'],
    'steak': ['Bœuf grillé (steak, rumsteck)'],
}

TEMPLATES = {
    'T0': 'a photo of {}, a type of food.',
    'T1': 'a photo of {}.',
    'T2': 'a close-up photo of {} on a plate.',
    'T3': 'a photo of a dish of {}.',
    'T4': 'a top-down photo of {}, food photography.',
    'T5': 'a photo of {}, a type of food, served on a plate.',
    'T6': 'a photo of delicious {}.',
}
ENSEMBLES = {
    'E_T0+T2+T4': ['T0', 'T2', 'T4'],
    'E_T0+T1+T2+T3': ['T0', 'T1', 'T2', 'T3'],
    'E_T0+T2+T4+T5': ['T0', 'T2', 'T4', 'T5'],
    'E_toutes': list(TEMPLATES),
}

cls = json.load(open('bench/classes.json', encoding='utf-8'))
foods = cls['foods']; bg = cls['bg']
names = [f['name'] for f in foods]
descs = [f['desc'] for f in foods] + bg
N = len(descs)

torch.set_grad_enabled(False)
model = CLIPModel.from_pretrained(MODEL).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

def text_emb(template_ids):
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

# ---- images ----
want = {k: [] for k in MAP}
ds = load_dataset('ethz/food101', split='validation', streaming=True).cast_column('image', HFImage(decode=False))
labels = ds.features['label'].names
missing = [k for k in MAP if k not in labels]
print('classes absentes de Food-101 :', missing, flush=True)
t0 = time.time(); seen = 0
for ex in ds:
    seen += 1
    lab = labels[ex['label']]
    if lab in want and len(want[lab]) < PER_CLASS:
        want[lab].append(ex['image']['bytes'])
    if all(len(v) >= PER_CLASS for k, v in want.items() if k in labels):
        break
    if seen % 2000 == 0:
        print('parcouru', seen, 'images,', sum(len(v) for v in want.values()), 'retenues', int(time.time() - t0), 's', flush=True)
print('images retenues :', {k: len(v) for k, v in want.items()}, flush=True)

feats, truth_sets, halves = [], [], []
for lab, lst in want.items():
    for j, b in enumerate(lst):
        img = Image.open(io.BytesIO(b)).convert('RGB')
        inp = proc(images=img, return_tensors='pt')
        e = model.visual_projection(model.vision_model(pixel_values=inp['pixel_values']).pooler_output)
        feats.append((e / e.norm(dim=-1, keepdim=True))[0])
        truth_sets.append({names.index(n) for n in MAP[lab]})
        halves.append(0 if j < PER_CLASS // 2 else 1)
IMG = torch.stack(feats)
print('images encodées :', len(IMG), flush=True)

def evaluate(txt):
    logits = 100 * IMG @ txt.T
    order = logits.argsort(dim=1, descending=True)
    res = {'top1': [0, 0], 'top3': [0, 0], 'n': [0, 0]}
    for i in range(len(IMG)):
        h = halves[i]
        top = order[i].tolist()
        res['n'][h] += 1
        if top[0] in truth_sets[i]: res['top1'][h] += 1
        if any(t in truth_sets[i] for t in top[:3]): res['top3'][h] += 1
    tot = lambda k: sum(res[k])
    return {'top1': tot('top1') / tot('n'), 'top3': tot('top3') / tot('n'),
            'top1_A': res['top1'][0] / max(1, res['n'][0]), 'top1_B': res['top1'][1] / max(1, res['n'][1])}

results = {}
for t in TEMPLATES:
    results[t] = evaluate(text_emb([t]))
for name, ts in ENSEMBLES.items():
    results[name] = evaluate(text_emb(ts))

# précision par classe pour la meilleure configuration
best = max(results, key=lambda k: results[k]['top1'])
cfg = ENSEMBLES.get(best, [best])
txt = text_emb(cfg)
logits = 100 * IMG @ txt.T
pred = logits.argmax(dim=1).tolist()
per = {}
idx = 0
for lab, lst in want.items():
    ok = 0
    conf = {}
    for j in range(len(lst)):
        p = pred[idx]; idx += 1
        if p in {names.index(n) for n in MAP[lab]}: ok += 1
        else: conf[(names + bg)[p]] = conf.get((names + bg)[p], 0) + 1
    top_conf = sorted(conf.items(), key=lambda x: -x[1])[:2]
    per[lab] = (ok, len(lst), top_conf)

print('\n===== RÉSULTATS =====')
print('images testées :', len(IMG), '| classes :', len(want), '| classement parmi', N, 'requêtes (aliments + arrière-plans)')
print('%-18s %7s %7s   %6s %6s' % ('configuration', 'top-1', 'top-3', 'moit.A', 'moit.B'))
for k, v in sorted(results.items(), key=lambda kv: -kv[1]['top1']):
    print('%-18s %6.1f%% %6.1f%%   %5.1f%% %5.1f%%' % (k, 100 * v['top1'], 100 * v['top3'], 100 * v['top1_A'], 100 * v['top1_B']))
print('\nmeilleure configuration :', best)
for lab, (ok, n, conf) in per.items():
    print('  %-20s %2d/%2d  confusions: %s' % (lab, ok, n, ', '.join('%s×%d' % c for c in conf)))
print('\nJSON_RESULTS=' + json.dumps({'best': best, 'results': results, 'per_class': {k: [v[0], v[1]] for k, v in per.items()}}, ensure_ascii=False))
with open('bench/results.json', 'w', encoding='utf-8') as f:
    json.dump({'best': best, 'results': results, 'per_class': {k: [v[0], v[1]] for k, v in per.items()}}, f, ensure_ascii=False, indent=1)
