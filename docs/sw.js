/* Service worker d'Assiette : l'app et le modèle d'IA restent disponibles hors ligne après le premier usage.
   - coquille de l'app : mise en cache à l'installation ;
   - bibliothèque Transformers.js et moteur WebAssembly (jsDelivr), polices Google : mis en cache au fil de l'eau ;
   - fichiers du modèle (Hugging Face) : déjà gardés par Transformers.js dans le cache du navigateur, on ne les double pas ici.
   Les données du journal restent dans le stockage local du navigateur, jamais ici. */
const VERSION = 'assiette-v1';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon.svg', './icon-192.png', './icon-512.png', './icon-180.png'];
const RUNTIME_HOSTS = ['cdn.jsdelivr.net', 'fonts.googleapis.com', 'fonts.gstatic.com'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('assiette-') && k !== VERSION && k !== VERSION + '-runtime').map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  /* L'app elle-même : réseau d'abord (pour recevoir les mises à jour), cache en secours hors ligne. */
  if (url.origin === self.location.origin) {
    e.respondWith(
      fetch(req)
        .then((res) => {
          if (res && res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
          return res;
        })
        .catch(() => caches.match(req).then((hit) => hit || caches.match('./index.html')))
    );
    return;
  }

  /* Bibliothèque, moteur WebAssembly, polices : cache d'abord (fichiers versionnés, donc immuables). */
  if (RUNTIME_HOSTS.some((h) => url.hostname === h || url.hostname.endsWith('.' + h))) {
    /* L'app ajoute ?retry=N à l'adresse de la bibliothèque après un échec : même entrée de cache que l'adresse d'origine. */
    const keyUrl = new URL(url.href); keyUrl.searchParams.delete('retry');
    const key = keyUrl.href;
    e.respondWith(
      caches.open(VERSION + '-runtime').then((cache) =>
        cache.match(key).then((hit) => {
          if (hit) return hit;
          return fetch(req).then((res) => {
            /* On ne garde que les réponses complètes et réussies (pas les 206 partiels, pas les erreurs). */
            if (res && res.ok && res.status === 200) cache.put(key, res.clone()).catch(() => {});
            return res;
          });
        })
      )
    );
  }
});
