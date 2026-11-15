# Assiette : suivi alimentaire par photo, 100 % sur l'appareil

Photographiez une assiette : un modèle d'IA qui tourne dans le navigateur repère les aliments et estime leur poids, l'app calcule calories et macros. Pas de compte, pas de Claude, pas de serveur à vous : **la photo n'est jamais envoyée nulle part**. Le journal reste dans le stockage local du navigateur.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `index.html` | L'application complète (un seul fichier) |
| `manifest.webmanifest`, `icon*.png`, `icon.svg` | Installation sur l'écran d'accueil |
| `sw.js` | Service worker : fonctionnement hors ligne après le premier usage |

## Mise en ligne (obligatoire : le modèle exige une adresse en HTTPS)

L'app doit être servie en **HTTPS**. Sans cela, le navigateur refuse le cache du modèle (il serait re-téléchargé à chaque fois) et l'installation sur l'écran d'accueil.

1. **Netlify Drop** (le plus simple) : ouvrez `app.netlify.com/drop`, glissez-y le dossier. Vous obtenez une adresse `https://…netlify.app`.
2. **GitHub Pages** : créez un dépôt, déposez les fichiers à la racine, activez *Settings → Pages → Deploy from a branch*.
3. **Cloudflare Pages** ou tout hébergeur statique : même principe.

Test rapide sur ordinateur : `python3 -m http.server 8000` dans le dossier, puis `http://localhost:8000` (localhost est accepté en HTTP). Sur un téléphone via l'adresse IP du réseau local, ça ne passera pas en HTTPS : passez par un hébergeur.

## Premier lancement

- Le premier scan télécharge un modèle de reconnaissance (CLIP, version quantifiée) : **de l'ordre de 100 à 200 Mo, estimation non mesurée**, la barre de progression affiche la taille réelle. Faites-le en Wi-Fi.
- Une fois téléchargé, il reste dans le cache du navigateur : les scans suivants fonctionnent **hors ligne**.
- L'app ajoute un second téléchargement, plus court, la première fois : elle encode une fois la liste des 198 aliments, puis garde le résultat.
- Les polices (Archivo, DM Mono) viennent de Google Fonts si le réseau est disponible ; sinon l'app utilise les polices du système.

Installer : Android/Chrome → menu ⋮ → *Installer l'application* ; iPhone/Safari → Partager → *Sur l'écran d'accueil*.

## Ce qui est envoyé, et à qui

- La photo : **jamais**.
- Téléchargés à l'installation : la bibliothèque Transformers.js et son moteur WebAssembly (jsDelivr), les fichiers du modèle (Hugging Face), éventuellement les polices (Google). Ces serveurs voient une demande de téléchargement (adresse IP, navigateur), pas votre repas.
- Le journal et les réglages : stockage local du navigateur, sur cet appareil uniquement. Vider les données du site efface le journal ; le bouton « Copier mon journal » en fait une sauvegarde JSON.

## Précision : ce qu'il faut savoir

- La reconnaissance est « zéro-shot » : le modèle compare la photo à une description anglaise de chacun des 198 aliments de la base. Il se débrouille bien sur des aliments distincts et bien séparés dans l'assiette ; il hésite sur les plats mixtes (gratins, sautés, sauces), les aliments voisins (riz / semoule / quinoa) et ne voit pas ce qui est caché (huile, beurre, sauce).
- Les grammes sont une **estimation grossière** : surface occupée × épaisseur moyenne par type d'aliment, ramenée vers une portion usuelle et bornée entre 0,4 et 2,5 fois cette portion. Comptez facilement 30 à 50 % d'erreur. Une seule photo ne donne pas la profondeur.
- Pour de meilleurs résultats : photo **de dessus**, **assiette entière** remplissant presque toute la largeur, bonne lumière, et réglez le **diamètre de votre assiette** dans les réglages (par défaut 26 cm).
- L'écran de résultat est conçu pour corriger : retirez les faux aliments, ajoutez-en via les « Autres propositions » ou la recherche, réglez chaque poids.
- Les valeurs nutritionnelles de la base intégrée sont des moyennes indicatives (pour 100 g tels que mangés), non vérifiées une à une. Pour un suivi médical, comparez-les à la table Ciqual de l'Anses.
- Cet outil aide à garder des repères ; il ne remplace pas l'avis d'un professionnel de santé.

## Précision mesurée sur de vraies photos

Le modèle n'est pas « entraîné » ici : CLIP est utilisé tel quel (zéro-shot), car un vrai entraînement demande des milliers de photos étiquetées et un GPU. Ce qui a été fait : un **banc d'essai** automatique (dossier `bench/`, lancé par GitHub Actions) qui mesure la reconnaissance sur de vraies photos du jeu Food-101 et sert à choisir la formulation des requêtes envoyées au modèle.

- Protocole : 22 plats de Food-101 qui existent dans la base de l'app ; chaque photo est classée parmi 208 candidats (les 198 aliments + 10 fonds de photo). Les réglages ont été choisis sur 528 photos (jeu « DEV »), puis vérifiés sur **880 autres photos jamais regardées** (jeu « TEST »). Seuls les chiffres du jeu TEST comptent.
- Résultat sur TEST : le bon plat est en tête dans **76,9 %** des cas avec les requêtes retenues, contre 74,0 % avec la formulation initiale (+3,0 points, intervalle à 95 % : +1,5 à +4,4). Le bon plat est dans les 3 premières propositions dans 89,4 % des cas (pas de différence entre les deux). Sur le jeu DEV on obtenait 84 % : c'est le chiffre optimiste d'un jeu sur lequel on a réglé, pas celui à retenir.
- Points faibles mesurés : le hamburger (23/40, pris pour un steak haché), le houmous (20/40), le porc et le saumon grillés (22 et 24/40, confondus avec poulet ou bœuf), le steak (26/40, pris pour du porc). Pour ces plats, utilisez les « Autres propositions ».
- À savoir : photos de plats de restaurant, 22 plats sur 198 ; modèle PyTorch en pleine précision, alors que l'app utilise une version quantifiée (légèrement moins précise, non mesurée). Des descriptions réécrites à la main pour corriger les confusions ont fait **baisser** les scores sur les deux jeux : elles n'ont pas été retenues.
- **Les grammes n'ont pas pu être mesurés** (aucun jeu de photos avec poids réels disponible) : ne comptez pas dessus mieux que les 30 à 50 % d'erreur indiqués plus haut.

## L'app apprend de vos corrections (sur votre appareil)

Quand vous enregistrez un repas scanné, l'app compare les grammes qu'elle proposait à ceux que vous avez validés et ajuste ses prochaines estimations : d'abord par aliment, puis de façon globale. Elle favorise aussi légèrement (jusqu'à ×1,5) les aliments que vous mangez souvent. Rien ne quitte l'appareil. Réglages > « Oublier ce qui a été appris » remet tout à zéro. Cet effet n'a pas pu être mesuré hors ligne : il se vérifiera à l'usage, après quelques repas corrigés.

## Ce qui a été testé, et ce qui ne l'a pas été

Testé : la logique de calcul (fusion des aliments voisins, bornes, effet du diamètre d'assiette, cache), l'interface dans un vrai Chrome avec une bibliothèque simulée (flux complet photo → résultat → journal, échec réseau puis nouvel essai, installation du service worker), le fonctionnement hors ligne du service worker.

**Non testé : le vrai modèle.** L'environnement de développement n'avait pas accès à Internet, donc le téléchargement et l'exécution réels de CLIP n'ont pas pu être essayés, ni la durée d'un scan, ni la précision sur de vraies assiettes. Si le modèle ne se charge pas, l'app affiche un message et propose de composer le repas à la main. Si « Réessayer » échoue encore, rechargez la page.

À surveiller au premier essai : durée d'un scan sur votre téléphone (le moteur tourne en WebAssembly ; sur un vieil appareil, ou un iPhone ancien, le modèle peut être trop lourd pour la mémoire disponible).

## Pour aller plus loin

Dans `index.html`, bloc « reconnaissance locale » :
- `ML_MODEL` : le modèle utilisé. Un modèle plus léger, s'il est compatible avec Transformers.js, se branche ici ; à tester.
- `ML_ROWS` : pour chaque aliment, la description anglaise donnée au modèle, un type de portion et une famille. Affiner ces descriptions est le levier le plus simple pour améliorer la reconnaissance.
- `GCM` : grammes par cm² visible, par type de portion : à ajuster en pesant quelques assiettes.

Une vraie application native (Expo / React Native avec un modèle TFLite) serait plus rapide et plus fiable qu'une page web, au prix d'un développement plus long et d'une publication sur les stores.
