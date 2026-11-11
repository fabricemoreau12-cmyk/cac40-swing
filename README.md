# CAC 40 Swing

Outil de swing trading sur le CAC 40 : repère les configurations de retournement sur niveaux clés, propose un niveau de déclenchement, un stop d'invalidation et des objectifs, et mesure honnêtement (hors échantillon) ce que ces règles auraient donné par le passé.

- **Moteur** (`src/cac40_swing.py`) : tourne chaque soir de semaine sur GitHub Actions, télécharge l'historique `^FCHI` (Yahoo Finance), recalcule tout et publie `data/cac40_swing.json` + un e-mail de synthèse.
- **Application** (`CAC40_Swing.html` + `Lancer CAC40 Swing.bat`) : s'ouvre dans Edge en mode application (sans barre d'adresse) et lit le fichier publié sur GitHub.

> Outil d'aide à la décision, pas un conseil en investissement. Résultats exprimés en R, hors frais, spread et financement des turbos.

## Installation (une seule fois)

1. **Créer un dépôt GitHub** (ex. `cac40-swing`, privé ou public) et y déposer tout le contenu de ce dossier en conservant l'arborescence (`.github/workflows/…`, `src/…`, `data/`, `requirements.txt`).
2. **Secrets pour l'e-mail** — *Settings → Secrets and variables → Actions → New repository secret* :
   | Secret | Exemple |
   |---|---|
   | `SMTP_SERVER` | `smtp.gmail.com` |
   | `SMTP_PORT` | `465` |
   | `SMTP_USER` | votre adresse d'envoi |
   | `SMTP_PASSWORD` | mot de passe d'application (Gmail : *Compte Google → Sécurité → Mots de passe des applications*) |
   | `MAIL_TO` | destinataire(s), séparés par des virgules |

   Vous pouvez reprendre les mêmes valeurs que pour votre screener Nasdaq. Sans ces secrets, l'analyse tourne quand même, sans e-mail.
3. **Autoriser l'écriture** : *Settings → Actions → General → Workflow permissions → Read and write permissions*.
4. **Premier lancement** : onglet *Actions → CAC40 Swing → Run workflow*. Durée ≈ 1 minute. Ensuite, exécution automatique du lundi au vendredi à 17:45 UTC (18:45 en hiver / 19:45 en été à Bruxelles).

## Application

1. Copier `CAC40_Swing.html` et `Lancer CAC40 Swing.bat` dans un même dossier sur le PC.
2. Double-cliquer sur `Lancer CAC40 Swing.bat` (on peut en faire un raccourci sur le bureau).
3. *Réglages* : propriétaire GitHub + nom du dépôt. Si le dépôt est **privé**, créer un jeton *fine-grained* (GitHub → Settings → Developer settings → Fine-grained tokens) limité à ce dépôt avec la permission **Contents : Read-only**, et le coller dans le champ « Jeton ».
4. *Actualiser* télécharge le dernier résultat. Le bouton *Importer…* permet aussi d'ouvrir un `cac40_swing.json` manuellement.

Onglets : **Aujourd'hui** (signal, conditions, niveaux clés), **Graphique** (chandeliers, niveaux, trades passés, ZigZag), **Performance hors échantillon** (walk-forward, comparaison au hasard), **Retournements passés** (statistiques des pivots 3 / 5 / 8 %), **Calculateur turbo** (taille de position, contrôle barrière vs stop), **Méthode**.

## Utilisation en local (facultatif)

```bash
pip install -r requirements.txt
python src/cac40_swing.py                      # données Yahoo
python src/cac40_swing.py --csv export_prt.csv # ou un export CSV (Date,Open,High,Low,Close)
```

## Réglages du moteur

En tête de `src/cac40_swing.py` : `CFG` (fenêtres, durée de validité de l'ordre, stop suiveur, périodes walk-forward, `COST_POINTS` pour intégrer un coût aller-retour en points d'indice) et `GRID` (valeurs testées pour chaque paramètre).

## Licence tierce

`CAC40_Swing.html` embarque [lightweight-charts](https://github.com/tradingview/lightweight-charts) 4.2.3 © TradingView, Inc., licence Apache 2.0.
