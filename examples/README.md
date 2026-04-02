# Génération de Dataset : Éolienne New Mexico

Ce module est dédié à la simulation aérodynamique de l'éolienne **New Mexico** via le code SVEN.

## Base de Données Aérodynamique
Le solveur calcule les forces aérodynamiques normales ($F_n$) et tangentielles ($F_t$) en faisant varier les paramètres suivants :

* **Position Radiale ($r$) :** Discrétisation de la pale basée sur la géométrie réelle du fichier `mexico.blade`.
* **Position Azimutale ($\theta$) :** Rotation complète du rotor (0 à 360°) avec un pas par défaut de 10°.
* **Angle de Lacet ($\gamma$) :** Désalignement de la turbine par rapport au vecteur vent entrant (Yaw).
* **Tip Speed Ratio (TSR) :** Ratio entre la vitesse en bout de pale et la vitesse du vent incident.

## Format de Sortie (Dataset)
Pour faciliter l'exploitation des données et l'entraînement de modèles de Machine Learning, les résultats ne sont plus stockés dans des fichiers texte individuels mais regroupés dans un **tenseur PyTorch** unique (`.pt`).

La structure du tenseur est la suivante :  
`[Index_Yaw, Index_TSR, Type_Force (Fn/Ft), Index_Azimut, Index_Section_Radiale]`.

## Suivi de la Convergence
Pour chaque combinaison (Yaw, TSR), un fichier d'historique est généré dans le dossier `outputs/`. Cela permet de vérifier graphiquement que la simulation a bien atteint un **régime périodique permanent** avant l'extraction de la moyenne finale.

---

*Note : Les anciens cas de test (aile elliptique et VAWT) ont été retirés pour se concentrer exclusivement sur la configuration New Mexico. *

# How to 

Include the python folders to be able to run the examples. This can be done before calling the scripts :

``` export PYTHONPATH=/path/to/repo:$PYTHONPATH ```  


Or inside the scripts : 

```python
import sys
sys.path.append('/path/to/repo')
