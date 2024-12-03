La régression Ridge est une méthode de régularisation utilisée en régression linéaire pour atténuer l'overfitting, c'est-à-dire lorsque le modèle est trop complexe et se soucie trop des détails et du bruit dans les données d'apprentissage. Cette méthode ajoute une pénalité à la somme des carrés des erreurs (RSS - Residual Sum of Squares) de la régression linéaire ordinaire, ce qui permet de contrôler l'amplitude des coefficients des variables explicatives.

### Principe de la Régression Ridge

Le principe de la régression Ridge repose sur l'ajout d'une contrainte sur la norme L2 (longueur euclidienne) des coefficients des variables explicatives. Cette contrainte vise à réduire l'importance des coefficients les plus importants tout en conservant ceux qui ont un impact significatif sur la prédiction. Le but est de "rétrécir" les coefficients vers zéro, mais sans les anéantir complètement, ce qui permet d'éviter l'underfitting où le modèle serait trop simple pour capturer les tendances importantes dans les données.

La fonction de coût de la régression Ridge inclut un terme de pénalité proportionnel à la somme des carrés des coefficients, multiplié par un paramètre λ (lambda). Ce paramètre λ contrôle le niveau de pénalisation appliqué aux coefficients. Une valeur élevée de λ pénalise davantage les coefficients, conduisant à une réduction de leur taille, tandis qu'une valeur faible de λ permet aux coefficients de conserver une importance plus grande.

Mathématiquement, l'estimateur Ridge est donné par :

\[
\hat{\beta}_{Ridge}(\lambda) = (X'X + \lambda I_p)^{-1} X'y
\]

où \(X\) est la matrice des variables explicatives, \(y\) est le vecteur des observations de la variable dépendante, \(I_p\) est la matrice identité de dimension \(p\times p\), et \(\lambda\) est le paramètre de pénalisation.

### Choix du Paramètre λ

Le choix optimal de λ est crucial pour le bon fonctionnement de la régression Ridge. Un choix trop élevé de λ peut conduire à un underfitting, car le modèle devient trop simple. À l'inverse, un choix trop faible de λ peut conduire à un overfitting, car le modèle est trop complexe et se soucie trop des détails spécifiques des données d'apprentissage. Pour choisir une valeur appropriée de λ, on utilise souvent la validation croisée, qui consiste à essayer plusieurs valeurs de λ et à évaluer les performances du modèle pour chacune d'entre elles.

### Avantages de la Régression Ridge

- **Stabilisation des coefficients**: La régression Ridge stabilise les coefficients des variables explicatives, rendant le modèle moins sensible aux variations des données.
- **Prédiction robuste**: Grâce à la pénalisation des coefficients, la régression Ridge peut produire des prédictions plus stables et robustes face à l'overfitting.
- **Interprétabilité**: Les coefficients ajustés par la régression Ridge sont généralement plus petits que ceux de la régression linéaire ordinaire, ce qui facilite leur interprétation.


