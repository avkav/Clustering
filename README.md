Descripción de cada bloque:

Lectura de datos: Se lee el archivo CSV y se muestra una vista preliminar.

Exploración de datos: Incluye la descripción del conjunto de datos y el tipo de cada variable.

Detección de valores nulos: Cuenta cuántos nulos hay por cada feature.

Buscar valores únicos: Se crean dos columnas donde una contiene las features y la otra sus valores únicos.

Tratar valores nulos: Se imputan o eliminan según sea necesario.

Eliminar features sin información: Se eliminan features con solo un valor único.

Separación de variables predictoras y variable a predecir: La variable objetivo es class.

Codificación de variables categóricas: Se realiza One Hot Encoding.

Train-test split: Dividimos el dataset en conjuntos de entrenamiento y prueba.

PCA: Reducción de dimensionalidad a dos componentes y visualización.

Entrenamiento con Random Forest: Se entrena y evalúa un modelo de Random Forest.

Reducción de features: Se prueba PCA con diferentes números de componentes y se visualiza la precisión.

K-Means Clustering: Se busca el número óptimo de clusters con el método del codo.

Visualización de clusters: Se visualiza la distribución de los clusters encontrados por KMeans.

##Columnas:


color del sombrero (cap-color):

marrón = n,
beige = b,
canela = c,
gris = g,
verde = r,
rosa = p,
morado = u,
rojo = e,
blanco = w,
amarillo = y
magulladuras (bruises):

sí = t,
no = f
olor (odor):

almendra = a,
anís = l,
creosota = c,
olor a pescado = y,
fétido = f,
mohoso = m,
sin olor = n,
picante = p,
especiado = s
unión de las láminas (gill-attachment):

adherida = a,
descendente = d,
libre = f,
muesca = n
espaciado de las láminas (gill-spacing):

juntas = c,
apiñadas = w,
distantes = d
tamaño de las láminas (gill-size):

anchas = b,
estrechas = n
color de las láminas (gill-color):

negro = k,
marrón = n,
beige = b,
chocolate = h,
gris = g,
verde = r,
naranja = o,
rosa = p,
morado = u,
rojo = e,
blanco = w,
amarillo = y
forma del tallo (stalk-shape):

agrandado = e,
afilado = t
raíz del tallo (stalk-root):

bulbosa = b,
en forma de garrote = c,
en forma de copa = u,
igual = e,
rizomorfos = z,
enraizada = r,
faltante = ?
superficie del tallo por encima del anillo (stalk-surface-above-ring):

fibrosa = f,
escamosa = y,
sedosa = k,
lisa = s
superficie del tallo por debajo del anillo (stalk-surface-below-ring):

fibrosa = f,
escamosa = y,
sedosa = k,
lisa = s
color del tallo por encima del anillo (stalk-color-above-ring):

marrón = n,
beige = b,
canela = c,
gris = g,
naranja = o,
rosa = p,
rojo = e,
blanco = w,
amarillo = y
color del tallo por debajo del anillo (stalk-color-below-ring):

marrón = n,
beige = b,
canela = c,
gris = g,
naranja = o,
rosa = p,
rojo = e,
blanco = w,
amarillo = y
tipo de velo (veil-type):

parcial = p,
universal = u
color del velo (veil-color):

marrón = n,
naranja = o,
blanco = w,
amarillo = y
número de anillos (ring-number):

ninguno = n,
uno = o,
dos = t
tipo de anillo (ring-type):

en forma de telaraña = c,
evanescente = e,
en expansión = f,
grande = l,
ninguno = n,
colgante = p,
en funda = s,
en zona = z
color de la impresión de esporas (spore-print-color):

negro = k,
marrón = n,
beige = b,
chocolate = h,
verde = r,
naranja = o,
morado = u,
blanco = w,
amarillo = y
población (population):

abundante = a,
agrupada = c,
numerosa = n,
dispersa = s,
varias = v,
solitaria = y
hábitat (habitat):

pastizales = g,
hojas = l,
prados = m,
caminos = p,
urbano = u,
desechos = w,
bosques = d