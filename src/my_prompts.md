
en src, implementa en una clase una estructura de datos "VGMatrix2D" que sirva para almacenar valores en forma matricial 
y que implemente de forma eficiente distintas operaciones sobre matrices bidimensionales como la suma y resta de matrices, 
la multiplicación con otra matriz de dimensiones correctas, o la multiplicación por un escalar. 
Es muy importante mantener la eficiencia tanto a la hora de asignar valores como al obtenerlos o realizar operaciones sobre ellos.
En el futuro, implementará métodos para aplicar máscaras o distintas operaciones de filtrado, 
aunque por el momento solo se requiere que se puedan aplicar máscaras de convolución.
A la hora de inicializar estas matrices, se indicará un tamaño como tupla (filas, columnas) y un valor por defecto para rellenar la matriz.
Después de inicializarla, debe ser posible redimensionarla a un nuevo tamaño, manteniendo los valores anteriores en la medida de lo posible y
haciéndolo de forma eficiente. Se inicializa así: matrix: VGMarix2D = VGMatrix2D((512, 512), 0.55) # Crea una matriz de 512x512 con valores iniciales de 0.55
La estructura de datos también debe permitir tener posiciones sin asignar, lo que se indicará con un None.
tendrá un método get_value_at(row: int, col: int) -> Optional[float] que devolverá el valor en la posición indicada o None si no se ha asignado ningún valor.
y un método set_value_at(row: int, col: int, value: Optional[float]) que asignará el valor en la posición indicada, permitiendo también asignar None para indicar que esa posición no tiene un valor asignado.
Si conoces una forma más eficiente de indicar las posiciones sin asignar, puedes implementarla, pero el uso de None es una opción válida.

