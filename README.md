# EDA
Código para generar un análisis exploratorio de datos. 

Para utilizar este código se ha de rellenar el fichero de configuración con los siguientes parametros:
  - pathProject: Ruta donde se quiere guardar el fichero csv y las distintas gráficas generadas.
  - pathFile: Ruta donde se encuentra el fichero csv que contiene los datos.
  - fileName: Nombre del fichero csv que contiene los datos.
  - target: Nombre de la variable utilizada target.
  - sep: Tipo de separador que tiene el fichero csv (ejemplo ','), este mismo separador será el utilizado para guardar el fichero csv con el análisis univariante. 
  - dict_replace: Si la varible target no es númerica se genera un diccionario con los nuevos valores de la target (ejemplo: {'No':0, 'Yes':1}), si la target ya es numerica dejarlo con un diccionario vacio (ejemplo {}).
  - fromNumToCat: Lista de variables que son leidas como variables numericas pero queremos que sean consideradas como variables categoricas a la hora de realizar las distintas graficas (ejemplo: ['seniorcitizen']), si no se quiere modificar ninguna variable dejar una lista vacia (ejemplo []).
  - fromCatToNum: Lista de variables que son leidas como variables categoricas pero queremos que sean consideradas como variables numericas a la hora de realizar las distintas graficas (ejemplo: ['seniorcitizen']), si no se quiere modificar ninguna variable dejar una lista vacia (ejemplo []).
  - varsNoPlot: Lista de variables de las cuales no queremos generar ninguna gráfica (ejemplo: ['customerid']), si queremos generar las gráficas de todas las variables dejar una lista vacia (ejemplo []).
  - makePlots: Parametro con el que se indica si se quiere generar las distintas gráficas o solo queremos obtener el fichero csv con el análisis exploratio. Las opciones son True o False.
  - savePlots: Parametro para indicar si queremos guardar las distintas gráficas en la ruta indicada en el parametro 'pathProject' o si preferimos que estas gráficas se impriman en la pantalla sin ser guardadas. Las opciones son True o False.

