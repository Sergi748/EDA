# Modo de ejecucion
# nohup python3 main.py &
from functions_eda import *

logging.basicConfig(filename=f"log_EDA.log", level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('#### Proceso EDA iniciado')
try:
    pathProject, pathFile, fileName, target, sep, dict_replace, fromNumToCat, fromCatToNum, varsNoPlot = get_parameters()

    get_file = get_file_and_univ(pathFile, fileName, sep, target, pathProject)
    df, varsNumerical, varsCategorical = get_file.run(fromCatToNum, fromNumToCat, dict_replace, varsNoPlot)
    
    # Generamos las graficas para las variables numericas
    plt_num = plot_numerical_var(df, target, pathProject, True)
    for var in varsNumerical:
        plt_num.run(var)
        
    # Generamos las graficas para las variables categoricas
    plt_cat = plot_categorical_var(df, target, pathProject, True)
    for varCat in varsCategorical:
        plt_cat.run(varCat)
        
    logging.info('#### Proceso EDA finalizado')
except Exception as Argument:
    logging.exception(Argument)    