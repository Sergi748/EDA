# Modo de ejecucion
# nohup python3 main.py &
from functions_eda import *
from datetime import datetime

logging.basicConfig(filename=f"log_EDA.log", level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('#### Proceso EDA iniciado')
try:
    startTime = datetime.now()
    pathProject, pathFile, fileName, target, sep, dict_replace, fromNumToCat, fromCatToNum, varsNoPlot, makePlots, savePlots = get_parameters()

    get_file = get_file_and_univ(pathFile, fileName, sep, target, pathProject)
    df, varsNumerical, varsCategorical = get_file.run(fromCatToNum, fromNumToCat, dict_replace, varsNoPlot)
    
    if makePlots == True:
        # Generamos las graficas para las variables numericas
        plt_num = plot_numerical_var(df, target, pathProject, savePlots)
        for var in varsNumerical:
            plt_num.run(var)

        # Generamos las graficas para las variables categoricas
        plt_cat = plot_categorical_var(df, target, pathProject, savePlots)
        for varCat in varsCategorical:
            plt_cat.run(varCat)
        
    logging.info(f'Duracion del proceso {datetime.now() - startTime}')
    logging.info('#### Proceso EDA finalizado')
except Exception as Argument:
    logging.exception(Argument)    


