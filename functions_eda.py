import os
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from matplotlib.gridspec import GridSpec
from configparser import ConfigParser


def get_parameters():
    
    config = ConfigParser()
    config.read(f'./config_EDA.ini')

    pathProject = config['EDA']['pathProject']
    pathFile = config['EDA']['pathFile']
    fileName = config['EDA']['fileName']
    target = config['EDA']['target']
    sep = eval(config['EDA']['sep'] )
    dict_replace = eval(config['EDA']['dict_replace'])
    fromNumToCat = eval(config['EDA']['fromNumToCat'])
    fromCatToNum = eval(config['EDA']['fromCatToNum'])
    varsNoPlot = eval(config['EDA']['varsNoPlot'] )
    
    return pathProject, pathFile, fileName, target, sep, dict_replace, fromNumToCat, fromCatToNum, varsNoPlot


class get_file_and_univ:
    
    def __init__(self, pathFile, file, sep, target, pathProject):
        self.pathFile = pathFile 
        self.file = file
        self.sep = sep 
        self.target = target
        self.pathProject = pathProject
        
        if os.path.exists(self.pathProject) == False:
            os.mkdir(self.pathProject)
            

    def __read_file(self):

        logging.info(f'Leyendo dataset {self.file} de la ruta {self.pathFile}')
        if 's3' in self.pathFile:
            df_dict = pd.read_csv(f'{self.pathFile}models/ChPy_cpl_03/dictionary.csv', sep='|')
            varsTrue = df_dict[df_dict.USED == True]['FEATURE'].str.lower().tolist()
            df = pd.read_csv(f'{self.pathFile}{self.file}', sep=self.sep, low_memory=False)\
            .sample(frac=0.1, replace=False, random_state=22).reset_index(drop=True)\
            [varsTrue]
        else:
            df = pd.read_csv(f'{self.pathFile}{self.file}', sep=self.sep)
            df.columns = df.columns.str.lower()

        logging.info(f'shape dataset: {df.shape}')
        return df


    def __get_vars_cat_num(df_, fromCatToNum, fromNumToCat):

        def __change_types_cat(df_, varsCategorical):
            df_[varsCategorical] = df_[varsCategorical].astype(str)
            return df_

        varsNumerical = df_.columns[df_.dtypes != 'object'].tolist()
        varsCategorical = df_.columns[df_.dtypes == 'object'].tolist()
        varsNumerical = [elem for elem in varsNumerical if elem not in fromNumToCat] + fromCatToNum
        varsCategorical = [elem for elem in varsCategorical if elem not in fromCatToNum] + fromNumToCat
        logging.info(f'En el dataset hay {len(varsNumerical)} vaiable numericas y {len(varsCategorical)} categoricas')

        if len(fromNumToCat) > 0:
            df_ = __change_types_cat(df_, varsCategorical)

        return df_, varsNumerical, varsCategorical
    

    def __get_univ(self, df_, varsCategorical):

        logging.info('Generando fichero csv con el analisis univariante')
        df_univ = pd.DataFrame(columns=['variable', 'count_informed', 'count_missing', '%_missing', 'nunique', 'top', 'freq', 'mean', 'std', 'min', 'max', 'type'])
        n = len(df_)

        for var in df_.columns:
            n_nas = df_[var].isna().sum()
            count_informed = n-n_nas
            pct_nas = round((n_nas/n)*100, 6)
            nunique = df_[var].nunique()
            type_ = df_[var].dtypes
            if var in varsCategorical:
                top = df_[var].value_counts(sort=True).index.tolist()[0]
                freq = df_[var].value_counts(sort=True).tolist()[0]
                mean, std, min_, max_ = '', '', '', ''
            else:
                top, freq = '', ''
                mean = df_[var].mean()
                std = df_[var].std()
                min_ = df_[var].min()
                max_ = df_[var].max()
            df_univ.loc[len(df_univ)] = [var, count_informed, n_nas, pct_nas, nunique, top, freq, mean, std, min_, max_, type_]

        df_univ.to_csv(f'{self.pathProject}univ_data.csv', sep=self.sep, index=False)

        
    def __get_vars_to_plot(varsNumerical, varsCategorical, varsNoPlot):
    
        logging.info(f'Variables que no se van a utilizar a la hora de generar las distintas graficas: {len(varsNoPlot), varsNoPlot}')
        varsCategorical = list(set(varsCategorical) - set(varsNoPlot))
        varsNumerical = list(set(varsNumerical) - set(varsNoPlot))
        logging.info(f'Finalmente, seran utilizadas para generar las distintas graficas {len(varsNumerical)} vaiables numericas y {len(varsCategorical)} variables categoricas')

        return varsNumerical, varsCategorical
    
    
    def run(self, fromCatToNum=[], fromNumToCat=[], dict_replace={}, varsNoPlot=[]):
        
        df = get_file_and_univ.__read_file(self)
        # Comprobamos si los posibles valores de la target son valores numericos, en el caso de que la target tenga valores categoricos estos deberan de ser reemplazados a valores numericos a traves de un diccionario
        target_dig = all(str(elem).isdigit() for elem in df[self.target].unique().tolist())
        if (target_dig == False) and (len(dict_replace) > 0):
            logging.info(f'Cambiando los valores de la target {self.target} utilizando el diccionario {dict_replace}')
            df[self.target] = df[self.target].replace(list(dict_replace.keys()), list(dict_replace.values()))
        elif (target_dig == False) and (len(dict_replace) == 0):
            logging.info(f'La target {target} posee valores categoricos, por favor complete la variable dict_replace con los nuevos valores asignados a cada valor de la target.')
            return
        
        df_, varsNumerical, varsCategorical = get_file_and_univ.__get_vars_cat_num(df, fromCatToNum, fromNumToCat)
        # Generamos el analisis univariante
        get_file_and_univ.__get_univ(self, df_, varsCategorical)
        # Eliminamos las variables que no queramos utilizar para generar las graficas
        varsNumerical, varsCategorical = get_file_and_univ.__get_vars_to_plot(varsNumerical, varsCategorical, varsNoPlot)
        
        return df_, varsNumerical, varsCategorical   


class plot_numerical_var:

    def __init__(self, df, target, pathProject, savePlots):
        
        self.df = df
        self.target = target
        self.pathProject = pathProject
        self.savePlots = savePlots
        

    def __hist_boxplot_plot(self, var, figsize=(17, 8)):

        fig = plt.figure(figsize=figsize)
        grid = GridSpec(nrows=2, ncols=1, figure=fig)
        ax0 = fig.add_subplot(grid[0, :])
        ax0.set_title(f'Histogram y BoxPlot de {var}')
        sns.distplot(self.df[var], ax=ax0, color='deepskyblue', kde=False)

        ax1 = fig.add_subplot(grid[1, :])
        plt.axis('off')
        sns.boxplot(x=self.df[var], ax=ax1, color='deepskyblue')
        if self.savePlots == True:
            plt.savefig(f'{self.pathProject}{var.upper()}_hist_boxplot.png')
            plt.close(fig)
        else:
            plt.show()


    def __boxplot_violin_target_plot(self, var, figsize=(20, 8)):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        sns.boxplot(x=self.target, y=var, data=self.df, palette='pastel', ax=ax1)
        sns.violinplot(x=self.target, y=var, data=self.df, inner=None, color=".8", ax=ax2)
        sns.stripplot(x=self.target, y=var, data=self.df, palette='pastel', ax=ax2)
        ax1.set_title(f'Boxplot de {var} por target')
        ax1.set_xlabel(f'{self.target} labels')
        ax1.set_ylabel(f'{var}')
        ax2.set_title(f'Violin y stripplot de {var} por target')
        ax2.set_xlabel(f'{self.target} labels')
        ax2.set_ylabel(f'{var}')
        if self.savePlots == True:
            plt.savefig(f'{self.pathProject}{var.upper()}_boxplot_por_target.png')
            plt.close(fig)
        else:
            plt.show()


    def __var_tramitificada_vs_target(self, var, n_bins=5, figsize=(14, 8)):

        def __get_cut(df_, var, target, n_bins):

            df_c = df_.copy()
            df_c[f'{var}_cut'] = pd.cut(df_c[var], bins=n_bins, include_lowest=True)
            df_c[f'{var}_cut'] = df_c[f'{var}_cut'].astype(str)    

            df_pr = df_c.groupby(f'{var}_cut').agg({target:['mean', 'count']})
            df_pr.columns = df_pr.columns.droplevel()
            df_pr = df_pr.reset_index()

            return df_pr

        df_plot = __get_cut(self.df, var, self.target, n_bins)
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.bar(df_plot[f'{var}_cut'], df_plot['count'], color='deepskyblue')
        ax2 = ax1.twinx()
        ax2.plot(df_plot[f'{var}_cut'], df_plot['mean'], marker='o', linestyle=':', color='black')
        plt.title(f'Volumen & target rate \n para {var}; target = {self.target}')
        ax1.tick_params(axis='x', rotation=90)
        ax1.set_xlabel(f'{var} buckets')
        ax1.set_ylabel('Volumen (counts)')
        ax2.set_ylabel('Target ratio')
        if self.savePlots == True:
            plt.savefig(f'{self.pathProject}{var.upper()}_tramificada.png')
            plt.close(fig)
        else:
            plt.show()
        
    def run(self, var):
        plot_numerical_var.__hist_boxplot_plot(self, var)
        plot_numerical_var.__boxplot_violin_target_plot(self, var)
        plot_numerical_var.__var_tramitificada_vs_target(self, var)     
        
              
class plot_categorical_var:

    def __init__(self, df, target, pathProject, savePlots):
        
        self.df = df
        self.target = target
        self.pathProject = pathProject
        self.savePlots = savePlots
        
        
    def __get_dataset_plt_cat(self, varCat):
    
        df_agg_cat = self.df[[varCat, self.target]].groupby(varCat).agg({self.target:['count', 'mean']})
        df_agg_cat.columns = df_agg_cat.columns.droplevel()
        df_agg_cat = df_agg_cat.reset_index()

        df_agg_cat_target = pd.DataFrame(self.df.groupby(varCat)[self.target].value_counts())
        df_agg_cat_target = df_agg_cat_target.rename(columns={self.target:'count'}).reset_index()
        df_agg_cat_target[self.target] = df_agg_cat_target[self.target].astype(str)

        df_agg_cat_target_pivot = pd.pivot_table(df_agg_cat_target, index=[varCat], columns=self.target)
        df_agg_cat_target_pivot.columns = list(map("_".join, df_agg_cat_target_pivot.columns))
        df_agg_cat_target_pivot = df_agg_cat_target_pivot.rename(columns={'count_0':f'{self.target}_0', 'count_1':f'{self.target}_1'}).reset_index()

        return df_agg_cat, df_agg_cat_target_pivot


    def __varCat_hist_target(self, df_agg_cat, varCat, figsize=(14, 8)):

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.bar(df_agg_cat[varCat], df_agg_cat['count'], color='deepskyblue')
        ax2 = ax1.twinx()
        ax2.plot(df_agg_cat[varCat], df_agg_cat['mean'], marker='o', linestyle=':', color='black')
        plt.title(f'Volumen & target rate \n para {varCat}; target = {self.target}')
        ax1.tick_params(axis='x', rotation=90)
        ax1.set_xlabel(f'{varCat} buckets')
        ax1.set_ylabel('Volumen (counts)')
        ax2.set_ylabel('Target ratio')
        if self.savePlots == True:
            plt.savefig(f'{self.pathProject}{varCat.upper()}_hist_por_target.png')
            plt.close(fig)
        else:
            plt.show()


    def __varCat_barplot_por_target(self, df, varCat):

        labels = df[varCat].unique().tolist()
        x = np.arange(len(labels))
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width/2, df[f'{self.target}_0'], width, label=f'{self.target}_0')
        rects2 = ax.bar(x + width/2, df[f'{self.target}_1'], width, label=f'{self.target}_1')

        ax.set_title(f'{varCat} por target')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(f'{varCat}')
        ax.set_ylabel('Volumen (count)')
        ax.legend()
        if self.savePlots == True:
            plt.savefig(f'{self.pathProject}{varCat.upper()}_barplot.png')
            plt.close(fig)
        else:
            plt.show()

            
    def run(self, varCat):
        
        df_agg_cat, df_agg_cat_target_pivot = plot_categorical_var.__get_dataset_plt_cat(self, varCat)
        plot_categorical_var.__varCat_hist_target(self, df_agg_cat, varCat)
        plot_categorical_var.__varCat_barplot_por_target(self, df_agg_cat_target_pivot, varCat)
        