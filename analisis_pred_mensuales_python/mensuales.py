# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:16:20 2019

@author: Ruben
"""

import sys
import os
import glob
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_all_pred_historical_data(folder_name, adjust):
    """
    :param adjust: True of False for downlaod adjusted data or not.
    """
    predicciones_alza = pd.read_csv("predicciones_alza.txt", delimiter="\t")
    predicciones_baja = pd.read_csv("predicciones_baja.txt", delimiter="\t")
        
    all_pred_symbols = get_all_pred_symbols_yahoo_format(predicciones_alza,
                                                         predicciones_baja)
    
    get_yahoo_hitorical_data(all_pred_symbols, folder_name, adjust)

def get_all_pred_symbols_yahoo_format(preds_alza, preds_baja):
    all_pred_symbols = pd.concat([preds_alza.accion, preds_baja.accion])
    all_pred_symbols = all_pred_symbols.drop_duplicates()
    all_pred_symbols[:] = all_pred_symbols+".MC"
    all_pred_symbols[all_pred_symbols == 'IBEX.MC'] = '^IBEX'
    return all_pred_symbols.str.upper()
    
def get_yahoo_hitorical_data(symbols, folder_name, adjust):
    """
    :param symbols: pd.Series with the symbols in format SAN.MC, FER.MC, ...
    """
    for s in symbols:
        try:
            t = yf.Ticker(s)
            data = t.history(period="max", interval="1mo", auto_adjust=adjust)  #auto_adjust=True by default
            data.to_csv(folder_name+"/"+t.info.get('symbol'))
        except:
            print(t.info.get('symbol'), sys.exc_info()[0],"occured.")

def load_all_yahoo_quotes_without_dividends_and_splits(folder_name):
    res = {}
    all_files = glob.glob(os.path.join(folder_name+"/","*"))
    for file in all_files:
        res.update({os.path.basename(file): pd.read_csv(file).dropna().reset_index(drop=True)})
    return res


def calcular_predicciones_normalizadas(quotes, predicciones, metodo, n, volatilidad_minima):
    """
    :param metodo: va a ser o metodo = pd.Series.median o metodo = pd.Series.mean
    :param volatilidad_minima: este parametro se utiliza para si por ejemplo estoy utilizando
                               una N pequeña (n=3 por ejemplo) y justamente en los ultimos 3 meses
                               la accion casi no se ha movido y ha tenido una volatilidad demasiado
                               pequeña, pues yo no quiero que se invierta tanto en esa accion, aunque
                               su volatilidad haya sido tan pequeña, ya que eso habra sido casualidad
                               de los ultimos N meses, pero minimo minimo cualquier accion tiene un
                               riesgo minimo de que en el proximo mes puede pasar lo que sea, asi que
                               le pas este parametro y cuanto mas alto pues menos sera la diferencia
                               entre lo que invierto en una accion y otra.
                               Si le paso un valor de 0 pues obviamente este parametro dejaria de tener
                               efecto. Lo suyo seria pasarle la volatiliad media mensual de cualquier
                               accion, que no sé si sera de entorno a un 4% o así.
    """
    c = ['min','cierre','max']
    pred_norm = pd.DataFrame(index=predicciones.index, columns=predicciones.columns)
    pred_norm.loc[:,['mes', 'direccion', 'accion','Date']] = predicciones.loc[:,['mes', 'direccion', 'accion','Date']]
    for idx, row in predicciones.iterrows():
        if row.accion != 'IBEX':
            q = quotes.get(row.accion.upper()+'.MC')
            q_mes = q.loc[q.Date == row.Date]
            if len(q_mes) == 0:
                print(q_mes, idx, row.accion)
                raise IndexError("No se ha encontrado la cotizacion de ese mes.")
            
            ## calculo la volatilidad segun el metodo que quiera
            if 'var' not in q.columns:
                q['var'] = (q.Close / q.Open - 1) *100
            volatilidad_media = metodo(abs(q['var'].iloc[max(q_mes.index.item()-n,0):q_mes.index.item()]))
            #print(row.mes, row.accion, volatilidad_media)
            volatilidad_media = max(volatilidad_media, volatilidad_minima)
            
            pred_norm.iloc[idx]['min'] = predicciones.iloc[idx]['min'] / volatilidad_media
            pred_norm.iloc[idx]['cierre'] = predicciones.iloc[idx]['cierre'] / volatilidad_media
            pred_norm.iloc[idx]['max'] = predicciones.iloc[idx]['max'] / volatilidad_media
    pred_norm[c] = pred_norm[c].apply(pd.to_numeric, errors='coerce')
    return pred_norm

def calcular_predicciones_con_SL_y_TP(predicciones, is_pred_baja, stop, profit):
    """
    En caso de que en un mes se toque tanto el stop como el profit, damos
    prioridad al stop, asi que nos ponemos en el peor de los casos posibles.
    SOLO MODIFICAMOS LA COLUMNA 'cierre', las columnas 'min' y 'max' quedan
    iguales aunque se haya ejecutado el stop o profit, por tanto sus datos
    dejan de ser "reales".
    
    :param is_pred_baja: hay que pasarle un True si las 'predicciones' son del lado bajista
                         ya que en ese caso hay que tener en cuenta que la columna 'cierre'
                         este en negativo en realidad es que hemos acertado, y el stop y profit
                         hay que palicarlos al reves.
    :param stop: numero en negativo, pasarle -1000 si no quiero poner stop
                    (y el numero esta en porcentaje, que es como estan las
                    columnas de 'min','cierre','max')
    :param profit: igual que el param stop, pero en positivo.
    """
    c = ['min','cierre','max']
    pred_sl_pf = pd.DataFrame(index=predicciones.index, columns=predicciones.columns)
    pred_sl_pf.loc[:,['mes', 'direccion', 'accion','min','max','Date']] = predicciones.loc[:,['mes', 'direccion', 'accion','min','max','Date']]
    for idx, row in predicciones.iterrows():
        if row.accion != 'IBEX':
            pred_sl_pf.iloc[idx]['cierre'] = predicciones.iloc[idx]['cierre']
            if is_pred_baja:
                if predicciones.iloc[idx]['max'] >= -stop:
                    pred_sl_pf.iloc[idx]['cierre'] = -stop
                elif predicciones.iloc[idx]['min'] <= -profit:
                    pred_sl_pf.iloc[idx]['cierre'] = -profit
            else:
                if predicciones.iloc[idx]['min'] <= stop:
                    pred_sl_pf.iloc[idx]['cierre'] = stop
                elif predicciones.iloc[idx]['max'] >= profit:
                    pred_sl_pf.iloc[idx]['cierre'] = profit
            
    pred_sl_pf[c] = pred_sl_pf[c].apply(pd.to_numeric, errors='coerce')
    return pred_sl_pf



def calcular_means_mensuales_con_conjunta(predicciones_alza, predicciones_baja):
    pa = predicciones_alza.groupby("mes", sort=False)
    pb = predicciones_baja.groupby("mes", sort=False)    
    
    pa_mean = pa.mean()
    pb_mean = pb.mean()
    pa_mean.columns = 'a_' + pa_mean.columns
    pb_mean.columns = 'b_' + pb_mean.columns
    
    means = meses.join([pa_mean, pb_mean])
    means['conj'] = (means.a_cierre.fillna(0)-means.b_cierre.fillna(0))/2
    return means


def print_retornos_info(titulo, df):
    #Le paso una Series con los meses como indices y los retornos mensuales
    #como columna y me imprime la ganancia el ratio, el mayor DD, etc
    df = df.to_frame()
    df['Cumulative'] = df.cumsum().round(2)
    df['HighValue'] = df['Cumulative'].cummax()    
    df['Drawdown'] = df['Cumulative'] - df['HighValue']
    print(titulo, "%:", df.Cumulative[-1], "DD:", df.Drawdown.min().round(2), "Ratio:",
          ((df.Cumulative[-1]/(len(df)/12))/-df.Drawdown.min()).round(2))

#def main():
#    get_all_pred_historical_data("historical_data_NOT_adjusted", False)
#  
#if __name__== "__main__":
#  main()


quotes = load_all_yahoo_quotes_without_dividends_and_splits("historical_data_adjusted")

meses = pd.read_csv("meses_ordenados.txt", index_col=0)
predicciones_alza = pd.read_csv("predicciones_alza.txt", delimiter="\t")
predicciones_baja = pd.read_csv("predicciones_baja.txt", delimiter="\t")
predicciones_alza = predicciones_alza.merge(meses, on='mes', how='left')
predicciones_baja = predicciones_baja.merge(meses, on='mes', how='left')

means = calcular_means_mensuales_con_conjunta(predicciones_alza, predicciones_baja)

##sns.lineplot(x=meses.mes, y=pa_mean.cierre.cumsum(), sort=False)
##sns.lineplot(x=meses.mes, y=pb_mean.cierre.cumsum(), sort=False)
#plt.figure(figsize=(9,4))
#plt.plot(means.a_cierre.cumsum().fillna(method='ffill'))
#plt.plot(means.b_cierre.cumsum().fillna(method='ffill'))
#plt.xticks(rotation=70)
#plt.tick_params(axis='x', which='major', labelsize=7)
#plt.plot()
#plt.show()


"""
#Aqui voy a calcular los resultados que darian de normal cogiendo las quotes
#que tengo descargadas en vez de los datos que yo tome a mano en el excel.
c = ['min','cierre','max']
predicciones_alza_2 = pd.DataFrame(index=predicciones_alza.index, columns=predicciones_alza.columns)
predicciones_alza_2.loc[:,['mes', 'direccion', 'accion','Date']] = predicciones_alza.loc[:,['mes', 'direccion', 'accion','Date']]
for idx, row in predicciones_alza.iterrows():
    if row.accion != 'IBEX':
        q = quotes.get(row.accion.upper()+'.MC')
        q_mes = q.loc[q.Date == row.Date]
        if len(q_mes) == 0:
            print(q_mes, idx, row.accion)
            raise IndexError("No se ha encontrado la cotizacion de ese mes.")
        predicciones_alza_2.iloc[idx]['min'] = (q_mes.Low.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
        predicciones_alza_2.iloc[idx]['cierre'] = (q_mes.Close.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
        predicciones_alza_2.iloc[idx]['max'] = (q_mes.High.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
predicciones_alza_2[c] = predicciones_alza_2[c].apply(pd.to_numeric, errors='coerce')

predicciones_baja_2 = pd.DataFrame(index=predicciones_baja.index, columns=predicciones_baja.columns)
predicciones_baja_2.loc[:,['mes', 'direccion', 'accion','Date']] = predicciones_baja.loc[:,['mes', 'direccion', 'accion','Date']]
for idx, row in predicciones_baja.iterrows():
    if row.accion != 'IBEX':
        q = quotes.get(row.accion.upper()+'.MC')
        q_mes = q.loc[q.Date == row.Date]
        if len(q_mes) == 0:
            print(q_mes, idx, row.accion)
            raise IndexError("No se ha encontrado la cotizacion de ese mes.")
        predicciones_baja_2.iloc[idx]['min'] = (q_mes.Low.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
        predicciones_baja_2.iloc[idx]['cierre'] = (q_mes.Close.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
        predicciones_baja_2.iloc[idx]['max'] = (q_mes.High.item() - q_mes.Open.item()) / q_mes.Open.item() * 100
predicciones_baja_2[c] = predicciones_baja_2[c].apply(pd.to_numeric, errors='coerce')

difs_alza = predicciones_alza.iloc[:,3:6] - predicciones_alza_2.iloc[:,3:6]
difs_baja = predicciones_baja.iloc[:,3:6] - predicciones_baja_2.iloc[:,3:6]

##
pa_2 = predicciones_alza_2.groupby("mes", sort=False)
pb_2 = predicciones_baja_2.groupby("mes", sort=False)

pa_mean_2 = pa_2.mean()
pb_mean_2 = pb_2.mean()
pa_mean_2.columns = 'a_' + pa_mean_2.columns
pb_mean_2.columns = 'b_' + pb_mean_2.columns

means_2 = meses.join([pa_mean_2, pb_mean_2])
##
"""




#fig, ax1 = plt.subplots(figsize=(9,4))
#ax1.plot(means.a_cierre.cumsum().fillna(method='ffill'), color='b')
#ax1.plot(means.b_cierre.cumsum().fillna(method='ffill'), color='r')
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#ax2.plot(means_norm.a_cierre.cumsum().fillna(method='ffill'), color='c') #means_2.a_cierre.cumsum() or means_norm.a_cierre.cumsum()
#ax2.plot(means_norm.b_cierre.cumsum().fillna(method='ffill'), color='m') #means_2.b_cierre.cumsum() or means_norm.b_cierre.cumsum()
#ax1.tick_params(axis='x', which='major', labelsize=7, rotation=70)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()



print_retornos_info("res A:", means.a_cierre)
print_retornos_info("res B:", means.b_cierre*-1)
print_retornos_info("res Conj:", means.conj)
plt.plot(means.conj.cumsum().fillna(method='ffill'))
plt.title("resultados")
plt.show()

metodo = pd.Series.median
n = 7
volt_min = 2  # volatilidad_minima
predicciones_alza_norm = calcular_predicciones_normalizadas(quotes, predicciones_alza, metodo, n, volt_min)
predicciones_baja_norm = calcular_predicciones_normalizadas(quotes, predicciones_baja, metodo, n, volt_min)
means_norm = calcular_means_mensuales_con_conjunta(predicciones_alza_norm, predicciones_baja_norm)
print(n)
print_retornos_info("res_norm A:", means_norm.a_cierre)
print_retornos_info("res_norm B:", means_norm.b_cierre*-1)
print_retornos_info("res_norm Conj:", means_norm.conj)
plt.plot(means_norm.conj.cumsum().fillna(method='ffill'))
#plt.plot(means.a_cierre.cumsum().fillna(method='ffill'), color='b')
#plt.plot(means_norm.a_cierre.cumsum().fillna(method='ffill'), color='c')
plt.title("resultados normalizados por volatilidad")
plt.show()
    
    
#for n in list(range(2,11))+[12,15,20,30,50]:
#    predicciones_alza_norm = calcular_predicciones_normalizadas(quotes, predicciones_alza, metodo, n, volt_min)
#    predicciones_baja_norm = calcular_predicciones_normalizadas(quotes, predicciones_baja, metodo, n, volt_min)
#    means_norm = calcular_means_mensuales_con_conjunta(predicciones_alza_norm, predicciones_baja_norm)
#    print(n)
#    print_retornos_info("res_norm A:", means_norm.a_cierre)
#    print_retornos_info("res_norm B:", means_norm.b_cierre*-1)
#    print_retornos_info("res_norm Conj:", means_norm.conj)


sl = -3.5
pf = 1000
pred_alza_sl_pf = calcular_predicciones_con_SL_y_TP(predicciones_alza, False, sl, pf)
pred_baja_sl_pf = calcular_predicciones_con_SL_y_TP(predicciones_baja, True, sl, pf)
means_sl_pf = calcular_means_mensuales_con_conjunta(pred_alza_sl_pf, pred_baja_sl_pf)
print_retornos_info("res_sl_pf A:", means_sl_pf.a_cierre)
print_retornos_info("res_sl_pf B:", means_sl_pf.b_cierre*-1)
print_retornos_info("res_sl_pf Conj:", means_sl_pf.conj)
plt.plot(means_sl_pf.conj.cumsum().fillna(method='ffill'))
plt.title("resultados con stop y profit")
plt.show()

sl2 = -1
pf2 = +1000  #+10
#for sl2 in [-0.5,-0.75,-1,-1.25,-1.5]:  # [x*-1.0 for x in range(1,6*1)]
#    for pf2 in [2,3,4,5,6,7,8,9,10,15]+[1000]:  # [j*1.0 for j in range(1,10*1)]
pred_alza_norm_sl_pf = calcular_predicciones_con_SL_y_TP(predicciones_alza_norm, False, sl2, pf2)
pred_baja_norm_sl_pf = calcular_predicciones_con_SL_y_TP(predicciones_baja_norm, True, sl2, pf2)
means_norm_sl_pf = calcular_means_mensuales_con_conjunta(pred_alza_norm_sl_pf, pred_baja_norm_sl_pf)
print(n, sl2, pf2)
print_retornos_info("res_norm_sl_pf A:", means_norm_sl_pf.a_cierre)
print_retornos_info("res_norm_sl_pf B:", means_norm_sl_pf.b_cierre*-1)
print_retornos_info("res_norm_sl_pf Conj:", means_norm_sl_pf.conj)
plt.plot(means_norm_sl_pf.conj.cumsum().fillna(method='ffill'))
#plt.plot(means_norm_sl_pf.a_cierre.cumsum().fillna(method='ffill'), color='b')
#plt.plot(means_norm_sl_pf.a_cierre.cumsum().fillna(method='ffill'), color='c')
plt.title("resultados normalizados con stop y profit")
plt.show()



### AQUI LO QUE DEBERIA HACER ES VER LOS RESULTADOS PERO QUITANDO LOS MESES
### EN LOS QUE PREDIGO QUE EL IBEX VA A SUBIR O BAJAR YA QUE SE HA VISTO QUE
### ESTA PREDICCION SE FALLA MUCHO, DE HECHO SUELE PASAR JUSTO LO CONTRARIO
### A LO QUE HE PREDICHO, Y PUEDE TENER SENTIDO, YA QUE ESA PREDICCION SOLO LA
### PONGO CUANDO SE VE MUY CLARO EN EL GRAFICO, Y YA SE SABE QUE CUANDO ALGO
### SE VE MUY CLARO/PARECE MUY OBVIO, ACABA PASANDO LO CONTRARIO.
"""
pai = predicciones_alza[predicciones_alza.accion == 'IBEX']
pbi = predicciones_baja[predicciones_baja.accion == 'IBEX']

means.loc[means.Date.isin(pai.Date) | means.Date.isin(pbi.Date), ['a_cierre','b_cierre','conj']] = 0
    
print_retornos_info("res_norm_sl_pf A:", means_norm_sl_pf.a_cierre)
print_retornos_info("res_norm_sl_pf B:", means_norm_sl_pf.b_cierre*-1)
print_retornos_info("res_norm_sl_pf Conj:", means_norm_sl_pf.conj)
plt.plot(means_norm_sl_pf.conj.cumsum().fillna(method='ffill'))
plt.title("resultados normalizados con stop y profit Y SIN MESES DE IBEX")
plt.show()
"""
