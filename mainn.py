import numpy as np
import datetime
import pandas as pd
data1={'sensor1':[],'sensor2':[],'sensor3':[],'time':[]}
now=datetime.datetime.now()                      		        
intime = now.strftime("%y-%m-%d %H:%M:%S")
data1["time"].append(str(intime))
sens1=np.random.randint(1,10)
data1["sensor1"].append(str(sens1))
sens2=np.random.randint(1,10)
data1["sensor2"].append(str(sens2))
sens3=np.random.randint(1,10)
data1["sensor3"].append(str(sens3))
print(data1)
