import ipytest
import nbimporter
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

os.chdir("C:/Users/Utilisateur/PROJET7") # Je vais chercher les fonctions dans le fichier placé dans PROJET7
from Renard_Anthony_2_scripts_042023 import confusion

os.chdir("C:/Users/Utilisateur/PROJET7/tests") #Je me replace dans le dossier où est situés mes tests


def test_confusion_matrix():
    y_test=[0,0,1,1,0,0,1,1]
    y_pred=[0,1,0,1,0,0,1,1]
    
    df = pd.DataFrame({'pred_0': pd.Series([3, 1], index = ['test_0', 'test_1']), 'pred_1': pd.Series([1, 3], index = ['test_0', 'test_1'])})
    mat = confusion(y_test,y_pred)
    pd.testing.assert_frame_equal(df,mat)
    

    
    
    
    
    