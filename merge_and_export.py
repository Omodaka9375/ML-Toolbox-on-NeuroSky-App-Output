import pandas as pd
import glob

path =r'C:\Users\brani\Desktop\ML research projects\DeepMagicAI\labeled_data' # use your path
allFiles = glob.glob(path + "/*.csv")

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)

frame = pd.concat(list_, axis = 0, ignore_index = True)

frame.drop('blinkStrength', axis=1, inplace=True)
frame.drop('theta', axis=1, inplace=True)
frame.drop('alphaLow', axis=1, inplace=True)
frame.drop('alphaHigh', axis=1, inplace=True)

frame.drop('betaLow', axis=1, inplace=True)
frame.drop('betaHigh', axis=1, inplace=True)
frame.drop('gammaLow', axis=1, inplace=True)
frame.drop('gammaMid', axis=1, inplace=True)

frame.to_csv('all.csv', index=False)

# csv_input = pd.read_csv('./test_data/trougao.csv')
# csv_input.drop('timestampMs', axis=1, inplace=True)
# csv_input.drop('poorSignal', axis=1, inplace=True)
# csv_input.drop('tagEvent', axis=1, inplace=True)
# csv_input.drop('location', axis=1, inplace=True)
# csv_input.to_csv('./clean_test_data/triangle.csv', index=False)