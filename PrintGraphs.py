import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------- FED Tuning ----------------------------------------------------------------------------------------#

#---------- IID----------#

#---------- Import DataBase ----------#
tuning1 = pd.read_csv("Results\Federated_Non-IID:False_LocalEpochs:2_Lr:0.001_momentum:0_wd:0_batchSize:32.csv")
tuning2 = pd.read_csv("Results\Federated_Non-IID:False_LocalEpochs:2_Lr:0.01_momentum:0_wd:0_batchSize:32.csv")


#---------- Plot the graphs ----------#
plt.figure(figsize=(10,10))
plt.plot(tuning1["Epochs"],tuning1["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'tuning1')
plt.plot(tuning2["Epochs"],tuning2["Test accuracy"],marker = "o",markersize=3.5,linestyle = "--", label = 'tuning2')
plt.ylim(bottom = 0)
plt.title("Comparison between methods with IID in FedAdp settings")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.savefig("Images/Comparison between methods with IID in FedAdp settings.png")
plt.show()