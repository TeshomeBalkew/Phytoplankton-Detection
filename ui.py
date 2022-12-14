import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd

class_names = ['Akashiwo', 'amoeba', 'Amphidinium_sp', 'Asterionellopsis', 'bad', 'bead', 'Cerataulina', 'Cerataulina_flagellate', 'Ceratium', 'Chaetoceros', 'Chaetoceros_didymus', 'Chaetoceros_didymus_flagellate', 'Chaetoceros_flagellate', 'Chaetoceros_other', 'Chaetoceros_pennate', 'Chrysochromulina', 'Ciliate_mix', 'clusterflagellate', 'Cochlodinium', 'Corethron', 'Coscinodiscus', 'Cylindrotheca', 'DactFragCerataul', 'Dactyliosolen', 'Delphineis', 'detritus', 'diatom_flagellate', 'Dictyocha', 'Didinium_sp', 'dino30', 'Dinobryon', 'Dinophysis', 'dino_large1', 'Ditylum', 'Ditylum_parasite', 'Emiliania_huxleyi', 'Ephemera', 'Eucampia', 'Euglena', 'Euplotes_sp', 'flagellate_sp3', 'Guinardia_delicatula', 'Guinardia_flaccida', 'Guinardia_striata', 'Gyrodinium', 'G_delicatula_detritus', 'G_delicatula_external_parasite', 'G_delicatula_parasite', 'Heterocapsa_triquetra', 'Katodinium_or_Torodinium', 'kiteflagellates', 'Laboea_strobila', 'Lauderia', 'Leegaardiella_ovalis', 'Leptocylindrus', 'Leptocylindrus_mediterraneus', 'Licmophora', 'Mesodinium_sp', 'mix_elongated', 'Odontella', 'other_interaction', 'Paralia', 'Parvicorbicula_socialis', 'pennate', 'pennates_on_diatoms', 'pennate_morphotype1', 'Phaeocystis', 'Pleuronema_sp', 'Pleurosigma', 'pollen', 'Prorocentrum', 'Proterythropsis_sp', 'Protoperidinium', 'Pseudochattonella_farcimen', 'Pseudonitzschia', 'Pyramimonas_longicauda', 'Rhizosolenia', 'Skeletonema', 'spore', 'Stephanopyxis', 'Strobilidium_morphotype1', 'Strombidium_capitatum', 'Strombidium_conicum', 'Strombidium_inclinatum', 'Strombidium_morphotype1', 'Strombidium_morphotype2', 'Strombidium_oculatum', 'Thalassionema', 'Thalassiosira', 'Thalassiosira_dirty', 'Tintinnid', 'Tontonia_gracillima', 'zooplankton']

model = load_model('bcnn2014.h5')
file1 = 'IFCB1_2006_158_000036_01314 copy.png'
file2 = 'IFCB1_2006_237_014941_00251.png'
file_pixel_data = []

oimg = cv2.imread(file2, 0)
cv2.imshow('Window', oimg)
cv2.waitKey(0)
img = cv2.resize(oimg,(28,28))
for i in range (img.shape[0]):
        for j in range (img.shape[1]):
                k = img[i][j]
                file_pixel_data.append(k)

formatdata = pd.DataFrame(file_pixel_data).T
colname = []
for val in range(784):
    colname.append(val)
formatdata.columns = colname
pixeldata = formatdata.values
pixeldata = pixeldata / 255
pixeldata = pixeldata.reshape(-1,28,28,1)

prediction = model.predict(pixeldata)
predarray = np.array(prediction[0])

classpred = class_names[np.argmax(predarray)]
print(classpred)
val = round(100*max(predarray),3)
print("Confidence: {}%".format(val))  