import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
from scipy import meshgrid
import math
import os
import PySimpleGUI as sg
import os.path


if __name__ == '__main__':

# ----- Full layout -----
 layout = [
		[
       		 sg.Text("Image Folder"),
       		 sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
       		 sg.FolderBrowse(),
       
   		],
   		[
        	 sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"), sg.Image(key="-IMAGE-")
       	],
        [sg.ReadFormButton('OK', size=(8,2)),],
]

window = sg.Window("Image Viewer", layout)
 
# Run the Event Loop
while True:
     event, values = window.read()
     if event == "Exit" or event == sg.WIN_CLOSED: 
         break
    # Folder name was filled in, make a list of files in the folder
     if event == "-FOLDER-":
         folder = values["-FOLDER-"]
         try:
            # Get list of files in folder
            file_list = os.listdir(folder)
         except:
            file_list = []

         fnames = [
                f 
                for f in file_list 
                if os.path.isfile(os.path.join(folder, f)) 
                and f.lower().endswith((".png"))
            ]
         window["-FILE LIST-"].update(fnames)
     elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-IMAGE-"].update(filename=filename)
        except:
            pass
     elif event == 'OK':
         break

window.close()
   # imfile = r'd:\Repos\feuersalamander_test_20200108\Specimen_2_adult.jpg'
print(filename)
imfile= filename
out_path = os.path.split(imfile)[0]
mark_coords = False
num_samples = 500
perc_width = 0.3


img = cv2.imread(imfile)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



if mark_coords:
    fig = plt.figure()
    plt.imshow(img)

    coords = []


    def onclick(event):
        global coords
        coords.append((event.xdata, event.ydata))


    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    coords = np.array(coords, np.float32)
else:
    coords = np.array([(3156.3052210681135, 1291.3304991267582), (3152.334267855502, 1335.010984465484),
                  (3148.363314642891, 1398.5462358672671), (3128.5085485798336, 1446.1976744186047),
                  (3100.7118760915537, 1513.7038790329993), (3072.915203603273, 1553.413411159114),
                  (3053.0604375402154, 1597.09389649784), (3033.205671477159, 1648.7162882617888),
                  (3001.4380457762672, 1696.3677268131262), (2957.7575604375406, 1755.9320250022981),
                  (2914.077075098815, 1811.5253699788584), (2854.5127769096434, 1851.234902104973),
                  (2771.122759444803, 1851.234902104973), (2675.819882342128, 1815.4963231914699),
                  (2620.2265373655673, 1763.873931427521), (2560.662239176396, 1732.1063057266294),
                  (2513.0108006250584, 1708.2805864509608), (2437.5626895854402, 1668.571054324846),
                  (2338.288859270154, 1632.832475411343), (2262.8407482305365, 1601.0648497104514),
                  (2203.276450041364, 1577.2391304347827), (2111.944526151301, 1573.2681772221713),
                  (2008.6997426234032, 1573.2681772221713), (1917.3678187333396, 1577.2391304347827),
                  (1826.035894843276, 1589.151990072617), (1734.703970953213, 1612.9777093482858),
                  (1647.3430002757607, 1616.9486625608972), (1579.8367956613663, 1636.8034286239545),
                  (1476.5920121334682, 1652.6872414744003), (1405.1148543064621, 1668.571054324846),
                  (1345.5505561172902, 1688.4258203879033), (1278.0443515028958, 1716.2224928761837),
                  (1190.6833808254435, 1740.0482121518523), (1119.2062229984374, 1759.9029782149096),
                  (1063.6128780218771, 1787.6996507031897), (976.2519073444253, 1823.4382296166928),
                  (861.094264178693, 1847.2639488923614), (761.8204338634066, 1886.9734810184761),
                  (662.5466035481202, 1930.653966357202), (523.5632411067193, 1994.1892177589853),
                  (412.3765511535987, 2049.7825627355455), (305.1608144130894, 2101.4049544994946),
                  (205.88698409780318, 2137.1435334129974), (114.55506020773964, 2164.9402059012777)], np.float32)

    # fig = plt.figure()
    # plt.imshow(img)
    # x = [x for (x,y) in coords]
    # y = [y for (x,y) in coords]
    # plt.plot( x, y )

dists = [ np.linalg.norm( np.array(curr) - np.array(prev) ) for (prev,curr) in zip( coords[:-1], coords[1:]) ]
total_dist = sum(dists)
total_width = total_dist * perc_width
acc_dists = np.cumsum(dists)

print(dists)
sample_stepsize_in_pix = total_dist * num_samples

num_orth_samples = math.floor(perc_width * num_samples)
sample_coords = np.zeros([num_samples, num_orth_samples, 2], np.float32)

for curr_sample in range(num_samples):
    curr_dist = curr_sample / num_samples * total_dist
    larger_idxs = np.argwhere( acc_dists > curr_dist )
    if len(larger_idxs) > 0:
        next_idx = int(larger_idxs[0])
        curr_idx = int(larger_idxs[0] - 1)
        if curr_idx >= 0:
            curr_coord = ((curr_dist - acc_dists[curr_idx]) / (acc_dists[next_idx] - acc_dists[curr_idx])) * coords[next_idx] \
                             + ((acc_dists[next_idx] - curr_dist) / (acc_dists[next_idx] - acc_dists[curr_idx])) * coords[curr_idx]
            curr_tang_vec = coords[next_idx] - coords[curr_idx]
            curr_orth_tang_vec = np.array([-curr_tang_vec[1], curr_tang_vec[0]])
            curr_orth_tang_vec = curr_orth_tang_vec / np.linalg.norm(curr_orth_tang_vec)
            for curr_orth_sample in range( num_orth_samples ):
                curr_sample_pnt = curr_coord - total_width / 2 * curr_orth_tang_vec + curr_orth_sample / num_orth_samples * total_width * curr_orth_tang_vec
                sample_coords[curr_sample, curr_orth_sample, :] = curr_sample_pnt

print(sample_coords)

x, y = np.arange(0, img.shape[0]), np.arange(0, img.shape[1])

img_func_r = interpolate.interp2d(y, x, np.squeeze(img[:, :, 0]), kind='cubic')
img_func_g = interpolate.interp2d(y, x, np.squeeze(img[:, :, 1]), kind='cubic')
img_func_b = interpolate.interp2d(y, x, np.squeeze(img[:, :, 2]), kind='cubic')

sampled_img = np.zeros( [num_samples, num_orth_samples, 3], np.float32 )

for sampled_img_r in range(num_samples):
    for sampled_img_c in range(num_orth_samples):
        sampled_img[sampled_img_r, sampled_img_c, :] = [ img_func(*sample_coords[sampled_img_r, sampled_img_c, :]) for img_func in [img_func_r, img_func_g, img_func_b] ]

plt.figure()
plt.imshow(img)
plt.plot( coords[:, 0].flatten(), coords[:, 1].flatten(), 'bo' )

plt.savefig(out_path + '/feuersalamender_profile.png')
window_name = 'image'
cv2.imshow(window_name, img)
plt.figure()
plt.imshow(sampled_img.astype(np.uint8))

plt.savefig(out_path + '/feuersalamender_rectified.png')