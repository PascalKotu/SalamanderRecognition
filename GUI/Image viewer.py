import PySimpleGUI as sg
import os
from PIL import Image


# define layout, show and read the form
col = [ [sg.Text("Image Folder"), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse(),],
        [sg.Image(key="-IMAGE-")],
    ]

col_files = [[sg.Listbox(values=[], enable_events=True, size=(60,30), key='listbox')],
            ]

layout = [[sg.Column(col_files), sg.Column(col)]]

# create the form that also returns keyboard events

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    #collects the events
    event, values = window.read()
    #execution based on events
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        #print(folder)
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
            #print(file_list)
            #print("filelist")
            # get list of jpg files in folder
            jpgcheck=[]
            for f in file_list:
                #print("this works")
                filename, extension = os.path.splitext(f)
                #print(extension)
                #convert jpg files to png
                if extension == '.JPG' or extension=='.jpg':
                    jpgcheck1=os.path.join(folder,f)
                    #using pil package to convert jpg to png and saving it
                    img=Image.open(jpgcheck1,mode="r")
                    img = img.resize((512,512))
                    #print("image opned")
                    newfile=os.path.join(folder,filename)
                    newfile=newfile+'.png'
                    #print(newfile)
                    img.save(newfile)  
                    #print("new file saved")  
                    jpgcheck.append(newfile)
                    #print(newfile)
                    #if the files are already png just add them to the list
                elif extention == '.PNG' or extension == '.png':
                    newfile=os.path.join(folder,filename)     
                    jpgcheck.append(newfile)     
                    #print(newfile)       
        except:
            file_list = []
            jpgcheck = []

        #print(jpgcheck)
        #print("this happened too")
        #updating the list of files on the window
        file_list = os.listdir(folder)
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".PNG"))
        ]
        window["listbox"].update(fnames)
        #if any file on the listis selected to execute action
    elif event == "listbox":  # A file was chosen from the listbox
        try:
            #filenames_only = [f for f in file_list if '.jpg' in f]
            filename_only = os.path.join(
                values["-FOLDER-"], values["listbox"][0]
            )
            #image_elem = sg.Image(filename=jpgtopng)
            #filename_display_elem = sg.Text(png_files[0], size=(80, 3))
            #file_num_display_elem = sg.Text('File 1 of {}'.format(len(png_files)), size=(15,1))
            #update selected image
            window["-IMAGE-"].update(filename = filename_only)
            
        except:
            pass
    
   
        



# make these 2 elements outside the layout because want to "update" them later
# initialize to the first PNG file in the list
#jpgtopng=Image.open(png_files[0])
#image_elem = sg.Image(filename=jpgtopng)
#filename_display_elem = sg.Text(png_files[0], size=(80, 3))
#file_num_display_elem = sg.Text('File 1 of {}'.format(len(png_files)), size=(15,1))

# define layout, show and read the form
#col = [[filename_display_elem],
#          [image_elem],
#          [sg.ReadFormButton('Next', size=(8,2)), sg.ReadFormButton('Prev', size=(8,2)), file_num_display_elem]]
#
#col_files = [[sg.Listbox(values=filenames_only, size=(60,30), key='listbox')],
#             [sg.ReadFormButton('Read')]]
#layout = [[sg.Column(col_files), sg.Column(col)]]
#button, values = form.LayoutAndRead(layout)          # Shows form on screen

# loop reading the user input and displaying image, filename
#i=0
#while True:

    # perform button and keyboard operations
 #   if button is None:
  #      break
   # elif button in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34') and i < len(png_files)-1:
 #       i += 1
  #  elif button in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33') and i > 0:
   #     i -= 1

    #if button == 'Read':
     #   filename = folder + '\\' + values['listbox'][0]
        # print(filename)
  #  else:
   #     jpgtopng = Image.open(png_files[i])
    #    filename = jpgtopng

    # update window with new image
 #   image_elem.Update(filename=filename)
    # update window with filename
  #  filename_display_elem.Update(filename)
    # update page display
   # file_num_display_elem.Update('File {} of {}'.format(i+1, len(png_files)))

    # read the form
   # button, values = form.Read()
