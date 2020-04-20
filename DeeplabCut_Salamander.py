import deeplabcut
task = 'Salamander'
experimenter = 'Brijesh'
video = ['videos/reachingvideo1.avi']

#path_config_file = deeplabcut.create_new_project(task,experimenter,video,copy_videos=True)
path_config_file = 'D:/Pycharm_Projects/TryDeepLabCut/Salamander-Brijesh-2020-04-18/config.yaml'
print(path_config_file)

#deeplabcut.extract_frames(path_config_file)

#deeplabcut.label_frames(path_config_file)
#deeplabcut.check_labels(path_config_file)

#deeplabcut.create_training_dataset(path_config_file)

#deeplabcut.train_network(path_config_file, saveiters=3000, maxiters=12000, displayiters=500)

#deeplabcut.evaluate_network(path_config_file, plotting=True)

#deeplabcut.analyze_videos(path_config_file,['Small_Salamander/Video/Salamander_Video.avi'], videotype='.avi', save_as_csv=True)

deeplabcut.create_labeled_video(path_config_file,['D:/Pycharm_Projects/TryDeepLabCut/Small_Salamander/Video/Salamander_Video.avi'], save_frames=True)