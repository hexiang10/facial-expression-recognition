import os
import pandas as pd

def image_emotion_mapping(path):
    # 读取emotion文件
    df_emotion = pd.read_csv('dataset/emotion.csv', header = None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(path)
    # 用于存放图片名
    path_list = []
    # 用于存放图片对应的emotion
    emotion_list = []
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的emotion取出，分别放入path_list和emotion_list这两个列表中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            emotion_list.append(df_emotion.iat[index, 0])

    # 将两个列表写进image_emotion.csv文件
    path_s = pd.Series(path_list)
    emotion_s = pd.Series(emotion_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(path+'\\image_emotion.csv', index=False, header=False)


def main():
    # 指定文件夹路径
    train_set_path = 'face_images/train_set'
    verify_set_path = 'face_images/verify_set'
    image_emotion_mapping(train_set_path)
    image_emotion_mapping(verify_set_path)

if __name__ == "__main__":
    main()