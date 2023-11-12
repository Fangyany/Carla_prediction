class CarlaDataset(Dataset):
    def __init__(self, traj_path, map_path):
        traj_file = './trajectories.pkl'
        map_file = './map_dict.pkl'
        self.data = pickle.load(open(traj_file, 'rb'))
        self.map_dict = pickle.load(open(map_file, 'rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        traj, label = self.data[index]
        # 获取target第19帧的坐标
        target_coordinate = traj[0][-1][:2].numpy().tolist()
        # 计算每个字典中坐标与 target_coordinate 的欧氏距离
        distances = {key: np.linalg.norm(np.array(target_coordinate) - np.array(key)) for key in map_dict.keys()}
        # 找到最小距离对应的字典的value
        closest_coordinate = min(distances, key=distances.get)
        closest_value = map_dict[closest_coordinate]

        lane_list = [torch.tensor(df[['x', 'y']].values, dtype=torch.float32) for df in closest_value]
        traj = torch.tensor(traj.copy())
        label = torch.tensor(label.copy())
        return traj, lane_list, label