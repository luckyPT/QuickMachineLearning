"""
argoverse-api only surport ubuntu or mac;
copy map-files to:
/home/pan/anaconda3/envs/vectornet/lib/python3.8/site-packages
"""
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap

from src.self_drive.track_forecasting.eval_util import ade, fde
from src.self_drive.track_forecasting.simple_rule import prediction_rule
import src.self_drive.track_forecasting.visual_util as visual_util

am = ArgoverseMap()
data_dir = '../../../data/avgoverse/motion_forecasting/data'
afl = ArgoverseForecastingLoader(data_dir)
for name in afl.seq_list:
    afl_ = afl.get(name)
    agent_traj = afl_.agent_traj
    cur_index = -20
    cur_loc = agent_traj[cur_index]
    # get lanes
    lane_ids = am.get_lane_ids_in_xy_bbox(cur_loc[0], cur_loc[1], afl_.city, 20)
    for lane_id in lane_ids:
        center_lane = am.get_lane_segment_centerline(lane_id, afl_.city)
        visual_util.plot_lane_border(center_lane)

    pre = prediction_rule(agent_traj[:cur_index], -cur_index, 5)
    label = agent_traj[cur_index:]
    print(ade(label, pre))
    print(fde(label, pre))
    # agent_traj[:cur_index-1] 这里如果设置为cur_index绘制会变乱，toDo：排查原因
    visual_util.plot_traj(agent_traj[:cur_index-1], color='#ff0000', line_width=4)
    visual_util.plot_traj(label, color='#0000ff', line_width=2)
    visual_util.plot_traj(pre, color='#00ff00', line_width=1)
    visual_util.show()
