#This script is adapted from https://github.com/jedeschaud/kitti_carla_simulator
#with changes made to save the Motion Segmentation ground truth based on 
#Vehicle and pedestrian semantic tags

import glob
import os
from turtle import pos

#Not necessary for pip installed Carla 0.9.12+
# import sys
# from pathlib import Path
# try:
#     sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
import time
from datetime import date
from modules import generator_KITTI as gen

import numpy as np
import json
import logging
from datetime import datetime
import sys

def main():
    start_record_full = time.time()

    # fps_simu = 1000.0
    fps_simu = 200
    time_stop = 2.0
    nbr_frame = 15000 #MAX = 10000
    nbr_walkers = 0
    nbr_vehicles = 125

    actor_list = []
    vehicles_list = []
    all_walkers_id = []

    root = "/Carla_Data_Collection"
    sequence_offset =  0 if len(os.listdir(os.path.join(root, "images"))) == 0 else int(max(os.listdir(os.path.join(root, "images"))))+1

    init_settings = None

    sequences=50
    spawn_pts_len = 265 #for Town03
    #Check spawn points present in folder structure, increment until existing
    spawn_points = np.random.choice(spawn_pts_len,sequences, replace=False)

    now = datetime.now()
    now_string = now.strftime(f"%d-%m-%Y_%H-%M")
    log_root = os.path.join(root, "logs")
    os.makedirs(log_root) if not os.path.exists(log_root) else print("Log root already exists")
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
    ])
    logger = logging.getLogger()


    try:
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()
        
        for i_sequence in range(sequences): 
            i_sequence += sequence_offset
            client.set_timeout(100.0)
            print(f"Starting sequence {i_sequence}")
            # world = client.load_world("Town02_Opt")
            world = client.load_world("Town03_Opt")
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)

            logger.info(f"Spawn point: {spawn_points[i_sequence]}")
            
            folder_ms_seq = os.path.join(root, "motion_segmentation", str('%04d' %(i_sequence)))
            folder_rgb_seq = os.path.join(root, "images", str('%04d' %(i_sequence)))
            folder_ss_seq = os.path.join(root, "semantic_segmentation", str('%04d' %(i_sequence)))
            folder_depth_seq = os.path.join(root, "depth", str('%04d' %(i_sequence)))
            folder_opt_flow_seq = os.path.join(root, "opt_flow", str('%04d' %(i_sequence)))

            os.makedirs(folder_ms_seq) if not os.path.exists(folder_ms_seq) else print("Motion seg dir already exists")
            os.makedirs(folder_rgb_seq) if not os.path.exists(folder_rgb_seq) else print("Image dir already exists")
            os.makedirs(folder_ss_seq) if not os.path.exists(folder_ss_seq) else print("Semantic seg dir already exists")
            os.makedirs(folder_depth_seq) if not os.path.exists(folder_depth_seq) else print("Depth dir already exists")
            os.makedirs(folder_opt_flow_seq) if not os.path.exists(folder_opt_flow_seq) else print("Opt flow dir already exists")

            folder_transforms = os.path.join(root, "transformations", str('%04d' %(i_sequence)))
            os.makedirs(folder_transforms) if not os.path.exists(folder_transforms) else print("Transform dir already exists")

            # os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
            client.start_recorder(os.path.join(root,"recording.log"))
            
            # Weather
            # world.set_weather(carla.WeatherParameters.WetCloudyNoon)
            weather_list = [1,2,4,5,9,10,12,14,15,17,18,20,21]
            weather_index = np.random.choice(len(weather_list))
            world.set_weather(getattr(carla.WeatherParameters, dir(carla.WeatherParameters)[weather_list[weather_index]]))

            logger.info(f"Setting weather to: {dir(carla.WeatherParameters)[weather_list[weather_index]]}")
            
            # Set Synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0/fps_simu
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            # Create KITTI vehicle
            blueprint_library = world.get_blueprint_library()
            bp_KITTI = blueprint_library.find('vehicle.tesla.model3')
            bp_KITTI.set_attribute('color', '228, 239, 241')
            bp_KITTI.set_attribute('role_name', 'KITTI')
            print("i_seq",i_sequence)
            start_pose = world.get_map().get_spawn_points()[spawn_points[i_sequence]]
            print(f"Spawning at point {spawn_points[i_sequence]} \n x={start_pose.location.x:.2f}, y = {start_pose.location.y:.2f}, z = {start_pose.location.z:.2f}")
            KITTI = world.spawn_actor(bp_KITTI, start_pose)
            waypoint = world.get_map().get_waypoint(start_pose.location)
            actor_list.append(KITTI)
            print('Created %s' % KITTI)

            # Spawn vehicles and walkers
            gen.spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id)

            # Wait for KITTI to stop
            start = world.get_snapshot().timestamp.elapsed_seconds
            print("Waiting for KITTI to stop ...")
            while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
            print("KITTI stopped")

            # Set sensors transformation from KITTI
            cam0_transform = carla.Transform(carla.Location(x=0.50, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))

            # Take a screenshot
            # gen.screenshot(KITTI, world, actor_list, root, carla.Transform(carla.Location(x=0.0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)))

            # Create our sensors
            gen.RGB.sensor_id_glob = 0
            gen.SS.sensor_id_glob = 10
            cam0 = gen.RGB(KITTI, world, actor_list, folder_rgb_seq, cam0_transform)
            # poses = gen.Poses(cam0)
            cam0_ss = gen.SS(KITTI, world, actor_list, folder_ss_seq, cam0_transform)
            cam0_depth = gen.Depth(KITTI, world, actor_list, folder_depth_seq, cam0_transform)
            cam0_is = gen.IS(KITTI, world, actor_list, folder_ms_seq, cam0_transform)
            cam0_of = gen.OptFlow(KITTI, world, actor_list, folder_opt_flow_seq, cam0_transform)
        
            #New list with potentially moving actors (vehicles and pedestrians)
            moving_list = vehicles_list + all_walkers_id
            # moving_list = vehicles_list

            # Launch KITTI
            KITTI.set_autopilot(True)

            # Pass to the next simulator frame to spawn sensors and to retrieve first data
            world.tick()
            
            # All sensors produce first data at the same time (this ts)
            gen.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds

            poses = gen.Poses(cam0)
            
            start_record = time.time()
            print("Start record : ")
            frame_current = 0
            while (frame_current < nbr_frame):
                cam0.save()
                cam0_ss.save()
                cam0_is.save(world, moving_list, poses)
                cam0_depth.save()
                cam0_of.save()

                gen.follow(KITTI.get_transform(), world)
                frame_current += 1
                world.tick()    # Pass to the next simulator frame
            
            poses.write(folder_transforms)
            cam0_of.write(folder_opt_flow_seq)

            print('Destroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list.clear()
            
            # Stop walker controllers (list is [controller, actor, controller, actor ...])
            all_actors = world.get_actors(all_walkers_id)
            for i in range(0, len(all_walkers_id), 2):
                all_actors[i].stop()
            print('Destroying %d walkers' % (len(all_walkers_id)//2))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
            all_walkers_id.clear()
                
            print('Destroying KITTI')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            actor_list.clear()
                
            print("Elapsed time : ", time.time()-start_record)
            print()
                
            time.sleep(2.0)

    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        world.apply_settings(init_settings)
        
        time.sleep(2.0)
        

if __name__ == '__main__':
    main()