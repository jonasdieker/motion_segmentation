#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pdb

import random
import time
import pdb
import pandas as pd

def visualize_image(image):
    data = np.array(image.raw_data) # shape is (image.height * image.width * 4,) 
    data_reshaped = np.reshape(data, (image.height, image.width,4))
    rgb_3channels = data_reshaped[:,:,:3] # first 3 channels
    
    cv2.imshow("image",rgb_3channels)
    cv2.waitKey(10)

def limit_steering(steering):
    if steering > 1:
        steering = 1
    if steering < -1:
        steering = -1
    return steering

def save_image(image, image_num, data_root):
    data = np.array(image.raw_data)
    data_reshaped = np.reshape(data, (image.height, image.width, 4))
    rgb_3channels = data_reshaped[:,:,:3]
    cv2.imwrite(os.path.join(data_root, f"{image_num}.png"), np.array(rgb_3channels))
    return image_num + 1

def main(data_root, df, steer_offset, frames, first_image_num, spawn_point):
    actor_list = []

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        # pdb.set_trace()

        # Print the possible Towns we have available
        print(client.get_available_maps())
        # Once we have a client we can retrieve the world that is currently
        # running.

        # world = client.get_world()
        world = client.load_world("/Game/Carla/Maps/Town02_Opt", carla.MapLayer.NONE) #, carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # world.unload_map_layer(carla.MapLayer.Buildings)
        world.unload_map_layer(carla.MapLayer.Foliage)
        # world = client.load_world("/Game/Carla/Maps/Town04")

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('model3'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        #transform = random.choice(world.get_map().get_spawn_points())
        # Always fix the starting position
        num_spawn_points = len(world.get_map().get_spawn_points())
        if spawn_point < num_spawn_points:
            transform = world.get_map().get_spawn_points()[spawn_point]
        else:
            print("spawn point out of range!")
            sys.exit()
        # pdb.set_trace()

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # It is important to note that the actors we create won't be destroyedstart of bash srcipt
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        # camera_bp = blueprint_library.find('sensor.camera.depth')
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        # actor_list.append(camera)
        # print('created %s' % camera.type_id)
        
        frame = 0
        image_num = first_image_num + 1

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # converting the pixels to gray-scale.
        cc = carla.ColorConverter.LogarithmicDepth
        raw = carla.ColorConverter.Raw
        depth = carla.ColorConverter.Depth
        # camera.listen(lambda image: image.save_to_disk('../../saved_data/%06d.png' % frame, cc) if frame>10 else None)
        
        
        rgb_list = list()
        
        # Let's add now an "RGB" camera attached to the vehicle.
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_bp_rgb.set_attribute('image_size_x',  str(640))
        camera_bp_rgb.set_attribute('image_size_y',  str(320))
        camera_bp_rgb.set_attribute('fov',  str(100))
        camera_transform_rgb = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera_rgb = world.spawn_actor(camera_bp_rgb, camera_transform_rgb, attach_to=vehicle)
        actor_list.append(camera_rgb)
        print('created %s' % camera_rgb.type_id)
        camera_rgb.listen(lambda image: rgb_list.append(image) if frame >= 10 else None )
        #camera_rgb.listen(lambda image: image.save_to_disk(f"/storage/remote/atcremers62/Carla-scratch/{first_image_num + frame}.png"))

        rgb_left_list = list()
        # add left offset camera
        camera_transform_rgb_left = carla.Transform(carla.Location(x=2.0, y=-0.5, z=1.4))
        camera_rgb_left = world.spawn_actor(camera_bp_rgb, camera_transform_rgb_left, attach_to=vehicle)
        actor_list.append(camera_rgb_left)
        print('created %s' % camera_rgb_left.type_id)
        camera_rgb_left.listen(lambda image: rgb_left_list.append(image) if frame >= 10 else None )
        # camera_rgb_left.listen(lambda image: image.save_to_disk(f"/storage/remote/atcremers62/Carla-scratch/{first_image_num + frames + frame}.png"))


        rgb_right_list = list()
        # add right offset camera
        camera_transform_rgb_right = carla.Transform(carla.Location(x=2.0, y=0.5, z=1.4))
        camera_rgb_right = world.spawn_actor(camera_bp_rgb, camera_transform_rgb_right, attach_to=vehicle)
        actor_list.append(camera_rgb_right)
        print('created %s' % camera_rgb_right.type_id)
        camera_rgb_right.listen(lambda image: rgb_right_list.append(image) if frame >= 10 else None )
        #camera_rgb_right.listen(lambda image: image.save_to_disk(f"/storage/remote/atcremers62/Carla-scratch/{first_image_num + frames*2 + frame}.png"))


        # Oh wait, I don't like the location we gave to the vehicle, I'm going
        # to move it a bit forward.
        location = vehicle.get_location()
        location.x += 1
        vehicle.set_location(location)
        print('moved vehicle to %s' % location)

        # But the city now is probably quite empty, let's add a few more
        # vehicles.
        # transform.location += carla.Location(x=40, y=-3.2)
        # transform.location = location
        vehicle_location = vehicle.get_transform().location
        vehicle_direction = vehicle.get_transform().get_forward_vector()
        vehicle_rotation = vehicle.get_transform().rotation

        new_loc = vehicle_location + 15*vehicle_direction
        new_vehicle_loc = carla.Location(new_loc.x, new_loc.y, new_loc.z + 2)
        # transform.rotation.yaw = -180.0
        for _ in range(0, 1):
            # new_vehicle_loc.y += 10.0
        
            bp = random.choice(blueprint_library.filter('carlacola'))
        
            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, carla.Transform(new_vehicle_loc, vehicle_rotation))
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                
                # npc.apply_control(carla.VehicleControl(throttle=0.2))
                print('created %s' % npc.type_id)



        for frame in range(frames):
            # Do tick
            world.tick()
            if frame>=10:
                visualize_image(rgb_list[-1])

                if frame % 3 == 0:
                    image_num = save_image(rgb_list[-1], image_num, data_root)
                    image_num = save_image(rgb_left_list[-1], image_num, data_root)
                    image_num = save_image(rgb_right_list[-1], image_num, data_root)

                    # add csv mapping (order: straight, left, right)
                    images_lst = [f"{image_num -3}.png", f"{image_num -2}.png", f"{image_num - 1}.png"]
                    # ensure max steering commands are not exceeded
                    steering_lst = [vehicle.get_control().steer, limit_steering(vehicle.get_control().steer  + steer_offset), limit_steering(vehicle.get_control().steer - steer_offset)]
                    sensor_lst = ["straight", "left", "right"]
                    spawn_point_list = [spawn_point, spawn_point, spawn_point]
                    df_temp = pd.DataFrame({"image": images_lst, "target": steering_lst, "sensor": sensor_lst, "spawn_point": spawn_point_list})
                    df = df.append(df_temp, ignore_index=True)

            # For applying manual control. Make sure that the vehicle.set_autopilot(True) is commented out above
            # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            # Always have the traffic light on green
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        
            print('frame %s' % frame)
            print("Throttle: {}, Steering: {}".format( vehicle.get_control().throttle, vehicle.get_control().steer))
            print("Vehicle location: (x,y,z): ({},{},{})".format(vehicle.get_location().x,vehicle.get_location().y, vehicle.get_location().z ))


            #pdb.set_trace()

        #time.sleep(5)

    finally:

        print('destroying actors')
        # camera.destroy()
        camera_rgb.destroy()
        camera_rgb_left.destroy()
        camera_rgb_right.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done with scene.')

    return df


if __name__ == '__main__':

    # settings
    frames = 2000
    data_root  = "/storage/remote/atcremers40/motion_seg/Carla/test_dataset"
    csv_name = "image_target_mapping.csv"
    steer_offset = 0.5
    if int(sys.argv[1]) >= 0 and int(sys.argv[1]) < 265:
        spawn_point = int(sys.argv[1])
    else:
        spawn_point = 0

    # ensure data is not overwritten by checking last image file number
    files_dataset = sorted(os.listdir(data_root))
    if len(files_dataset) == 0:
        last_image_num = 0
    else:
        last_image_num = sorted([int(filename.split(".")[0]) for filename in files_dataset[:-2]])[-1]

    # load csv if it exists
    path_to_csv = os.path.join(data_root, csv_name)
    if os.path.isfile(path_to_csv):
        print("loading csv")
        df = pd.read_csv(path_to_csv)
    else:
        print("creating new csv")
        df = pd.DataFrame(columns=["image", "target", "sensor", "spawn_point"])

    # running main and returning dataframe
    df = main(data_root, df, steer_offset, frames, last_image_num, spawn_point)

    # write csv
    df.to_csv(path_to_csv, index=False)
