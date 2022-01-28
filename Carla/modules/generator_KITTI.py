#This script is adapted from https://github.com/jedeschaud/kitti_carla_simulator
#with changes made to save the Motion Segmentation ground truth based on 
#Vehicle and pedestrian semantic tags

#!/usr/bin/env python3

import glob
import os
import sys

#Not necessary for pip installed Carla 0.9.12+
#from pathlib import Path
# try:
#     sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
#     sys.path.append(glob.glob('../../')[0])

# except IndexError:
#     pass

import carla
from carla import VehicleLightState as vls

from PIL import Image

import logging
import queue
import numpy as np
import random
import threading

def sensor_callback(ts, sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

class Sensor:
    initial_ts = 0.0
    initial_loc = carla.Location()
    initial_rot = carla.Rotation()

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        self.queue = queue.Queue()
        self.bp = self.set_attributes(world.get_blueprint_library())
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        actor_list.append(self.sensor)
        self.sensor.listen(lambda data: sensor_callback(data.timestamp - Sensor.initial_ts, data, self.queue))
        self.sensor_id = self.__class__.sensor_id_glob;
        self.__class__.sensor_id_glob += 1
        self.folder_output = folder_output
        self.ts_tmp = 0

class Camera(Sensor):
    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Sensor.__init__(self, vehicle, world, actor_list, folder_output, transform)
        self.sensor_frame_id = 0
        self.frame_output = self.folder_output
        with open(self.folder_output+"/full_ts_camera.txt", 'w') as file:
            file.write("# frame_id timestamp\n")

        print('created %s' % self.sensor)

    def save(self, color_converter=carla.ColorConverter.Raw, motion_mask=False):
        while not self.queue.empty():
            data = self.queue.get()

            ts = data.timestamp-Sensor.initial_ts
            if ts - self.ts_tmp > 0.11 or (ts - self.ts_tmp) < 0: #check for 10Hz camera acquisition
                print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %(self.ts_tmp, ts))
                sys.exit()
            self.ts_tmp = ts

            file_path = self.frame_output+"/%04d.png" %(self.sensor_frame_id)
            x = threading.Thread(target=data.save_to_disk, args=(file_path, color_converter))
            x.start()
            print("Export : "+file_path)

            if self.sensor_id == 0:
                with open(self.folder_output+"/full_ts_camera.txt", 'a') as file:
                    file.write(str(self.sensor_frame_id)+" "+str(data.timestamp - Sensor.initial_ts)+"\n") #bug in CARLA 0.9.10: timestamp of camera is one tick late. 1 tick = 1/fps_simu seconds
            self.sensor_frame_id += 1

class RGB(Camera):
    sensor_id_glob = 0

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list, folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '1392')
        camera_bp.set_attribute('image_size_y', '1024')
        camera_bp.set_attribute('fov', '72') #72 degrees # Always fov on width even if width is different than height
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera_bp.set_attribute('sensor_tick', '0.10') # 10Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('motion_blur_max_distortion', '0')
        camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('shutter_speed', '1000') #1 ms shutter_speed
        camera_bp.set_attribute('lens_k', '0')
        camera_bp.set_attribute('lens_kcube', '0')
        camera_bp.set_attribute('lens_x_size', '0')
        camera_bp.set_attribute('lens_y_size', '0')
        return camera_bp
     
    def save(self):
        Camera.save(self)
        

class SS(Camera):
    sensor_id_glob = 10
    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list, folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.semantic_segmentation')

        camera_ss_bp.set_attribute('image_size_x', '1392')
        camera_ss_bp.set_attribute('image_size_y', '1024')
        camera_ss_bp.set_attribute('fov', '72') #72 degrees # Always fov on width even if width is different than height
        camera_ss_bp.set_attribute('sensor_tick', '0.10') # 10Hz camera
        return camera_ss_bp

    def save(self, color_converter=carla.ColorConverter.CityScapesPalette):
        Camera.save(self, color_converter)


class IS(Camera):
    sensor_id_glob = 10
    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list, folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.instance_segmentation')

        camera_ss_bp.set_attribute('image_size_x', '1392')
        camera_ss_bp.set_attribute('image_size_y', '1024')
        camera_ss_bp.set_attribute('fov', '72') #72 degrees # Always fov on width even if width is different than height
        camera_ss_bp.set_attribute('sensor_tick', '0.10') # 10Hz camera
        return camera_ss_bp

    def save(self, world, vehicles_list, color_converter=carla.ColorConverter.Raw):
        while not self.queue.empty():
            data = self.queue.get()

            ts = data.timestamp-Sensor.initial_ts
            if ts - self.ts_tmp > 0.11 or (ts - self.ts_tmp) < 0: #check for 10Hz camera acquisition
                print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %(self.ts_tmp, ts))
                sys.exit()
            self.ts_tmp = ts

            file_path = self.frame_output+"/%04d_%d.png" %(self.sensor_frame_id, self.sensor_id)

            data.convert(color_converter)
            y = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            y = np.reshape(y, (data.height, data.width, 4))
            bgr = y[:,:,:3]
            bg = bgr[:,:,:2]

            z = np.zeros_like(y[:,:,0])
            for player_id in vehicles_list:

                velocity = world.get_actor(player_id).get_velocity()
                v = np.array([velocity.x, velocity.y, velocity.z])
                v_norm = np.linalg.norm(v, 2)

                if abs(v_norm) >= 0.1:
                        g = (player_id & 0x00ff) >> 0
                        b = (player_id & 0xff00) >> 8
                        new_z = np.where((bg[:,:,0] == b) & (bg[:,:,1] == g), 255, 0)
                        z = np.add(z, new_z)
        
            im=Image.fromarray(z.astype(np.uint8))
            im.convert("L")
            x = threading.Thread(target=im.save, args=(file_path,))


            x.start()
            print("Export : "+file_path)

            if self.sensor_id == 0:
                with open(self.folder_output+"/full_ts_camera.txt", 'a') as file:
                    file.write(str(self.sensor_frame_id)+" "+str(data.timestamp - Sensor.initial_ts)+"\n") #bug in CARLA 0.9.10: timestamp of camera is one tick late. 1 tick = 1/fps_simu seconds
            self.sensor_frame_id += 1


def screenshot(vehicle, world, actor_list, folder_output, transform):
    sensor = world.spawn_actor(RGB.set_attributes(RGB, world.get_blueprint_library()), transform, attach_to=vehicle)
    actor_list.append(sensor)
    screenshot_queue = queue.Queue()
    sensor.listen(screenshot_queue.put)
    print('created %s' % sensor)

    while screenshot_queue.empty(): world.tick()

    file_path = folder_output+"/screenshot.png"
    screenshot_queue.get().save_to_disk(file_path)
    print("Export : "+file_path)
    actor_list[-1].destroy()
    print('destroyed %s' % actor_list[-1])
    del actor_list[-1]


            
def spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id):
        world = client.get_world()

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        
        #traffic_manager.set_hybrid_physics_mode(True)
        #traffic_manager.set_random_device_seed(args.seed)

        traffic_manager.set_synchronous_mode(True)
        synchronous_master = True

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        safe = True
        if safe:
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
                blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
                blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
                blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("Number of spawn points : ", number_of_spawn_points)

        if nbr_vehicles <= number_of_spawn_points:
                random.shuffle(spawn_points)
        elif nbr_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                logging.warning(msg, nbr_vehicles, number_of_spawn_points)
                nbr_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
                if n >= nbr_vehicles:
                        break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                        blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                # prepare the light state of the cars to spawn
                light_state = vls.NONE
                car_lights_on = False
                if car_lights_on:
                        light_state = vls.Position | vls.LowBeam | vls.LowBeam

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                        .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                        logging.error(response.error)
                else:
                        vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        walkers_list = []
        percentagePedestriansRunning = 0.0            # how many pedestrians will run
        percentagePedestriansCrossing = 0.0         # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        all_loc = []
        i = 0
        while i < nbr_walkers:
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if ((loc != None) and not(loc in all_loc)):
                        spawn_point.location = loc
                        spawn_points.append(spawn_point)
                        all_loc.append(loc)
                        i = i + 1
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                        if (random.random() > percentagePedestriansRunning):
                                # walking
                                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                        else:
                                # running
                                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                        print("Walker has no speed")
                        walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
                if results[i].error:
                        logging.error(results[i].error)
                else:
                        walkers_list.append({"id": results[i].actor_id})
                        walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
                if results[i].error:
                        logging.error(results[i].error)
                else:
                        walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
                all_walkers_id.append(walkers_list[i]["con"])
                all_walkers_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_walkers_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_walkers_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('Spawned %d vehicles and %d walkers' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

def follow(transform, world):    # Transforme carla.Location(x,y,z) from sensor to world frame
    rot = transform.rotation
    rot.pitch = -25 
    world.get_spectator().set_transform(carla.Transform(transform.transform(carla.Location(x=-15,y=0,z=5)), rot))