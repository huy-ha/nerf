from math import radians
import os
import json
import bpy
import numpy as np
import random
import time

NUM_SCENES = 500
VIEWS = 128
RESOLUTION = 512
RESULTS_PATH = 'data/metamonkeyfall_dec8/'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
CIRCLE_FIXED_START = (0, 0, 0)
CIRCLE_FIXED_END = (.7, 0, 0)
MAX_TIME = 50
DT = 1

random.seed(int(time.time()))
PI = np.pi
scene = bpy.data.scenes[0]
for scene in bpy.data.scenes:
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 16
    scene.render.image_settings.compression = 95


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 1)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def setup():
    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True
    bpy.context.scene.render.image_settings.file_format = str(FORMAT)
    bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

    if 'Custom Outputs' not in tree.nodes:
        # Create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        render_layers.label = 'Custom Outputs'
        render_layers.name = 'Custom Outputs'

        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.name = 'Depth Output'
        if FORMAT == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'],
                      depth_file_output.inputs[0])
        else:
            # Remap as other types can not represent the full range of depth.
            map = tree.nodes.new(type="CompositorNodeMapRange")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.inputs['From Min'].default_value = 0
            map.inputs['From Max'].default_value = 8
            map.inputs['To Min'].default_value = 1
            map.inputs['To Max'].default_value = 0
            links.new(render_layers.outputs['Depth'], map.inputs[0])

            links.new(map.outputs[0], depth_file_output.inputs[0])

        normal_file_output = tree.nodes.new(
            type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        normal_file_output.name = 'Normal Output'
        links.new(render_layers.outputs['Normal'],
                  normal_file_output.inputs[0])

    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.render.film_transparent = True

    # Create collection for objects not to render with background

    objs = [ob for ob in bpy.context.scene.objects if ob.type in (
        'EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100

    cam = scene.objects['Camera']
    cam.location = (0, 5.0, 0.5)
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    stepsize = 360.0 / VIEWS
    vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
    rotation_mode = 'XYZ'

    for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
        output_node.base_path = ''

    return b_empty, cam, stepsize, vertical_diff


if __name__ == '__main__':
    objects = bpy.context.scene.objects
    monkey = objects['Suzanne']
    fp = bpy.path.abspath(f"//{RESULTS_PATH}")
    if not os.path.exists(fp):
        os.makedirs(fp)
    b_empty, cam, stepsize, vertical_diff = setup()
    tree = bpy.context.scene.node_tree
    scene_ids = list(range(NUM_SCENES))
    random.shuffle(scene_ids)
    for i_scene in scene_ids:
        print('#'*10 + f' scene {i_scene}' + '#'*10)
        scene_output_dir = fp + f'{i_scene:06d}'
        if os.path.exists(scene_output_dir):
            continue
        # Data to store in JSON file
        out_data = {
            'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
        }
        bpy.context.scene.frame_set(0)
        # randomize scene
        monkey.location.x = random.uniform(-1.0, 1.0)
        monkey.location.y = random.uniform(-1.0, 1.0)
        monkey.location.z = random.uniform(1.0, 2.5)
        monkey.rotation_euler = (
            random.uniform(-PI, PI),
            random.uniform(-PI, PI),
            random.uniform(-PI, PI)
        )

        out_data['frames'] = []

        b_empty.rotation_euler = CIRCLE_FIXED_START
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

        for i in range(0, VIEWS):
            print("Rotation {}, {}".format(
                (stepsize * i), radians(stepsize * i)))
            b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + \
                (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
            b_empty.rotation_euler[2] += radians(2*stepsize)
            for t in range(0, MAX_TIME, DT):
                bpy.context.scene.frame_set(t)
                scene.render.filepath = scene_output_dir + f'/r_{i}_t{t}'

                tree.nodes['Depth Output'].file_slots[0].path = scene.render.filepath + "_depth_"
                tree.nodes['Normal Output'].file_slots[0].path = scene.render.filepath + "_normal_"

                bpy.ops.render.render(write_still=True)  # render still

                frame_data = {
                    'file_path': scene.render.filepath,
                    'rotation': radians(stepsize),
                    'transform_matrix': listify_matrix(cam.matrix_world),
                    'timestep': t
                }
                out_data['frames'].append(frame_data)

                with open(scene_output_dir + '/' + 'transforms.json', 'w') as out_file:
                    json.dump(out_data, out_file, indent=4)
