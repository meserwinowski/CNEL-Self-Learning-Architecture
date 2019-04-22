"""
keyboard_agent.py - Python script supplied by OpenAI for the OpenAI Retro Gym.
Generates an emulation instance of whatever game is specified. This game can
then be played with the keyboard. ESC will close the game and save all frames
to a bk2 file.

"""

import argparse
import random
import pyglet
import sys
import ctypes
import os
import cv2
import re
import errno

from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *

import retro

# TODO:
# indicate to user when episode is over (hard without save/restore lua state)
# record bk2 directly
# resume from bk2

SAVE_PERIOD = 60  # frames
numbers = re.compile(r'(\d+)')


def numericalSort(x):
    parts = numbers.split(x)
    parts[1::2] = map(int, parts[1::2])
    return parts


def save_image(img, path, name):
    if (len(img.shape) != 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (os.path.isdir(path)):
        cv2.imwrite(os.path.join(path, name), img)
    else:
        os.makedirs(path)
        cv2.imwrite(os.path.join(path, name), img)


def output_video(output, path):
    for image in sorted(os.listdir(path), key=numericalSort):
        image_path = os.path.join(path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        output.write(frame)  # Write out frame to video
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break


def delete_images(path):
    for image in os.listdir(path):
        os.remove(os.path.join(path, image))


def delete_directory(path):
    for image in os.listdir(path):
        os.rmdir(os.path.join(path, image))
    try:
        os.rmdir(path)  # Remove temporary image directory
    except OSError as e:
        print("Directory specified for removal probably not empty")
        print(e)


class buttoncodes:
    A = 15
    B = 16
    X = 17
    Y = 18
    START = 8
    SELECT = 9
    XBOX = 14
    LEFT_BUMPER = 12
    RIGHT_BUMPER = 13
    RIGHT_STICK = 11
    LEFT_STICK = 10
    D_LEFT = 6
    D_RIGHT = 7
    D_UP = 4
    D_DOWN = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('game',
                        help='the name or path for the game to run')
    parser.add_argument('state', nargs='?',
                        help='the initial state file to load, minus the extension')
    parser.add_argument('--scenario', '-s', default='scenario',
                        help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-r', action='store_true',
                        help='record bk2 movies')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='increase verbosity (can be specified multipl times)')
    parser.add_argument('--quiet', '-q', action='count', default=0,
                        help='decrease verbosity (can be specified multipl times)')
    parser.add_argument('--players', '-p', type=int, default=1,
                        help='number of players/agents (default: 1)')
    args = parser.parse_args()

    if args.game is None:
        print('Please specify a game with --game <game>')
        print('Available games:')
        for game in sorted(retro.data.list_games()):
            print(game)
        sys.exit(1)

    if args.state is None:
        print('Please specify a state with --state <state>')
        print('Available states:')
        for state in sorted(retro.data.list_states(args.game)):
            print(state)
        sys.exit(1)

    mfile = '%s-%s-%04d' % (args.game, args.state, 0)
    dir_path = os.getcwd() + mfile + '/'
    output = mfile + '.mp4'  # Video file extension
    try:
        os.makedirs(dir_path)  # Create temporary image directory
    except OSError as e:
        if e.errno != errno.EEXIST:  # Handle directory already existing
            raise

    env = retro.make(args.game,
                     args.state or retro.State.DEFAULT,
                     scenario=args.scenario,
                     record=args.record,
                     players=args.players)

    obs = env.reset()
    screen_height, screen_width = obs.shape[:2]

    random.seed(0)

    key_handler = pyglet.window.key.KeyStateHandler()
    win_width = 600
    win_height = win_width * screen_height // screen_width
    win = pyglet.window.Window(width=win_width, height=win_height, vsync=False)

    pixel_scale = 1
    if hasattr(win.context, '_nscontext'):
        pixel_scale = win.context._nscontext.view().backingScaleFactor()

    win.width = win.width // pixel_scale
    win.height = win.height // pixel_scale

    joysticks = pyglet.input.get_joysticks()
    if len(joysticks) > 0:
        joystick = joysticks[0]
        joystick.open()
    else:
        joystick = None

    win.push_handlers(key_handler)

    key_previous_states = {}
    button_previous_states = {}

    steps = 0
    recorded_actions = []
    recorded_states = []

    pyglet.app.platform_event_loop.start()

    fps_display = pyglet.clock.ClockDisplay()
    clock.set_fps_limit(60)

    glEnable(GL_TEXTURE_2D)
    texture_id = GLuint(0)
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    while not win.has_exit:
        win.dispatch_events()

        win.clear()

        keys_clicked = set()
        keys_pressed = set()
        for key_code, pressed in key_handler.items():
            if pressed:
                keys_pressed.add(key_code)

            if not key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
            key_previous_states[key_code] = pressed

        buttons_clicked = set()
        buttons_pressed = set()
        if joystick is not None:
            for button_code, pressed in enumerate(joystick.buttons):
                if pressed:
                    buttons_pressed.add(button_code)

                if not button_previous_states.get(button_code, False) and pressed:
                    buttons_clicked.add(button_code)
                button_previous_states[button_code] = pressed

        if keycodes.R in keys_clicked or buttoncodes.LEFT_BUMPER in buttons_clicked:
            if len(recorded_states) > 1:
                recorded_states.pop()
                steps, save_state = recorded_states.pop()
                recorded_states = recorded_states[:steps]
                recorded_actions = recorded_actions[:steps]
                env.em.set_state(save_state)

        if keycodes.ESCAPE in keys_pressed or buttoncodes.XBOX in buttons_clicked:
            # record all the actions so far to a bk2 and exit
            i = 0
            while True:
                movie_filename = 'human/%s/%s/%s-%s-%04d.bk2' % (args.game,
                                                                 args.scenario,
                                                                 args.game,
                                                                 args.state, i)
                if not os.path.exists(movie_filename):
                    break
                i += 1
            os.makedirs(os.path.dirname(movie_filename), exist_ok=True)
            env.record_movie(movie_filename)
            env.reset()
            for step, act in enumerate(recorded_actions):
                if step % 1000 == 0:
                    print('saving %d/%d' % (step, len(recorded_actions)))
                env.step(act)
            env.stop_record()
            print("creating mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#            print("TEST ", win_width, " ", win_height)
            out = cv2.VideoWriter(output, fourcc, 60.0, (win_width, win_height))
            output_video(out, dir_path + "original/")
            delete_images(dir_path + "original/")
            out.release()
            delete_directory(dir_path)
            print('complete')
            sys.exit(1)

        inputs = {
            'A': keycodes.Z in keys_pressed or buttoncodes.A in buttons_pressed,
            'B': keycodes.X in keys_pressed or buttoncodes.B in buttons_pressed,
            'C': keycodes.C in keys_pressed,
            'X': keycodes.A in keys_pressed or buttoncodes.X in buttons_pressed,
            'Y': keycodes.S in keys_pressed or buttoncodes.Y in buttons_pressed,
            'Z': keycodes.D in keys_pressed,

            'UP': keycodes.UP in keys_pressed or buttoncodes.D_UP in buttons_pressed,
            'DOWN': keycodes.DOWN in keys_pressed or buttoncodes.D_DOWN in buttons_pressed,
            'LEFT': keycodes.LEFT in keys_pressed or buttoncodes.D_LEFT in buttons_pressed,
            'L': keycodes.LEFT in keys_pressed or buttoncodes.D_LEFT in buttons_pressed,
            'RIGHT': keycodes.RIGHT in keys_pressed or buttoncodes.D_RIGHT in buttons_pressed,
            'R': keycodes.RIGHT in keys_pressed or buttoncodes.D_RIGHT in buttons_pressed,

            'MODE': keycodes.TAB in keys_pressed or buttoncodes.SELECT in buttons_pressed,
            'START': keycodes.ENTER in keys_pressed or buttoncodes.START in buttons_pressed,
            'SELECT': keycodes.SELECT in keys_pressed or buttoncodes.SELECT in buttons_pressed,
        }
        try:
            action = [inputs[b] for b in env.buttons]
        except KeyError as ke:
            print(ke)

        if steps % SAVE_PERIOD == 0:
            recorded_states.append((steps, env.em.get_state()))
        obs, rew, done, info = env.step(action)
        if (steps == 0):
            print("win_width: ", win_width)
            print("win_width: ", win_height)
            height, width, channels = obs.shape
            win_height = height
            win_width = width
            print("H: ", height, "W: ", width, "C: ", channels)
        image_name = str(steps) + '.png'
        save_image(obs, dir_path + "original/", image_name)
        recorded_actions.append(action)
        steps += 1

        glBindTexture(GL_TEXTURE_2D, texture_id)
        video_buffer = ctypes.cast(obs.tobytes(), ctypes.POINTER(ctypes.c_short))
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, obs.shape[1], obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        h = win.height
        w = win.width

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

        fps_display.draw()

        win.flip()

        # process joystick events
        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)

        clock.tick()

    pyglet.app.platform_event_loop.stop()


main()
