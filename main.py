import base64
import os
import pickle
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from sys import platform as _platform
import cv2  # opencv
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
# from IPython.display import clear_output
from PIL import Image
from flask import Flask, request
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Flatten, Activation
# keras imports
from keras.models import Sequential
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tensorflow.keras.optimizers import Adam


def get_current_path(*args):
    filepath = os.path.join(*args)
    path = os.path.join(os.path.dirname('/Users/i353584/Documents/Python/ML/Dino-Game/main.py'), filepath)
    print("Path : ", path)
    return path


# K.tensorflow_backend._get_available_gpus()
# path variables
game_url = "chrome://dino"
chrome_driver_path = get_current_path('chromedriver')
loss_file_path = get_current_path('objects', 'loss_df.csv')
actions_file_path = get_current_path('objects', 'actions_df.csv')
q_value_file_path = get_current_path('objects', 'q_values.csv')
scores_file_path = get_current_path('objects', 'scores_df.csv')
weights_file = get_current_path("model.h5")

# scripts
# create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# get image from canvas
getbase64Script = "canvasRunner = document.getElementsByClassName('runner-canvas')[0]; \
return canvasRunner.toDataURL().substring(22)"
from selenium.webdriver.chrome.options import Options


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        try:
            self._driver.get('chrome://dino')
        except:
            pass
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(
            score_array)  # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


class DinoAgent:
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game;
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class Game_sate:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()  # display the processed image on screen using openCV, implemented using python coroutine
        self._display.__next__()  # initiliaze the display coroutine

    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1]  # storing actions in a dataframe
        score = self._game.get_score()
        reward = 0.1
        is_over = False  # game over
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver)
        self._display.send(image)  # display the image on screen
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score  # log the score when game is over
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over  # return the Experience tuple


def save_obj(obj, name):
    filepath = get_current_path('objects', name + '.pkl')
    with open(filepath, 'wb+') as f:  # dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    filepath = get_current_path('objects', name + '.pkl')
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)  # processing image as required
    return image


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])
q_values_df = pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns=['qvalues'])

# game parameters
ACTIONS = 2  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 10000  # number of previous transitions to remember
BATCH = 1000  # size of minibatch
FRAME_PER_ACTION = 100000
LEARNING_RATE = 0.001
img_rows, img_cols = 80, 80
img_channels = 1  # We stack 4 frames
executor = ThreadPoolExecutor(max_workers=10)


# training variables saved as checkpoints to filesystem to resume training from the same step
def init_cache():
    """initial variable caching, done only once"""
    if not os.path.isfile(get_current_path("objects", "epsilon.pkl")):
        save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    if not os.path.isfile(get_current_path("objects", "time.pkl")):
        save_obj(t, "time")
    D = deque()
    if not os.path.isfile(get_current_path("objects", "D.pkl")):
        save_obj(D, "D")


def buildmodel():
    print("Now we build the model")

    model = Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, 1)))  # 80*80*1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(ACTIONS, activation="softmax"))

    huber = tf.keras.losses.Huber(delta=0.1)
    adam = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=huber, optimizer=adam)

    if os.path.isfile(weights_file):
        model.build((BATCH, img_rows, img_cols, 1))
        model.load_weights(weights_file)
        print("model loaded successfully")

    print("We finish building the model")
    return model


def train_on_batch():
    global loss, Q_sa, model, D, STOP_LOOP
    last_found_neg_rew = 0
    while True:

        if STOP_LOOP:
            break

        timeIt = time.time()
        # sample a minibatch to train on
        if len(D) < BATCH:
            time.sleep(1)
            continue
        minibatch = random.sample(D, int(BATCH / 3))

        last_found_neg_rew = (last_found_neg_rew + int(BATCH * 0.66)) % len(D)

        for i in range(last_found_neg_rew, len(D)):
            reward_t = D[i][2]
            minibatch.append(D[i])
            if len(minibatch) >= BATCH:
                break

        print("taking index ", last_found_neg_rew, len(D))

        inputs = np.zeros((BATCH, img_rows, img_cols, 1))  # 32, 20, 40, 4
        targets = np.zeros((BATCH, 2))  # 32, 2

        # Now we do the experience replay
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]  # 4D stack of images
            action_t = minibatch[i][1]  # This is action index
            reward_t = minibatch[i][2]  # reward at state_t due to action_t
            state_t1 = minibatch[i][3]  # next state
            is_over = minibatch[i][4]  # wheather the agent died or survided due the action

            inputs[i:i + 1] = state_t

            alt_action = (action_t + 1) % 2

            if is_over:
                targets[i, action_t] = 0
                targets[i, alt_action] = 1  # if terminated, only equals reward
            else:
                # targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # q values for predict next step
                targets[i, action_t] = reward_t + (1 - reward_t) * np.max(Q_sa)
                targets[i, alt_action] = 1 - targets[i, action_t]

            # if is_over:
            #     targets[i] = alt_action
            # else:
            #     targets[i] = action_t
        print("data preprocessed starting training")
        loss1 = model.train_on_batch(inputs, targets)
        loss = (loss1 + loss) / 2
        loss_df.loc[len(loss_df)] = loss
        q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        model.save_weights(weights_file, overwrite=True)
        print("time taken : ", time.time() - timeIt, "loss : ", loss, "delta : ", loss1)
    return loss


def trainNetwork(game_state, observe=False):
    global loss, Q_sa, model, D, STOP_LOOP
    last_time = time.time()
    # store the previous observations in replay memory

    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => do nothing,
    # 1=> jump

    image, reward, is_over = game_state.get_state(do_nothing)  # get next step after performing the action

    # s_t = np.stack((image, image, image, image), axis=2)  # stack 4 images to create placeholder input

    s_t = image.reshape(1, img_rows, img_cols, 1)  # 1*80*80*1
    # s_t = image

    initial_state = s_t

    if observe:
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = FINAL_EPSILON
    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")

    t = load_obj("time")  # resume from the previous time step stored in file system
    print("loaded time ", t, len(D))
    last_over_time = time.time()
    highest_time = 16

    while (True):  # endless running

        if STOP_LOOP:
            break

        action_index = 0
        r_t = 0  # reward at 4
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # randomly explore an action
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:  # predict the output
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)  # chosing index with maximum q value
            action_index = max_Q
            a_t[action_index] = 1  # o=> do nothing, 1=> jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and reward
        x_t1, r_t, is_over = game_state.get_state(a_t)
        fps = int(1 / (time.time() - last_time))
        print('fps: {0}'.format(fps), r_t)  # helpful for measuring frame rate
        last_time = time.time()
        x_t1 = x_t1 if int(time.time() % 2) == 0 else cv2.bitwise_not(x_t1)
        x_t1 = keras.preprocessing.utils.normalize(x_t1)
        x_t1 = x_t1.reshape(1, img_rows, img_cols, 1)  # 1x80x80x1
        # s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # append the new image to input stack and remove the first one
        s_t1 = x_t1

        current_time = time.time() - last_over_time
        if current_time > highest_time:
            highest_time = current_time
        if not is_over:
            r_t = r_t + abs(r_t * (current_time / highest_time))
        else:
            last_over_time = time.time()
            for index in range(len(D) - (fps >> 1), len(D)):
                item = list(D[index])
                item[2] = -1
                D[index] = tuple(item)

        # store the transition in D
        if current_time - 3 > 0:
            D.append((s_t, action_index, r_t, s_t1, is_over))
            t = t + 1
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            pass
        s_t = initial_state if is_over else s_t1  # reset game to initial frame if terminate
        # t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause()  # pause game while saving to filesystem
            # model.save_weights(weights_file, overwrite=True)
            save_obj(D, "D")  # saving episodes
            save_obj(t, "time")  # caching time steps
            save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv(get_current_path("objects", "loss_df.csv"), index=False)
            scores_df.to_csv(get_current_path("objects", "scores_df.csv"), index=False)
            actions_df.to_csv(get_current_path("objects", "actions_df.csv"), index=False)
            q_values_df.to_csv(q_value_file_path, index=False)
            game_state._game.resume()
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # time.sleep(0.001)

        if t % 10 == 0:
            print("current time ", current_time, "highest time ", highest_time)
            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
                  "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


model: Sequential = None
loss = 0
Q_sa = 0
D: deque = None
STOP_LOOP = False

app = Flask(__name__)


def run_server():
    global app
    app.run(port=8000)


@app.route('/stop')
def end_loop():
    global STOP_LOOP
    STOP_LOOP = True
    print("Stopping Loops")
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Success"


# main function
def playGame(observe=False):
    global model, D
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino, game)
    model = buildmodel()
    D = load_obj("D")  # load from file system
    try:
        threads = []
        trainer = threading.Thread(target=train_on_batch)
        threads.append(trainer)
        # collector = threading.Thread(target=trainNetwork, args=(game_state, observe))
        # threads.append(collector)
        # trainNetwork(model, game_state, loss, D, observe)
        server = threading.Thread(target=run_server)
        threads.append(server)
        for thread in threads:
            thread.start()
        trainNetwork(game_state, observe)
        for thread in threads:
            thread.join()
    except StopIteration:
        game.end()


def main():
    init_cache()
    playGame(observe=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
