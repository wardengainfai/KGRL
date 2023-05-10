# This is a gym wrapper for the OpenAI gym environment.
# It is used to create a custom environment for the agent.
from collections import Counter
from itertools import chain
import math
import random
import time

import numpy as np

import gym
import scipy
import gensim

from sklearn import preprocessing
from sklearn.cluster import KMeans

import nltk
from nltk.cluster import KMeansClusterer

from gym_minigrid.wrappers import FlatObsWrapper


class Callback:
    def __init__(self) -> None:
        self.exploration_time_repetitions = []
        self.solve_word2vec_time_repetitions = []
        self.solve_kmeans_time_repetitions = []

        self.longlife_exploration_std_repetitions = []
        self.longlife_exploration_mean_repetitions = []

    def _end_exploration(self, time_cost, agent_e, tracks):
        print("exploration_time:", time_cost)
        self.exploration_time_repetitions.append(time_cost)

        self.longlife_exploration_std_repetitions.append(
            np.std(agent_e.states_long_life[np.array(agent_e.states_long_life.values) > 0])
        )
        print("longlife_exploration_std:", self.longlife_exploration_std_repetitions)

        self.longlife_exploration_mean_repetitions.append(
            np.mean(agent_e.states_long_life[np.array(agent_e.states_long_life.values) > 0])
        )
        print("longlife_exploration_mean:", self.longlife_exploration_mean_repetitions)

        print(
            "longlife_exploration_sum:",
            np.sum(agent_e.states_long_life[np.array(agent_e.states_long_life.values) > 0]),
        )

        print("len of self.sentences_period:", len(tracks))
        flatten_list = list(chain.from_iterable(tracks))
        counter_dict = Counter(flatten_list)
        print("min counter value:", min(counter_dict.values()))
        under5 = [k for k, v in counter_dict.items() if v < 5]
        print("under5:", under5)
        print("under5 length:", len(under5))
        print("len(flatten_list):", len(flatten_list))
        print("unique len(flatten_list)", len(set(flatten_list)))


class BaselineWrapper(gym.Wrapper):
    def __init__(self, env, callback: Callback):
        super().__init__(env)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.explore_config = {
            "e_mode": "sarsa",
            "e_start": "random",
            "e_eps": 5000,
            "max_move_count": 150,
            "ds_factor": 0.5,  # downsamping factor
            "lr": 0.1,
            "gamma": 0.999,
            "epsilon_e": 0.01,
        }
        self.w2v_config = {
            "latent_size": 128,
            "window_size": 75,
            "sip_gram": True,
            "workers": 2056,
        }

        self.shapping_config = {
            "gamma": 0.999,
            "omega": 100,
        }

        self.num_clusters = 9  # [9, 16, 25, 36]
        self.k_means_pkg = "sklearn"  # 'sklearn' or 'nltk'
        self.repetitions = 20

        self.callback = callback

        self.previous_obs = self.reset()
        self._learn_amdp()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        previous_abstract_obs = self.amdp.get_abstract_state(self.previous_obs)
        abstract_obs = self.amdp.get_abstract_state(obs)
        potential_previous_abstract_obs = self.amdp.get_value(previous_abstract_obs)
        potential_abstract_obs = self.amdp.get_value(abstract_obs)

        shaping = (
            self.shapping_config["gamma"] * potential_abstract_obs - potential_previous_abstract_obs
        ) * self.shapping_config["omega"]

        return obs, reward + shaping, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def _learn_amdp(self):
        print("-----Begin learn amdp-----")
        self.reward_episodes = []
        self.move_count_episodes = []

        self.tracks_collected = []
        self.tracks = []  # for gensim w2v
        self.tracks_complete = []

        self._explore()
        gensim_operater = self._w2v_and_kmeans()
        self.amdp = AMDP_General(self.tracks_complete, env=self.env, gensim_opt=gensim_operater)
        self.amdp.solve_amdp()

        print("-----Finish learn amdp-----")

    def _explore(self):
        print("-----Begin Exploration-----")
        start_exploration = time.time()
        self.agent_e = ExploreStateBrain(env=self.env, explore_config=self.explore_config)
        env = self.env
        env = FlatObsWrapper(env)
        env = ObsPreprocessWrapper(env)
        # valid_states_ = tuple(env.valid_states)
        for ep in range(self.explore_config["e_eps"]):
            # if (ep + 1) % 100 == 0:
            #     print(f"episode_100: {ep} | avg_move_count: {int(np.mean(self.move_count_episodes[-100:]))} | "
            #           f"avd_reward: {int(np.mean(self.reward_episodes[-100:]))} | "
            #           f"env.state: {env.state} | "
            #           f"env.flagcollected: {env.flags_collected} | "
            #           f"agent.epsilon: {agent_e.epsilon} | "
            #           f"agent.lr: {agent_e.lr}")
            move_count = 0

            self.agent_e.epsilon = self.explore_config["epsilon_e"]

            state = env.reset()
            # state = str(state)

            self.agent_e.reset_episodic_staff()

            if self.explore_config["e_mode"] == "sarsa":
                track = [state]
                a = self.agent_e.policy_explore_rl(state)
                while move_count < self.explore_config["max_move_count"]:
                    # self.agent_e.state_visit_episodic[state] += 1
                    state_visit = self.agent_e.state_visit_long_life.get(state, 0)
                    self.agent_e.state_visit_long_life[state] = state_visit + 1
                    state_prime, _, _, _ = env.step(a)
                    # state_prime = str(state_prime)
                    move_count += 1
                    track.append(state_prime)

                    r = self._explore_reward_by_state_visit(state_prime)
                    a_prime = self.agent_e.policy_explore_rl(state_prime)
                    self.agent_e.learn_explore_sarsa(state, a, state_prime, a_prime, r)
                    state = state_prime
                    a = a_prime

            elif self.explore_config["e_mode"] == "softmax":
                track = [state]
                a = self.agent_e.policy_explore_softmax(state)
                while move_count < self.explore_config["max_move_count"]:
                    self.agent_e.state_actions_long_life[state][a] -= 1
                    state_prime, _, _, _ = env.step(a)
                    # state_prime = str(state_prime)
                    move_count += 1
                    track.append(state_prime)
                    a_prime = self.agent_e.policy_explore_softmax(state_prime)
                    state = state_prime
                    a = a_prime
            else:
                raise Exception("Invalid self.explore_config['e_mode']")

            self.move_count_episodes.append(move_count)

            # downsampling track by "ds_factor"
            if self.explore_config["ds_factor"] == 1:
                self.tracks.append(track)
                self.tracks_complete.append(track)
            else:
                for _ in range(2):
                    down_sampled = [
                        track[index]
                        for index in sorted(
                            random.sample(
                                range(len(track)),
                                math.floor(len(track) * self.explore_config["ds_factor"]),
                            )
                        )
                    ]
                    self.tracks.append(down_sampled)
                self.tracks_complete.append(track)

        end_exploration = time.time()
        # callbacks
        self.callback._end_exploration()(
            end_exploration - start_exploration, self.agent_e, self.tracks
        )

        self.tracks_collected.extend(self.tracks)
        self.tracks = []

        print("-----Finish Exploration-----")

    def _explore_reward_by_state_visit(self, obs):
        r1 = -self.agent_e.state_visit_long_life.get(obs, 0)
        # r2 = -self.agent_e.state_visit_episodic[str(obs)]
        r = r1 * 10
        return r

    def _w2v_and_kmeans(self):
        print("-----Begin w2v and k-means-----")
        random.shuffle(self.tracks_collected)
        gensim_opt = GensimOperator_General(self.env)
        solve_wor2vec_time, solve_kmeans_time = gensim_opt.get_cluster_labels(
            tracks=self.tracks_collected,
            size=self.w2v_config["latent_size"],
            window=self.w2v_config["win_size"],
            clusters=self.num_clusters,
            skip_gram=self.w2v_config["sg"],
            workers=self.w2v_config["workers"],
            package=self.k_means_pkg,
        )
        self.solve_word2vec_time_repetitions.append(solve_wor2vec_time)
        self.solve_kmeans_time_repetitions.append(solve_kmeans_time)
        print("-----Finish w2v and k-means-----")
        return gensim_opt


class ExploreStateBrain:
    def __init__(self, env, explore_config: dict):
        self.env = env
        # self.state_size = env.size
        # self.action_size = env.num_of_actions
        # self.explore_config = explore_config
        self.epsilon = explore_config["epsilon_e"]
        self.lr = explore_config["lr"]
        self.gamma = explore_config["gamma"]
        self.e_mode = explore_config["e_mode"]

        if self.e_mode == "sarsa":  # only support sarsa so far
            # self.q_table2 = np.zeros((self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size))
            # self.q_table2 = np.random.rand(self.state_size[0], self.state_size[1], 2, 2, 2, self.action_size)
            self.q_table = {}
            self.q_init_non_zero = 1
            self.state_visit_long_life = {}
            self.state_visit_episodic = {}
        elif self.e_mode == "softmax":
            self.state_actions_long_life = {}
            self.state_actions_episodic = {}
        else:
            raise Exception("invalid e_mode")

    def reset_episodic_staff(self):
        # self.state_actions_episodic = np.zeros((self.state_size[0], self.state_size[1], self.action_size))
        if self.e_mode == "sarsa":
            self.states_visit_episodic = {}
        elif self.e_mode == "softmax":
            self.state_actions_episodic = {}
        else:
            raise Exception("invalid e_mode")

    def policy_explore_rl(self, state: str):
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            return self.env.action_space.sample()

        retrieved = self.q_table.get(state, None)
        if not retrieved:
            self.q_table[state] = np.zeros(self.env.action_space.n, dtype=np.float32)
            return self.env.action_space.sample()
        else:
            return np.argmax(retrieved)

    def policyNoRand_explore_rl(self, state, actions):
        retrieved = self.q_table.get(state, None)
        if not retrieved:
            self.q_table[state] = np.zeros(self.env.action_space.n, dtype=np.float32)
            return self.env.action_space.sample()
        else:
            return np.argmax(retrieved)

    def policy_explore_softmax(self, state: str):
        ##epsilon greedy policy
        ran = np.random.randint(100)
        if ran < self.epsilon * 100:
            return self.env.action_space.sample()

        retrieved = self.q_table.get(state, None)
        if not retrieved:
            self.q_table[state] = np.zeros(self.env.action_space.n, dtype=np.float32)
            return self.env.action_space.sample()
        else:
            probs = scipy.special.softmax(self.state_actions_long_life[state])
            return np.random.choice(np.arange(self.env.action_space.n), p=probs)

    def learn_explore_sarsa(self, state1: str, action1: int, state2: str, action2: int, reward):
        max_value = self.q_table[state2][action2]
        delta = reward + (self.gamma * max_value) - self.q_table[state1][action1]
        self.q_table[state1][action1] += self.lr * delta


class GensimOperator_General:
    def __init__(self, env):
        # self.sentences = sentences   # 2-D list, strings inside
        self.env = env
        self.wv = None
        self.cluster_labels = None
        self.num_of_tracks_last_time = 0
        self.model = None
        print("GensimOperator_General initialized!!")

    def get_cluster_labels(
        self,
        tracks,
        vector_size,
        window,
        clusters,
        skip_gram=1,
        min_count=5,
        workers=30,
        negative=5,
        package="sklearn",
    ):
        self.sentences = tracks
        self.num_clusters = clusters
        print("start gensim Word2Vec model training...")
        if len(tracks) == self.num_of_tracks_last_time:
            model = self.model
        else:
            start = time.time()
            model = gensim.models.Word2Vec(
                sentences=tracks,
                min_count=min_count,
                vector_size=vector_size,
                workers=workers,
                window=window,
                sg=skip_gram,
                negative=negative,
            )
            end = time.time()
            w2v_time = end - start
            print(f"internal w2v training time: {w2v_time}")
        self.wv = model.wv
        self.embeddings = []
        self.words = []
        self.weights = []
        flatten_list = list(chain.from_iterable(tracks))
        self.counter_dict = Counter(flatten_list)
        # print("self.counter_dict:", self.counter_dict)
        print(len(self.counter_dict))

        for i, word in enumerate(model.wv.index_to_word):
            self.words.append(word)
            self.embeddings.append(self.wv[word])
            self.weights.append(self.counter_dict[word])
        self.weights = None

        print("start check unvisited nodes...")
        self.check_unvisited_states()

        print("start clustering...")
        start = time.time()
        if package == "sklearn":
            norm_embeddings = preprocessing.normalize(self.embeddings)
            kmeans_labels = KMeans(n_clusters=clusters, init="k-means++", tol=0.00001).fit_predict(
                np.array(norm_embeddings), sample_weight=self.weights
            )
            self.cluster_labels = kmeans_labels
            print("gensim_opt.cluster_labels:", self.cluster_labels[:10])

        if package == "nltk":
            embeddings = preprocessing.normalize(self.embeddings)
            kclusterer = KMeansClusterer(
                clusters, distance=nltk.cluster.util.cosine_distance, repeats=25
            )
            embeddings = [np.array(f) for f in embeddings]
            assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
            self.cluster_labels = assigned_clusters

        end = time.time()
        kmeans_time = end - start
        print(f"internal k-means time: {kmeans_time}")
        self.model = model
        self.num_of_tracks_last_time = len(tracks)

        self.dict_gstates_astates = dict(zip(self.words, self.cluster_labels.tolist()))
        return w2v_time, kmeans_time

    def check_unvisited_states(self):  # deprecated to use
        valid_states = self.env.valid_states  # contain strings
        visited_states = set(self.words)  # contain strings
        print(
            "len(valid_states), len(visited_states):",
            len(valid_states),
            len(visited_states),
        )
        for i in valid_states:
            if str(i) not in visited_states:
                print("not visited: ", i)

    def get_cluster_labels_online(  # yet to be adapted
        self,
        sentences,
        size,
        window,
        clusters,
        skip_gram=1,
        min_count=5,
        workers=32,
        negative=5,
        package="sklearn",
    ):
        self.sentences = sentences
        self.num_clusters = clusters
        print("start gensim Word2Vec model training...")

        start = time.time()
        if not self.model:
            self.model = gensim.models.Word2Vec(
                sentences=sentences,
                min_count=min_count,
                size=size,
                workers=workers,
                window=window,
                sg=skip_gram,
                negative=negative,
            )
            self.wv = self.model.wv
            self.words = []
            self.embeddings = []
            for i, word in enumerate(self.wv.vocab):
                self.words.append(word)
                self.embeddings.append(self.wv[word])
            self.weights = None
        else:
            # self.model.build_vocab(sentences, update=False)
            # self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
            self.model = self.model.train(
                sentences, total_examples=len(sentences), epochs=self.model.iter
            )
            # self.model = gensim.src.Word2Vec(sentences=sentences, min_count=min_count, size=size, workers=workers,
            #                                window=window, sg=skip_gram, negative=negative)
            self.wv = self.model.wv
            self.words = []
            self.embeddings = []
            self.weights = []
            flatten_list = list(chain.from_iterable(sentences))
            set_flatten_list = set(flatten_list)
            self.counter_dict = Counter(flatten_list)
            # print("self.counter_dict:", self.counter_dict)
            print(len(self.counter_dict))
            for i, word in enumerate(self.wv.vocab):
                self.words.append(word)
                self.embeddings.append(self.wv[word])
                # self.weights.append(self.counter_dict[word] ** 2)
            self.weights = None

        end = time.time()
        w2v_time = end - start
        print(f"internal w2v training time: {w2v_time}")

        print("start check unvisited nodes...")
        self.check_unvisited_states()

        print("start clustering...")
        start = time.time()
        if package == "sklearn":
            norm_embeddings = preprocessing.normalize(self.embeddings)
            kmeans_labels = KMeans(n_clusters=clusters, init="k-means++").fit_predict(
                np.array(norm_embeddings), sample_weight=self.weights
            )
            self.cluster_labels = kmeans_labels
            print("gensim_opt.cluster_labels:", self.cluster_labels[:10])

        if package == "nltk":
            embeddings = preprocessing.normalize(self.embeddings)
            kclusterer = KMeansClusterer(
                clusters, distance=nltk.cluster.util.cosine_distance, repeats=25
            )
            embeddings = [np.array(f) for f in embeddings]
            assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
            self.cluster_labels = assigned_clusters

        end = time.time()
        kmeans_time = end - start
        print(f"internal k-means time: {kmeans_time}")

        self.dict_gstates_astates = dict(zip(self.words, self.cluster_labels.tolist()))
        return w2v_time, kmeans_time

    def reduce_dimensions_and_visualization(self, wv):  # deprecated to use
        import matplotlib.pyplot as plt
        import random
        from sklearn.manifold import TSNE

        num_dimensions = 2  # final num dimensions (2D, 3D, etc)

        vectors = []  # positions in vector space
        labels = []  # keep track of words to label our data again later
        for word in wv.vocab:
            vectors.append(wv[word])
            labels.append(word)

        # extract the words & their vectors, as numpy arrays
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)  # fixed-width numpy strings

        # reduce using t-SNE
        tsne = TSNE(n_components=num_dimensions, random_state=0)
        vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        y_vals = [v[1] for v in vectors]

        random.seed(0)
        # plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(x_vals, y_vals)
        indices = list(range(len(labels)))
        selected_indices = random.sample(indices, 30)
        for i in selected_indices:
            ax.annotate(labels[i], (x_vals[i], y_vals[i]), fontsize=10, fontweight="normal")
        fig.show()


class AMDP_General:
    def __init__(
        self, tracks_complete, env=None, gensim_opt=None
    ):  # tiling_mode is same with tiling_size
        self.tracks_complete = tracks_complete
        assert (env != None) and (gensim_opt != None), "env and gensim_opt need to be assigned"
        self.env = env
        # self.manuel_layout = env.room_layout     # np array
        self.goal = env.goal
        # self.flags = env.flags

        self.gensim_opt: GensimOperator_General = gensim_opt
        print("self.gensim_opt.sentences[:5]:", self.gensim_opt.sentences[:5])

        self.list_of_abstract_states = np.arange(self.gensim_opt.num_clusters).tolist()
        self.dict_gstates_astates = self.gensim_opt.dict_gstates_astates
        # self.dict_gstates_astates = dict(zip(self.gensim_opt.words, self.gensim_opt.cluster_labels.tolist()))
        print(
            "len(gensim_opt.words), len(gensim_opt.cluster_labels):",
            len(self.gensim_opt.words),
            len(self.gensim_opt.cluster_labels.tolist()),
        )

        print("start setting amdp transition and reward...")
        self.set_transition_and_rewards()
        # self.set_transition_and_rewards_stochastic()

    def get_abstract_state(self, state):
        # if not isinstance(state, str):
        #     state = str(state)
        return self.dict_gstates_astates[state]

    def set_transition_and_rewards(self):
        self.list_of_abstract_states.append("bin")
        num_abstract_states = len(self.list_of_abstract_states)
        # +1 for the absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        for track in self.tracks_complete:
            for i in range(len(track)):
                if i < (len(track) - 1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(track[i])
                    cluster_label2 = self.get_abstract_state(track[i + 1])
                    if not cluster_label1 == cluster_label2:
                        transition[cluster_label2, cluster_label1, cluster_label2] = 1
                        # transition[cluster_label2, cluster_label1, cluster_label1] = 0.2
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                        rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                        # when highest value is 0
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(track[i])
                state = eval(track[i])
                if state == self.goal:
                    transition[-1, cluster_label1, -1] = 1
                    # rewards[-1, cluster_label1, -1] = 3000
                    # above to comment when highest value is 0
        transition[-1, -1, -1] = 1
        # when highest value is 0, otherwise max(list_of_values) report error of empty sequence

        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards

    def set_transition_and_rewards_stochastic(self):
        start = time.time()
        self.list_of_abstract_states.append("bin")
        num_abstract_states = len(self.list_of_abstract_states)  # +1 for absorbing abstract state
        transition = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        # transition_mask = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        rewards = np.zeros(shape=(num_abstract_states, num_abstract_states, num_abstract_states))
        alpha = 0.05
        beta = 0.01
        s_num = 0
        transition_mask = np.zeros(
            shape=(num_abstract_states, num_abstract_states, num_abstract_states)
        )
        for sentence in self.tracks_complete:
            if s_num % 1 == 0:
                transition_mask = np.zeros(
                    shape=(
                        num_abstract_states,
                        num_abstract_states,
                        num_abstract_states,
                    )
                )
            s_num += 1
            for i in range(len(sentence)):
                if i < (len(sentence) - 1):
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # index2 = self.list_of_ground_states.index(sentence[i+1])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    # cluster_label2 = self.list_of_abstract_states[index2]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                    cluster_label2 = self.get_abstract_state(sentence[i + 1])
                    if not cluster_label1 == cluster_label2:
                        transition_mask[cluster_label2, cluster_label1, cluster_label2] = 1
                        # transition[cluster_label2, cluster_label1, cluster_label2] = 1
                        # transition[cluster_label2, cluster_label1, cluster_label1] = 0.2
                        # transition[cluster_label1, cluster_label2, cluster_label1] = 1
                        # rewards[cluster_label2, cluster_label1, cluster_label2] = -1
                else:
                    # index1 = self.list_of_ground_states.index(sentence[i])
                    # cluster_label1 = self.list_of_abstract_states[index1]
                    cluster_label1 = self.get_abstract_state(sentence[i])
                state_in_tuple = eval(sentence[i])
                if state_in_tuple == (self.goal[0], self.goal[1], 1, 1, 1):
                    transition[-1, cluster_label1, -1] = 1
                    rewards[-1, cluster_label1, -1] = 3000  # to comment when highest value is 0
            transition = transition + alpha * (1 - transition) * transition_mask
            # transition = transition + beta * (0 - transition) * (1 - transition_mask)
        # transition[-1, -1, -1] = 1  #when highest value is 0, other max(list_of_values) report error of empty sequence
        self.num_abstract_states = num_abstract_states
        self.transition = transition
        self.rewards = rewards
        end = time.time()
        print("time of set_transition_and_rewards_stochastic:", end - start)

    def solve_amdp(self, synchronous=0, monitor=0):
        values = np.zeros(self.num_abstract_states)
        if synchronous:
            values2 = copy.deepcopy(values)
        print("len(values):", len(values))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    if self.transition[a, i, a] != 0:  #  when highest value is 0
                        value = self.transition[a, i, a] * (
                            self.rewards[a, i, a] + 0.99 * values[a]
                        )
                        list_of_values.append(value)
                if synchronous:
                    values2[i] = max(list_of_values)
                    delta = max(delta, abs(v - values2[i]))
                else:
                    values[i] = max(list_of_values)
                    delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
            if synchronous:
                values = copy.deepcopy(values2)
            if monitor:
                self.plot_current_values(self.env, values)  # plot current values
        # print(values)
        # values -= min(values[:-1])
        # to comment when highest value is 0, this also helps reduce the effect of negative shapings

        self.values_of_abstract_states = values
        # print("self.values_of_abstract_states:")
        # print(self.values_of_abstract_states)
        # print(len(self.list_of_abstract_states), len(self.values_of_abstract_states))
        self.dict_as_v = dict(
            zip(
                (str(i) for i in self.list_of_abstract_states),
                self.values_of_abstract_states,
            )
        )
        print("self.dict_as_v:")
        print(self.dict_as_v)

    def solve_amdp_asynchronous(self):
        values = np.zeros(self.num_abstract_states)
        print("len(values):", len(values))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    value = self.transition[a, i, a] * (self.rewards[a, i, a] + 0.99 * values[a])
                    list_of_values.append(value)
                values[i] = max(list_of_values)
                delta = max(delta, abs(v - values[i]))
            print("delta:", delta)
            # self.plot_current_values(self.env, values)            # plot current values
        # print(V)
        values -= min(values[:-1])
        self.values_of_abstract_states = values
        # print("self.values_of_abstract_states:")
        # print(self.values_of_abstract_states)
        # print(len(self.list_of_abstract_states), len(self.values_of_abstract_states))
        self.dict_as_v = dict(
            zip(
                (str(i) for i in self.list_of_abstract_states),
                self.values_of_abstract_states,
            )
        )
        print("self.dict_as_v:")
        pprint(self.dict_as_v)

    def solve_amdp_synchronous(self):
        values = np.zeros(self.num_abstract_states)
        values2 = copy.deepcopy(values)
        print("len(values):", len(values))
        delta = 0.2
        theta = 0.1
        print("Value Iteration delta values:")
        while delta > theta:
            delta = 0
            for i in range(0, len(values)):
                v = values[i]
                list_of_values = []
                for a in range(len(values)):
                    value = self.transition[a, i, a] * (self.rewards[a, i, a] + 0.99 * values[a])
                    list_of_values.append(value)
                values2[i] = max(list_of_values)
                delta = max(delta, abs(v - values2[i]))
            print("delta:", delta)
            values = copy.deepcopy(values2)
            # self.plot_current_values(self.env, values)            # plot current values
        # print(V)
        values -= min(values[:-1])
        self.values_of_abstract_states = values
        # print("self.values_of_abstract_states:")
        # print(self.values_of_abstract_states)
        # print(len(self.list_of_abstract_states), len(self.values_of_abstract_states))
        self.dict_as_v = dict(
            zip(
                (str(i) for i in self.list_of_abstract_states),
                self.values_of_abstract_states,
            )
        )
        print("self.dict_as_v:")
        pprint(self.dict_as_v)

    def plot_current_values(self, env, values, plot_label=1):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5 * 3, 4 * 4))
        my_cmap = copy.copy(plt.cm.get_cmap("hot"))
        vmax = np.amax(values)
        vmin = 1500
        # vmin = 0
        my_cmap.set_under("grey")
        my_cmap.set_bad("lime")
        my_cmap.set_over("dodgerblue")
        asp = "auto"
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    plate = []
                    plate2 = []
                    for i in range(env.size[0]):
                        row = []
                        row2 = []
                        for j in range(env.size[1]):
                            current_coord = (i, j)
                            current_state = (i, j, k, l, m)
                            if current_state in env.valid_states:
                                a_state = self.get_abstract_state(current_state)
                                if current_state == env.start_state:
                                    row.append(vmax)
                                    row2.append(str(a_state))
                                elif current_coord == env.goal:
                                    row.append(vmax + 1)
                                    row2.append(str(a_state))
                                else:
                                    v = values[a_state]
                                    row.append(v)
                                    row2.append(str(a_state))
                            elif str([i, j]) in env.walls:
                                row.append(vmin)
                                row2.append("w")
                            elif env.flags.index((i, j)) == 0 and k == 0:
                                row.append(np.nan)
                                row2.append("f")
                            elif env.flags.index((i, j)) == 1 and l == 0:
                                row.append(np.nan)
                                row2.append("f")
                            elif env.flags.index((i, j)) == 2 and m == 0:
                                row.append(np.nan)
                                row2.append("f")

                        plate.append(row)
                        plate2.append(row2)
                    if k == 0 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 11)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 0 and m == 0:
                        ax = fig.add_subplot(4, 3, 7)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 8)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 9)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 1 and m == 0:
                        ax = fig.add_subplot(4, 3, 4)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 0 and m == 1:
                        ax = fig.add_subplot(4, 3, 5)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 0 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 6)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    elif k == 1 and l == 1 and m == 1:
                        ax = fig.add_subplot(4, 3, 2)
                        im = ax.imshow(plate, vmin=vmin, vmax=vmax, aspect=asp, cmap=my_cmap)
                    if plot_label:
                        np_cluster_layout = np.array(plate2)
                        c = 0
                        for a_state_ in self.list_of_abstract_states:
                            coords = np.argwhere(np_cluster_layout == str(a_state_))
                            if len(coords) > 0:
                                if isinstance(a_state_, int):
                                    a_state_head = a_state_
                                elif isinstance(a_state_, list):
                                    if a_state_[0].isdigit():
                                        a_state_head = a_state_[0]
                                    else:
                                        a_state_head = f"{a_state_[0][1]}-{a_state_[0][4]}"
                                mean = np.mean(coords, axis=0)
                                c += 1
                                v_ = round(values[a_state_])
                                ax.text(
                                    mean[1],
                                    mean[0],
                                    f"{str(a_state_head)}\n{str(v_)}",
                                    horizontalalignment="center",
                                    verticalalignment="center",
                                    fontsize=10,
                                    fontweight="semibold",
                                    color="k",
                                )
                                # ax.text(mean[1], mean[0], str(v_), horizontalalignment='center', verticalalignment='center',
                                #         fontsize=10, fontweight='semibold', color='k')
                    ax.set_title(f"{k}-{l}-{m}-c{c}", fontsize=15, fontweight="semibold")
        # fig.subplots_adjust(right=0.85)
        # cax = fig.add_axes([0.9, 0.23, 0.03, 0.5])
        # fig.colorbar(im, cax=cax)
        fig.show()

    def get_value(self, astate):
        assert isinstance(astate, int), "astate has to be int"
        value = self.values_of_abstract_states[astate]
        # print("value:",value)
        return value


class ObsPreprocessWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return str(obs)
